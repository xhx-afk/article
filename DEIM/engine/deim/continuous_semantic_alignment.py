"""SQ-Align V3 continuous mask-supported query alignment utilities.

All boxes accepted by this module are normalized ``cxcywh`` tensors.  Target
construction is deliberately detached from predicted boxes; semantic losses
must not perturb the detector's box regression path in the first V3 version.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


TensorDict = Dict[str, torch.Tensor]
MatchedIndices = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def _group_count(channels: int) -> int:
    """Choose a small GroupNorm group count that divides ``channels``."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DefectnessHead(nn.Module):
    """Predict binary defect logits ``[B,1,H,W]`` from the finest feature."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class SemanticPixelProjection(nn.Module):
    """Project encoder pixels to semantic embeddings ``[B,D,Hs,Ws]``."""

    def __init__(self, in_channels: int = 256, embed_dim: int = 64, downsample: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=2 if downsample else 1,
                padding=1,
                groups=embed_dim,
                bias=False,
            ),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class QuerySemanticProjection(nn.Module):
    """Project decoder queries ``[B,Q,C]`` to normalized ``[B,Q,D]``."""

    def __init__(self, query_dim: int = 256, embed_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(query_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(query).float(), dim=-1, eps=1e-6).to(query.dtype)


def semantic_mask_statistics(mask_logits: torch.Tensor) -> torch.Tensor:
    """Extract ten robust statistics from ``[B,Q,H,W]`` query-mask logits."""
    if mask_logits.ndim != 4:
        raise ValueError(f"expected mask_logits [B,Q,H,W], got {mask_logits.shape}")
    probabilities = mask_logits.float().sigmoid()
    flat = probabilities.flatten(-2)
    pixel_count = flat.shape[-1]
    if pixel_count == 0:
        return probabilities.new_zeros((*probabilities.shape[:2], 10))

    top10_count = max(1, int(math.ceil(pixel_count * 0.10)))
    top25_count = max(1, int(math.ceil(pixel_count * 0.25)))
    height, width = probabilities.shape[-2:]
    center_h0, center_h1 = height // 4, max(height // 4 + 1, height - height // 4)
    center_w0, center_w1 = width // 4, max(width // 4 + 1, width - width // 4)
    center_mask = torch.zeros((height, width), device=probabilities.device, dtype=torch.bool)
    center_mask[center_h0:center_h1, center_w0:center_w1] = True
    center_values = probabilities[..., center_mask]
    border_values = probabilities[..., ~center_mask]
    center_mean = center_values.mean(-1) if center_values.shape[-1] else flat.mean(-1)
    border_mean = border_values.mean(-1) if border_values.shape[-1] else flat.mean(-1)
    stats = torch.stack(
        (
            flat.mean(-1),
            flat.amax(-1),
            flat.std(-1, unbiased=False),
            flat.topk(top10_count, dim=-1, sorted=False).values.mean(-1),
            flat.topk(top25_count, dim=-1, sorted=False).values.mean(-1),
            (flat >= 0.50).float().mean(-1),
            (flat >= 0.75).float().mean(-1),
            center_mean,
            border_mean,
            center_mean - border_mean,
        ),
        dim=-1,
    )
    return torch.nan_to_num(stats, nan=0.0, posinf=1.0, neginf=-1.0)


class SemanticQualityHead(nn.Module):
    """Learn a semantic-quality logit from mask statistics and query features."""

    def __init__(
        self,
        query_dim: int = 256,
        query_projection_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.Linear(query_dim, query_projection_dim),
            nn.LayerNorm(query_projection_dim),
            nn.SiLU(inplace=True),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(10 + query_projection_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mask_logits: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("mask logits and query features must share [B,Q]")
        stats = semantic_mask_statistics(mask_logits)
        query_stats = self.query_projection(query_features.float())
        return self.quality_mlp(torch.cat((stats, query_stats), dim=-1))


def pairwise_box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return float32 pairwise IoU for normalized ``cxcywh`` boxes."""
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError(f"expected [N,4] boxes, got {boxes1.shape} and {boxes2.shape}")
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    b1_xy1 = boxes1[:, :2] - boxes1[:, 2:].clamp_min(0) / 2
    b1_xy2 = boxes1[:, :2] + boxes1[:, 2:].clamp_min(0) / 2
    b2_xy1 = boxes2[:, :2] - boxes2[:, 2:].clamp_min(0) / 2
    b2_xy2 = boxes2[:, :2] + boxes2[:, 2:].clamp_min(0) / 2
    inter_xy1 = torch.maximum(b1_xy1[:, None], b2_xy1[None])
    inter_xy2 = torch.minimum(b1_xy2[:, None], b2_xy2[None])
    intersection = (inter_xy2 - inter_xy1).clamp_min(0).prod(-1)
    area1 = (b1_xy2 - b1_xy1).clamp_min(0).prod(-1)
    area2 = (b2_xy2 - b2_xy1).clamp_min(0).prod(-1)
    union = area1[:, None] + area2[None] - intersection
    return torch.nan_to_num(intersection / union.clamp_min(1e-7), nan=0.0).clamp_(0.0, 1.0)


def _matched_query_mask(
    shape: Tuple[int, int],
    matched_indices: MatchedIndices | None,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    if matched_indices is not None:
        for batch_index, (source_indices, _) in enumerate(matched_indices):
            mask[batch_index, source_indices.to(device=device)] = True
    return mask


def build_best_gt_assignment(
    pred_boxes: torch.Tensor,
    targets: Sequence[TensorDict],
    matched_indices: MatchedIndices | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return detached associated IoU and GT index, each shaped ``[B,Q]``.

    Hungarian assignments override the unconstrained best-IoU association.
    Images without GT use IoU zero and GT index ``-1``.
    """
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"expected pred_boxes [B,Q,4], got {pred_boxes.shape}")
    if len(targets) != pred_boxes.shape[0]:
        raise ValueError("target and prediction batch sizes differ")
    if matched_indices is not None and len(matched_indices) != pred_boxes.shape[0]:
        raise ValueError("matched_indices and prediction batch sizes differ")

    batch_size, query_count = pred_boxes.shape[:2]
    best_iou = torch.zeros((batch_size, query_count), device=pred_boxes.device, dtype=torch.float32)
    best_gt_index = torch.full(
        (batch_size, query_count), -1, device=pred_boxes.device, dtype=torch.long
    )
    for batch_index, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if gt_boxes is None or gt_boxes.numel() == 0 or query_count == 0:
            continue
        gt_boxes = gt_boxes.detach().to(device=pred_boxes.device)
        ious = pairwise_box_iou_cxcywh(pred_boxes[batch_index].detach(), gt_boxes)
        best_iou[batch_index], best_gt_index[batch_index] = ious.max(dim=1)
        if matched_indices is not None:
            source, target_index = matched_indices[batch_index]
            source = source.to(device=pred_boxes.device, dtype=torch.long)
            target_index = target_index.to(device=pred_boxes.device, dtype=torch.long)
            if source.numel():
                best_gt_index[batch_index, source] = target_index
                best_iou[batch_index, source] = ious[source, target_index]
    return best_iou, best_gt_index


def _normalized_cxcywh_to_map_xyxy(
    boxes: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    cx, cy, box_width, box_height = boxes.unbind(-1)
    x1 = ((cx - box_width / 2) * width).clamp(0.0, float(width))
    y1 = ((cy - box_height / 2) * height).clamp(0.0, float(height))
    x2 = ((cx + box_width / 2) * width).clamp(0.0, float(width))
    y2 = ((cy + box_height / 2) * height).clamp(0.0, float(height))
    epsilon = 1e-4
    x2 = torch.maximum(x2, (x1 + epsilon).clamp_max(float(width)))
    y2 = torch.maximum(y2, (y1 + epsilon).clamp_max(float(height)))
    return torch.stack((x1, y1, x2, y2), dim=-1)


def roi_align_query_pixel_features(
    pixel_embeddings: torch.Tensor,
    pred_boxes: torch.Tensor,
    output_size: int = 7,
    sampling_ratio: int = 2,
    aligned: bool = True,
    detach_boxes: bool = True,
) -> torch.Tensor:
    """ROI-align pixels to ``[B,Q,D,output_size,output_size]``."""
    if pixel_embeddings.ndim != 4 or pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError("expected pixel embeddings [B,D,H,W] and boxes [B,Q,4]")
    if pixel_embeddings.shape[0] != pred_boxes.shape[0]:
        raise ValueError("pixel embeddings and boxes must share batch size")
    if int(output_size) <= 0:
        raise ValueError("output_size must be positive")
    batch_size, query_count = pred_boxes.shape[:2]
    embed_dim = pixel_embeddings.shape[1]
    if batch_size == 0 or query_count == 0:
        return pixel_embeddings.new_zeros(
            (batch_size, query_count, embed_dim, int(output_size), int(output_size))
        )
    boxes = pred_boxes.detach() if detach_boxes else pred_boxes
    height, width = pixel_embeddings.shape[-2:]
    boxes_xyxy = _normalized_cxcywh_to_map_xyxy(boxes.float(), height, width)
    batch_indices = torch.arange(batch_size, device=boxes.device, dtype=boxes_xyxy.dtype)
    batch_indices = batch_indices[:, None].expand(batch_size, query_count).reshape(-1, 1)
    rois = torch.cat((batch_indices, boxes_xyxy.reshape(-1, 4)), dim=1)
    aligned_features = roi_align(
        pixel_embeddings.float(),
        rois,
        output_size=(int(output_size), int(output_size)),
        spatial_scale=1.0,
        sampling_ratio=int(sampling_ratio),
        aligned=bool(aligned),
    )
    return aligned_features.reshape(batch_size, query_count, embed_dim, int(output_size), int(output_size))


def query_conditioned_mask_logits(
    query_embeddings: torch.Tensor, roi_pixel_features: torch.Tensor
) -> torch.Tensor:
    """Compute query-conditioned ROI mask logits ``[B,Q,H,W]``."""
    if query_embeddings.ndim != 3 or roi_pixel_features.ndim != 5:
        raise ValueError("expected query embeddings [B,Q,D] and ROI features [B,Q,D,H,W]")
    if query_embeddings.shape != roi_pixel_features.shape[:3]:
        raise ValueError("query embeddings and ROI pixel features have incompatible shapes")
    scale = math.sqrt(max(query_embeddings.shape[-1], 1))
    return torch.einsum(
        "bqd,bqdhw->bqhw", query_embeddings.float(), roi_pixel_features.float()
    ) / scale


def pool_gt_instance_roi_masks(
    targets: Sequence[TensorDict],
    pred_boxes: torch.Tensor,
    best_gt_index: torch.Tensor,
    best_iou: torch.Tensor,
    matched_mask: torch.Tensor | None = None,
    output_size: int = 7,
    association_iou_threshold: float = 0.10,
    detach_boxes: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return soft GT ROI masks ``[B,Q,H,W]`` and validity ``[B,Q]``.

    Far-background queries receive an all-zero valid target.  Associated
    queries whose instance masks are unavailable or invalid are ignored.
    """
    if pred_boxes.shape[:2] != best_gt_index.shape or best_iou.shape != best_gt_index.shape:
        raise ValueError("assignment tensors must match pred_boxes [B,Q]")
    batch_size, query_count = pred_boxes.shape[:2]
    roi_targets = pred_boxes.new_zeros(
        (batch_size, query_count, int(output_size), int(output_size)), dtype=torch.float32
    )
    if matched_mask is None:
        matched_mask = torch.zeros_like(best_iou, dtype=torch.bool)
    associated = matched_mask | (best_iou >= float(association_iou_threshold))
    valid = ~associated
    boxes = pred_boxes.detach() if detach_boxes else pred_boxes

    for batch_index, target in enumerate(targets):
        selected_queries = torch.nonzero(associated[batch_index], as_tuple=False).flatten()
        if selected_queries.numel() == 0:
            continue
        masks = target.get("masks")
        mask_valid = target.get("mask_valid")
        if masks is None or masks.ndim != 3 or masks.shape[0] == 0:
            continue
        masks = masks.to(device=pred_boxes.device, dtype=torch.float32)
        if mask_valid is None:
            mask_valid = torch.ones((masks.shape[0],), device=pred_boxes.device, dtype=torch.bool)
        else:
            mask_valid = mask_valid.to(device=pred_boxes.device, dtype=torch.bool)
        gt_index = best_gt_index[batch_index, selected_queries]
        in_range = (gt_index >= 0) & (gt_index < masks.shape[0])
        query_valid = in_range.clone()
        if in_range.any():
            query_valid[in_range] &= mask_valid[gt_index[in_range]]
        chosen_queries = selected_queries[query_valid]
        if chosen_queries.numel() == 0:
            continue
        chosen_gt = best_gt_index[batch_index, chosen_queries]
        height, width = masks.shape[-2:]
        boxes_xyxy = _normalized_cxcywh_to_map_xyxy(
            boxes[batch_index, chosen_queries].float(), height, width
        )
        rois = torch.cat((chosen_gt[:, None].to(boxes_xyxy.dtype), boxes_xyxy), dim=1)
        pooled = roi_align(
            masks[:, None],
            rois,
            output_size=(int(output_size), int(output_size)),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )[:, 0]
        roi_targets[batch_index, chosen_queries] = pooled.clamp(0.0, 1.0)
        valid[batch_index, chosen_queries] = True
    return roi_targets, valid


def _box_bounds(boxes: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, ...]:
    xyxy = _normalized_cxcywh_to_map_xyxy(boxes.float(), height, width)
    x1 = xyxy[:, 0].floor().long().clamp(0, width)
    y1 = xyxy[:, 1].floor().long().clamp(0, height)
    x2 = xyxy[:, 2].ceil().long().clamp(0, width)
    y2 = xyxy[:, 3].ceil().long().clamp(0, height)
    x2 = torch.maximum(x2, (x1 + 1).clamp_max(width))
    y2 = torch.maximum(y2, (y1 + 1).clamp_max(height))
    return x1, y1, x2, y2


def _integral_box_sum(
    integral: torch.Tensor, gt_indices: torch.Tensor, boxes: torch.Tensor, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = _box_bounds(boxes, height, width)
    values = (
        integral[gt_indices, y2, x2]
        - integral[gt_indices, y1, x2]
        - integral[gt_indices, y2, x1]
        + integral[gt_indices, y1, x1]
    )
    areas = ((x2 - x1) * (y2 - y1)).float()
    return values, areas


def compute_continuous_mask_support(
    instance_masks: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_indices: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute CMSQ targets for associated queries without per-query crops.

    Args:
        instance_masks: GT masks ``[N,H,W]``.
        pred_boxes: detached predicted boxes ``[Q,4]``.
        gt_boxes: GT boxes ``[N,4]``.
        gt_indices: associated GT index for each query ``[Q]``.
    """
    if instance_masks.ndim != 3 or pred_boxes.ndim != 2 or gt_boxes.ndim != 2:
        raise ValueError("expected masks [N,H,W], pred boxes [Q,4], and GT boxes [N,4]")
    if pred_boxes.shape[0] != gt_indices.shape[0]:
        raise ValueError("gt_indices must have one entry per predicted box")
    query_count = pred_boxes.shape[0]
    if query_count == 0:
        return pred_boxes.new_zeros((0,), dtype=torch.float32)
    result = pred_boxes.new_zeros((query_count,), dtype=torch.float32)
    valid = (gt_indices >= 0) & (gt_indices < instance_masks.shape[0])
    if not valid.any():
        return result

    masks = instance_masks.to(device=pred_boxes.device, dtype=torch.float32)
    height, width = masks.shape[-2:]
    integral = F.pad(masks.cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    selected_gt = gt_indices[valid].long()
    selected_pred = pred_boxes[valid].detach().float()
    selected_gt_boxes = gt_boxes.to(device=pred_boxes.device)[selected_gt].detach().float()
    pred_intersection, pred_area = _integral_box_sum(
        integral, selected_gt, selected_pred, height, width
    )
    gt_intersection, gt_area = _integral_box_sum(
        integral, selected_gt, selected_gt_boxes, height, width
    )
    mask_area = masks.flatten(1).sum(-1)[selected_gt]
    coverage = pred_intersection / mask_area.clamp_min(float(eps))
    pred_density = pred_intersection / pred_area.clamp_min(float(eps))
    gt_density = gt_intersection / gt_area.clamp_min(float(eps))
    relative_density = (pred_density / gt_density.clamp_min(float(eps))).clamp(0.0, 1.0)
    support = torch.sqrt((coverage.clamp(0.0, 1.0) * relative_density).clamp_min(0.0))
    result[valid] = torch.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return result


def build_semantic_quality_targets(
    pred_boxes: torch.Tensor,
    targets: Sequence[TensorDict],
    best_gt_index: torch.Tensor,
    best_iou: torch.Tensor,
    matched_indices: MatchedIndices | None = None,
    association_iou_threshold: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CMSQ targets and mutually-exclusive matched/near/far masks.

    Returns ``target, valid, matched, near_unmatched, far_background``, each
    shaped ``[B,Q]``.  Invalid associated instance masks are ignored.
    """
    batch_size, query_count = pred_boxes.shape[:2]
    matched = _matched_query_mask((batch_size, query_count), matched_indices, pred_boxes.device)
    near = (~matched) & (best_iou >= float(association_iou_threshold))
    far = (~matched) & ~near
    target_values = pred_boxes.new_zeros((batch_size, query_count), dtype=torch.float32)
    valid = far.clone()

    for batch_index, target in enumerate(targets):
        associated = matched[batch_index] | near[batch_index]
        query_indices = torch.nonzero(associated, as_tuple=False).flatten()
        if query_indices.numel() == 0:
            continue
        masks = target.get("masks")
        gt_boxes = target.get("boxes")
        mask_valid = target.get("mask_valid")
        if masks is None or masks.ndim != 3 or gt_boxes is None or gt_boxes.numel() == 0:
            continue
        masks = masks.to(device=pred_boxes.device)
        gt_boxes = gt_boxes.to(device=pred_boxes.device)
        if mask_valid is None:
            mask_valid = torch.ones((masks.shape[0],), device=pred_boxes.device, dtype=torch.bool)
        else:
            mask_valid = mask_valid.to(device=pred_boxes.device, dtype=torch.bool)
        associated_gt = best_gt_index[batch_index, query_indices]
        in_range = (associated_gt >= 0) & (associated_gt < masks.shape[0])
        query_valid = in_range.clone()
        if in_range.any():
            query_valid[in_range] &= mask_valid[associated_gt[in_range]]
        chosen_queries = query_indices[query_valid]
        if chosen_queries.numel() == 0:
            continue
        chosen_gt = best_gt_index[batch_index, chosen_queries]
        target_values[batch_index, chosen_queries] = compute_continuous_mask_support(
            masks,
            pred_boxes[batch_index, chosen_queries].detach(),
            gt_boxes,
            chosen_gt,
        )
        valid[batch_index, chosen_queries] = True
    return target_values, valid, matched, near, far


def semantic_suppression_gate(
    semantic_quality_logits: torch.Tensor,
    gate_lambda: float = 0.20,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return a suppression-only gate with values in ``[1-lambda, 1]``."""
    if not 0.0 <= float(gate_lambda) <= 1.0:
        raise ValueError("gate_lambda must be in [0,1]")
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive")
    quality = semantic_quality_logits.float().sigmoid()
    gate = 1.0 - float(gate_lambda) * (1.0 - quality).pow(float(gamma))
    return gate.clamp(min=1.0 - float(gate_lambda), max=1.0)


CandidateGroups = List[List[Dict[str, torch.Tensor]]]


def build_per_gt_candidate_groups(
    class_scores: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: Sequence[TensorDict],
    matched_indices: MatchedIndices,
    iou_threshold: float = 0.30,
    max_candidates_per_gt: int = 10,
) -> CandidateGroups:
    """Build per-GT near-candidate groups using detached selection."""
    if class_scores.ndim != 3 or pred_boxes.shape[:2] != class_scores.shape[:2]:
        raise ValueError("expected class scores [B,Q,C] and boxes [B,Q,4]")
    groups: CandidateGroups = []
    detached_scores = class_scores.detach()
    detached_boxes = pred_boxes.detach()
    for batch_index, target in enumerate(targets):
        image_groups: List[Dict[str, torch.Tensor]] = []
        gt_boxes = target.get("boxes")
        labels = target.get("labels")
        if gt_boxes is None or labels is None or gt_boxes.numel() == 0:
            groups.append(image_groups)
            continue
        gt_boxes = gt_boxes.to(device=pred_boxes.device)
        labels = labels.to(device=pred_boxes.device, dtype=torch.long)
        ious = pairwise_box_iou_cxcywh(detached_boxes[batch_index], gt_boxes)
        matched_source, matched_target = matched_indices[batch_index]
        matched_source = matched_source.to(device=pred_boxes.device, dtype=torch.long)
        matched_target = matched_target.to(device=pred_boxes.device, dtype=torch.long)
        for gt_index in range(gt_boxes.shape[0]):
            candidate_indices = torch.nonzero(
                ious[:, gt_index] >= float(iou_threshold), as_tuple=False
            ).flatten()
            if candidate_indices.numel() == 0:
                continue
            label = labels[gt_index]
            true_scores = detached_scores[batch_index, candidate_indices, label]
            keep_count = min(max(int(max_candidates_per_gt), 1), candidate_indices.numel())
            candidate_indices = candidate_indices[true_scores.topk(keep_count).indices]
            matched_for_gt = matched_source[matched_target == gt_index]
            matched_for_gt = matched_for_gt[
                ious[matched_for_gt, gt_index] >= float(iou_threshold)
            ]
            if matched_for_gt.numel():
                anchor = matched_for_gt[ious[matched_for_gt, gt_index].argmax()]
                if not (candidate_indices == anchor).any():
                    candidate_indices[-1] = anchor
            else:
                anchor = torch.tensor(-1, device=pred_boxes.device, dtype=torch.long)
            image_groups.append(
                {
                    "gt_index": torch.tensor(gt_index, device=pred_boxes.device, dtype=torch.long),
                    "label": label,
                    "candidate_indices": candidate_indices,
                    "candidate_ious": ious[candidate_indices, gt_index],
                    "anchor_index": anchor,
                }
            )
        groups.append(image_groups)
    return groups


def near_gt_candidate_rank_loss(
    rank_scores: torch.Tensor,
    candidate_groups: CandidateGroups,
    iou_gap: float = 0.10,
    candidate_margin: float = 0.03,
    class_margin: float = 0.05,
    temperature: float = 0.10,
    topk_per_gt: int = 3,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Rank localization and class candidates; gradients touch scores only."""
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    zero = rank_scores.sum() * 0.0
    localization_losses: List[torch.Tensor] = []
    class_losses: List[torch.Tensor] = []
    violations: List[torch.Tensor] = []
    anchor_margins: List[torch.Tensor] = []
    group_counts: List[torch.Tensor] = []

    for batch_index, image_groups in enumerate(candidate_groups):
        for group in image_groups:
            indices = group["candidate_indices"]
            anchor = group["anchor_index"]
            if indices.numel() < 2 or int(anchor.item()) < 0:
                continue
            label = group["label"]
            ious = group["candidate_ious"]
            scores = rank_scores[batch_index, indices, label]
            pair_mask = ious[:, None] > ious[None, :] + float(iou_gap)
            higher, lower = torch.nonzero(pair_mask, as_tuple=True)
            if higher.numel():
                pair_violation = scores[lower] - scores[higher] + float(candidate_margin)
                keep = min(max(int(topk_per_gt), 1), pair_violation.numel())
                hard = pair_violation.detach().topk(keep).indices
                localization_losses.append(
                    F.softplus(pair_violation[hard] / float(temperature)).mean()
                )
                violations.append((pair_violation.detach() > 0).float().mean())

            anchor_true = rank_scores[batch_index, anchor, label]
            if rank_scores.shape[-1] > 1:
                other_scores = rank_scores[batch_index, anchor].clone()
                other_scores[label] = torch.finfo(other_scores.dtype).min
                hardest_other = other_scores.max()
                class_losses.append(
                    F.softplus(
                        (hardest_other - anchor_true + float(class_margin)) / float(temperature)
                    )
                )
            non_anchor = indices[indices != anchor]
            if non_anchor.numel():
                hardest_candidate = rank_scores[batch_index, non_anchor, label].max()
                anchor_margins.append((anchor_true - hardest_candidate).detach())
            group_counts.append(rank_scores.new_tensor(float(indices.numel())))

    parts: List[torch.Tensor] = []
    if localization_losses:
        parts.append(torch.stack(localization_losses).mean())
    if class_losses:
        parts.append(torch.stack(class_losses).mean())
    loss = torch.stack(parts).sum() if parts else zero
    details = {
        "gt_count": rank_scores.new_tensor(float(len(group_counts))),
        "candidate_count_mean": torch.stack(group_counts).mean() if group_counts else zero.detach(),
        "violation_ratio": torch.stack(violations).mean() if violations else zero.detach(),
        "anchor_margin": torch.stack(anchor_margins).mean() if anchor_margins else zero.detach(),
    }
    if return_details:
        return loss, details
    return loss


def build_union_defect_target(
    targets: Sequence[TensorDict],
    output_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``[B,1,Hd,Wd]`` union targets with max-preserving pooling."""
    unions: List[torch.Tensor] = []
    for target in targets:
        masks = target.get("masks")
        valid = target.get("mask_valid")
        if masks is None or masks.ndim != 3:
            source_size = target.get("orig_size")
            height = int(source_size[1]) if source_size is not None else output_size[0]
            width = int(source_size[0]) if source_size is not None else output_size[1]
            union = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
        else:
            masks = masks.to(device=device)
            if valid is None:
                valid = torch.ones((masks.shape[0],), device=device, dtype=torch.bool)
            else:
                valid = valid.to(device=device, dtype=torch.bool)
            selected = masks[valid]
            if selected.shape[0] == 0:
                union = torch.zeros((1, 1, *masks.shape[-2:]), device=device, dtype=torch.float32)
            else:
                union = selected.bool().any(dim=0, keepdim=True).unsqueeze(0).float()
        unions.append(F.adaptive_max_pool2d(union, output_size))
    if not unions:
        return torch.zeros((0, 1, *output_size), device=device, dtype=dtype)
    return torch.cat(unions, dim=0).to(dtype=dtype)


__all__ = [
    "DefectnessHead",
    "SemanticPixelProjection",
    "QuerySemanticProjection",
    "SemanticQualityHead",
    "semantic_mask_statistics",
    "pairwise_box_iou_cxcywh",
    "build_best_gt_assignment",
    "roi_align_query_pixel_features",
    "query_conditioned_mask_logits",
    "pool_gt_instance_roi_masks",
    "compute_continuous_mask_support",
    "build_semantic_quality_targets",
    "semantic_suppression_gate",
    "build_per_gt_candidate_groups",
    "near_gt_candidate_rank_loss",
    "build_union_defect_target",
]
