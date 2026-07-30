"""SQ-Align V2 heads, targets, score fusion, and ranking utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class DefectnessHead(nn.Module):
    """Predict a binary defectness logit map from the finest encoder feature."""

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
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class QueryLocalizationQualityHead(nn.Module):
    """Predict one all-query localization-quality logit per decoder query."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.layers(query)


def pairwise_box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized ``cxcywh`` boxes, computed in float32."""
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


def build_all_query_best_iou_target(
    pred_boxes: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Build detached best-IoU targets for every query, with shape ``[B,Q]``."""
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"expected pred_boxes [B,Q,4], got {pred_boxes.shape}")
    if len(targets) != pred_boxes.shape[0]:
        raise ValueError(f"target batch {len(targets)} != prediction batch {pred_boxes.shape[0]}")

    detached_boxes = pred_boxes.detach()
    batch_targets: List[torch.Tensor] = []
    for batch_index, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if gt_boxes is None or gt_boxes.numel() == 0:
            batch_targets.append(
                torch.zeros((pred_boxes.shape[1],), device=pred_boxes.device, dtype=torch.float32)
            )
            continue
        gt_boxes = gt_boxes.detach().to(device=pred_boxes.device)
        iou = pairwise_box_iou_cxcywh(detached_boxes[batch_index], gt_boxes)
        batch_targets.append(iou.amax(dim=1))

    if not batch_targets:
        return pred_boxes.new_zeros((0, pred_boxes.shape[1]), dtype=torch.float32)
    return torch.nan_to_num(torch.stack(batch_targets), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _normalized_cxcywh_to_map_xyxy(
    pred_boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    cx, cy, box_w, box_h = pred_boxes.unbind(-1)
    x1 = (cx - box_w / 2) * width
    y1 = (cy - box_h / 2) * height
    x2 = (cx + box_w / 2) * width
    y2 = (cy + box_h / 2) * height
    x1 = x1.clamp(0.0, float(width))
    y1 = y1.clamp(0.0, float(height))
    x2 = x2.clamp(0.0, float(width))
    y2 = y2.clamp(0.0, float(height))
    epsilon = 1e-4
    x2 = torch.maximum(x2, (x1 + epsilon).clamp_max(float(width)))
    y2 = torch.maximum(y2, (y1 + epsilon).clamp_max(float(height)))
    x1 = torch.minimum(x1, x2)
    y1 = torch.minimum(y1, y2)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def pool_query_semantic_evidence(
    defect_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    output_size: int = 7,
    topk_ratio: float = 0.20,
    detach_boxes: bool = True,
) -> torch.Tensor:
    """ROI-align defect probabilities and return top-k-mean evidence ``[B,Q,1]``."""
    if defect_logits.ndim != 4 or defect_logits.shape[1] != 1:
        raise ValueError(f"expected defect_logits [B,1,H,W], got {defect_logits.shape}")
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"expected pred_boxes [B,Q,4], got {pred_boxes.shape}")
    if defect_logits.shape[0] != pred_boxes.shape[0]:
        raise ValueError("defect logits and boxes must have the same batch size")
    if int(output_size) <= 0:
        raise ValueError("output_size must be positive")
    if not 0.0 < float(topk_ratio) <= 1.0:
        raise ValueError("topk_ratio must be in (0, 1]")

    batch_size, query_count = pred_boxes.shape[:2]
    if batch_size == 0 or query_count == 0:
        return defect_logits.sum().reshape(1, 1, 1)[:batch_size, :query_count].expand(batch_size, query_count, 1)

    boxes = pred_boxes.detach() if detach_boxes else pred_boxes
    height, width = defect_logits.shape[-2:]
    boxes_xyxy = _normalized_cxcywh_to_map_xyxy(boxes.float(), height, width)
    batch_indices = torch.arange(batch_size, device=boxes.device, dtype=boxes_xyxy.dtype)
    batch_indices = batch_indices[:, None].expand(batch_size, query_count).reshape(-1, 1)
    rois = torch.cat((batch_indices, boxes_xyxy.reshape(-1, 4)), dim=1)

    # Float32 keeps CPU/CUDA AMP behavior stable while preserving gradients.
    roi_logits = roi_align(
        defect_logits.float(),
        rois,
        output_size=(int(output_size), int(output_size)),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    probabilities = roi_logits.sigmoid().flatten(1)
    pixel_count = probabilities.shape[1]
    k = min(pixel_count, max(1, int(math.ceil(pixel_count * float(topk_ratio)))))
    evidence = probabilities.topk(k, dim=1, sorted=False).values.mean(dim=1)
    return evidence.reshape(batch_size, query_count, 1)


def build_query_semantic_supervision(
    best_iou: torch.Tensor,
    matched_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    background_iou_threshold: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return targets, valid mask, matched-positive mask, and far-background mask."""
    if best_iou.ndim != 2 or len(matched_indices) != best_iou.shape[0]:
        raise ValueError("best_iou must be [B,Q] and match the index batch")
    positive_mask = torch.zeros_like(best_iou, dtype=torch.bool)
    for batch_index, (source_indices, _) in enumerate(matched_indices):
        positive_mask[batch_index, source_indices.to(best_iou.device)] = True
    far_background_mask = (~positive_mask) & (best_iou < float(background_iou_threshold))
    valid_mask = positive_mask | far_background_mask
    semantic_target = positive_mask.to(dtype=best_iou.dtype)
    return semantic_target, valid_mask, positive_mask, far_background_mask


def compose_dual_quality_scores(
    class_scores: torch.Tensor,
    pred_loc_quality: torch.Tensor | None = None,
    pred_sem_quality: torch.Tensor | None = None,
    loc_quality_power: float = 0.25,
    semantic_quality_power: float = 0.25,
) -> torch.Tensor:
    """Fuse class probabilities with localization and semantic query quality."""
    if class_scores.ndim != 3:
        raise ValueError(f"expected class_scores [B,Q,C], got {class_scores.shape}")
    scores = class_scores
    if pred_loc_quality is not None and float(loc_quality_power) != 0.0:
        if pred_loc_quality.shape != (*class_scores.shape[:2], 1):
            raise ValueError("pred_loc_quality must have shape [B,Q,1]")
        loc_quality = pred_loc_quality.sigmoid().to(dtype=scores.dtype)
        scores = scores * loc_quality.clamp_min(1e-6).pow(float(loc_quality_power))
    if pred_sem_quality is not None and float(semantic_quality_power) != 0.0:
        if pred_sem_quality.shape != (*class_scores.shape[:2], 1):
            raise ValueError("pred_sem_quality must have shape [B,Q,1]")
        semantic_quality = pred_sem_quality.to(dtype=scores.dtype).clamp(1e-6, 1.0)
        scores = scores * semantic_quality.pow(float(semantic_quality_power))
    return scores


def select_final_score_background(
    final_scores: torch.Tensor,
    best_iou: torch.Tensor,
    matched_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    topk: int = 5,
    iou_threshold: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top final-score unmatched far-background queries per image."""
    if final_scores.ndim != 3 or best_iou.shape != final_scores.shape[:2]:
        raise ValueError("final_scores must be [B,Q,C] and best_iou must be [B,Q]")
    if len(matched_indices) != final_scores.shape[0]:
        raise ValueError("matched_indices batch size mismatch")

    batch_size, query_count = best_iou.shape
    selected = torch.zeros_like(best_iou, dtype=torch.bool)
    counts = final_scores.new_zeros((batch_size,), dtype=torch.float32)
    detached_difficulty = final_scores.detach().amax(dim=-1)
    for batch_index, (source_indices, _) in enumerate(matched_indices):
        unmatched = torch.ones((query_count,), device=best_iou.device, dtype=torch.bool)
        unmatched[source_indices.to(best_iou.device)] = False
        candidates = torch.nonzero(
            unmatched & (best_iou[batch_index] < float(iou_threshold)), as_tuple=False
        ).flatten()
        k = min(max(int(topk), 0), int(candidates.numel()))
        if k:
            relative_indices = detached_difficulty[batch_index, candidates].topk(k).indices
            chosen = candidates[relative_indices]
            selected[batch_index, chosen] = True
            counts[batch_index] = float(k)
    return selected, counts


def final_score_ranking_loss(
    final_scores: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    matched_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    selected_background: torch.Tensor,
    margin: float = 0.05,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Pair weakest matched true-class scores with strongest selected background scores."""
    if selected_background.shape != final_scores.shape[:2]:
        raise ValueError("selected_background must have shape [B,Q]")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")

    per_image_losses: List[torch.Tensor] = []
    for batch_index, (source_indices, target_indices) in enumerate(matched_indices):
        if source_indices.numel() == 0 or not selected_background[batch_index].any():
            continue
        source_indices = source_indices.to(final_scores.device)
        target_indices = target_indices.to(final_scores.device)
        labels = targets[batch_index]["labels"].to(final_scores.device)[target_indices]
        positive_scores = final_scores[batch_index, source_indices, labels]
        background_scores = final_scores[batch_index, selected_background[batch_index]].amax(dim=-1)
        k = min(int(positive_scores.numel()), int(background_scores.numel()))
        if not k:
            continue
        weakest_positive = positive_scores.topk(k, largest=False).values
        strongest_background = background_scores.topk(k, largest=True).values
        per_image_losses.append(
            F.softplus(
                (strongest_background - weakest_positive + float(margin)) / float(temperature)
            ).mean()
        )
    if not per_image_losses:
        return final_scores.sum() * 0.0
    return torch.stack(per_image_losses).mean()


def build_union_defect_target(
    targets: Sequence[Dict[str, torch.Tensor]],
    output_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``[B,1,Hd,Wd]`` union-mask targets with max-preserving resize."""
    unions: List[torch.Tensor] = []
    for target in targets:
        masks = target.get("masks")
        valid = target.get("mask_valid")
        if masks is None or masks.ndim != 3:
            source_size = target.get("orig_size")
            # Detection targets store orig_size as [width, height].
            height = int(source_size[1]) if source_size is not None else output_size[0]
            width = int(source_size[0]) if source_size is not None else output_size[1]
            union = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
        else:
            masks = masks.to(device=device)
            if valid is None:
                valid = torch.ones((masks.shape[0],), device=device, dtype=torch.bool)
            else:
                valid = valid.to(device=device, dtype=torch.bool)
            selected_masks = masks[valid]
            if selected_masks.shape[0] == 0:
                union = torch.zeros(
                    (1, 1, masks.shape[-2], masks.shape[-1]), device=device, dtype=torch.float32
                )
            else:
                union = selected_masks.bool().any(dim=0, keepdim=True).unsqueeze(0).float()
        unions.append(F.adaptive_max_pool2d(union, output_size))
    if not unions:
        return torch.zeros((0, 1, *output_size), device=device, dtype=dtype)
    return torch.cat(unions, dim=0).to(dtype=dtype)


__all__ = [
    "DefectnessHead",
    "QueryLocalizationQualityHead",
    "pairwise_box_iou_cxcywh",
    "build_all_query_best_iou_target",
    "pool_query_semantic_evidence",
    "build_query_semantic_supervision",
    "compose_dual_quality_scores",
    "select_final_score_background",
    "final_score_ranking_loss",
    "build_union_defect_target",
]
