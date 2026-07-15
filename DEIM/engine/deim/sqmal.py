"""SQ-MAL heads and tensor-only semantic quality utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectnessHead(nn.Module):
    """Lightweight binary defectness head for the highest-resolution encoder feature."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class QueryQualityHead(nn.Module):
    """Predict one joint localization-semantic quality logit per query."""

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


def apply_quality_rerank(
    scores: torch.Tensor,
    quality_logit: Optional[torch.Tensor],
    quality_power: float,
) -> torch.Tensor:
    """Apply query quality before class flatten/top-k; power=0 is identity."""
    if quality_logit is None or float(quality_power) == 0.0:
        return scores
    quality = torch.sigmoid(quality_logit)
    if quality.shape[-1] != 1 or quality.shape[:2] != scores.shape[:2]:
        raise ValueError(f"quality shape {quality.shape} is incompatible with scores {scores.shape}")
    return scores * quality.to(dtype=scores.dtype).pow(float(quality_power))


def semantic_beta_schedule(max_beta: float, warmup_epochs: int, current_epoch: int) -> float:
    if warmup_epochs <= 0:
        return float(max_beta)
    progress = min(1.0, max(0.0, float(current_epoch) / float(warmup_epochs)))
    return float(max_beta) * progress


def _boxes_to_pixel_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert normalized cxcywh [N,4] to clamped integer xyxy [N,4]."""
    if boxes.numel() == 0:
        return torch.empty((0, 4), device=boxes.device, dtype=torch.long)
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = torch.floor(((cx - bw / 2.0).clamp(0, 1) * width).float())
    y1 = torch.floor(((cy - bh / 2.0).clamp(0, 1) * height).float())
    x2 = torch.ceil(((cx + bw / 2.0).clamp(0, 1) * width).float())
    y2 = torch.ceil(((cy + bh / 2.0).clamp(0, 1) * height).float())
    x1 = x1.clamp(0, max(width - 1, 0)).long()
    y1 = y1.clamp(0, max(height - 1, 0)).long()
    x2 = torch.maximum(x2.clamp(1, width).long(), x1 + 1)
    y2 = torch.maximum(y2.clamp(1, height).long(), y1 + 1)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def compute_box_mask_sums(masks: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact per-mask rectangle sums with integral images.

    Args:
        masks: [N,H,W] bool/uint8/float, one GT mask per matched query.
        boxes: [N,4] normalized cxcywh.
    Returns:
        mask_sum_in_box [N], pixel_box_area [N].
    """
    if masks.ndim != 3 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"expected masks [N,H,W] and boxes [N,4], got {masks.shape}, {boxes.shape}")
    if masks.shape[0] != boxes.shape[0]:
        raise ValueError("masks and boxes must have the same N")
    if boxes.shape[0] == 0:
        empty = boxes.new_zeros((0,), dtype=torch.float32)
        return empty, empty
    height, width = int(masks.shape[-2]), int(masks.shape[-1])
    pixel_boxes = _boxes_to_pixel_xyxy(boxes.detach(), height, width)
    x1, y1, x2, y2 = pixel_boxes.unbind(-1)
    integral = F.pad(masks.float().cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    rows = torch.arange(masks.shape[0], device=masks.device)
    sums = integral[rows, y2, x2] - integral[rows, y1, x2] - integral[rows, y2, x1] + integral[rows, y1, x1]
    areas = ((x2 - x1) * (y2 - y1)).to(dtype=sums.dtype)
    return sums, areas


def compute_semantic_support(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_valid: torch.Tensor,
    min_mask_area: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute detached support s=sqrt(coverage*relative_density), shape [N]."""
    count = pred_boxes.shape[0]
    if count == 0:
        return pred_boxes.new_zeros((0,), dtype=torch.float32)
    fallback = pred_boxes.new_ones((count,), dtype=torch.float32)
    if gt_masks.ndim != 3 or gt_masks.shape[0] != count or gt_masks.shape[-2] <= 0 or gt_masks.shape[-1] <= 0:
        return fallback
    valid = mask_valid.to(device=pred_boxes.device, dtype=torch.bool).reshape(-1)
    if valid.numel() != count:
        return fallback
    masks = gt_masks.to(device=pred_boxes.device)
    mask_area = masks.float().flatten(1).sum(-1)
    pred_sum, pred_area = compute_box_mask_sums(masks, pred_boxes)
    gt_sum, gt_area = compute_box_mask_sums(masks, gt_boxes)
    coverage = (pred_sum / (mask_area + eps)).clamp(0.0, 1.0)
    pred_density = pred_sum / (pred_area + eps)
    gt_density = gt_sum / (gt_area + eps)
    relative_density = (pred_density / (gt_density + eps)).clamp(0.0, 1.0)
    support = torch.sqrt((coverage * relative_density).clamp_min(0.0))
    valid = valid & (mask_area >= float(min_mask_area)) & (gt_area >= 1.0) & torch.isfinite(support)
    support = torch.where(valid, support, fallback)
    return torch.nan_to_num(support, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0).detach()


def compute_sq_quality_target(
    iou: torch.Tensor,
    semantic_support: torch.Tensor,
    beta: float,
    iou_power: float = 1.0,
) -> torch.Tensor:
    target = iou.detach().float().clamp(0.0, 1.0).pow(float(iou_power))
    if float(beta) != 0.0:
        target = target * semantic_support.detach().float().clamp(0.0, 1.0).pow(float(beta))
    return torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def build_union_defect_target(
    targets: Sequence[Dict[str, torch.Tensor]],
    output_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B,1,Hd,Wd] binary union targets with adaptive max pooling."""
    unions: List[torch.Tensor] = []
    for target in targets:
        masks = target.get("masks")
        valid = target.get("mask_valid")
        if masks is None or masks.ndim != 3:
            source_size = target.get("orig_size")
            h = int(source_size[1]) if source_size is not None else output_size[0]
            w = int(source_size[0]) if source_size is not None else output_size[1]
            union = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
        else:
            masks = masks.to(device=device)
            valid = torch.ones((masks.shape[0],), device=device, dtype=torch.bool) if valid is None else valid.to(device=device, dtype=torch.bool)
            selected = masks[valid]
            if selected.shape[0] == 0:
                union = torch.zeros((1, 1, masks.shape[-2], masks.shape[-1]), device=device, dtype=torch.float32)
            else:
                union = selected.bool().any(dim=0, keepdim=True).unsqueeze(0).float()
        unions.append(F.adaptive_max_pool2d(union, output_size))
    if not unions:
        return torch.zeros((0, 1, *output_size), device=device, dtype=dtype)
    return torch.cat(unions, dim=0).to(dtype=dtype)


def pairwise_box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    b1_xy1 = boxes1[:, :2] - boxes1[:, 2:] / 2
    b1_xy2 = boxes1[:, :2] + boxes1[:, 2:] / 2
    b2_xy1 = boxes2[:, :2] - boxes2[:, 2:] / 2
    b2_xy2 = boxes2[:, :2] + boxes2[:, 2:] / 2
    inter_xy1 = torch.maximum(b1_xy1[:, None], b2_xy1[None])
    inter_xy2 = torch.minimum(b1_xy2[:, None], b2_xy2[None])
    inter = (inter_xy2 - inter_xy1).clamp_min(0).prod(-1)
    area1 = (b1_xy2 - b1_xy1).clamp_min(0).prod(-1)
    area2 = (b2_xy2 - b2_xy1).clamp_min(0).prod(-1)
    return inter / (area1[:, None] + area2[None] - inter).clamp_min(1e-6)


def select_hard_background_queries(
    pred_logits: torch.Tensor,
    pred_quality: Optional[torch.Tensor],
    pred_boxes: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    matched_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    topk: int,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return selected mask [B,Q] and selected counts [B]."""
    batch, queries = pred_logits.shape[:2]
    selected = torch.zeros((batch, queries), device=pred_logits.device, dtype=torch.bool)
    counts = torch.zeros((batch,), device=pred_logits.device, dtype=torch.float32)
    quality = torch.sigmoid(pred_quality.squeeze(-1)) if pred_quality is not None else torch.ones((batch, queries), device=pred_logits.device)
    confidence = torch.sigmoid(pred_logits).amax(-1)
    difficulty = confidence * (1.0 - quality)
    for batch_index in range(batch):
        unmatched = torch.ones((queries,), device=pred_logits.device, dtype=torch.bool)
        unmatched[matched_indices[batch_index][0]] = False
        gt_boxes = targets[batch_index]["boxes"].to(pred_boxes.device)
        if gt_boxes.shape[0] == 0:
            far_from_gt = torch.ones_like(unmatched)
        else:
            far_from_gt = pairwise_box_iou_cxcywh(pred_boxes[batch_index].detach(), gt_boxes).amax(-1) < float(iou_threshold)
        candidates = torch.nonzero(unmatched & far_from_gt, as_tuple=False).squeeze(1)
        k = min(max(int(topk), 0), int(candidates.numel()))
        if k > 0:
            chosen = candidates[torch.topk(difficulty[batch_index, candidates], k=k, sorted=False).indices]
            selected[batch_index, chosen] = True
            counts[batch_index] = float(k)
    return selected, counts
