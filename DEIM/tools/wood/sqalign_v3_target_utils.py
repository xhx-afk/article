"""Evaluation-only target conversion for SQ-Align V3 diagnostics.

The validation data pipeline exposes boxes as absolute ``xyxy`` coordinates,
whereas the detector matcher and all semantic-alignment helpers consume
normalized ``cxcywh``.  This module keeps those two representations separate;
it never mutates the raw target dictionaries.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torchvision
from torchvision import tv_tensors


def _resolve_canvas_hw(
    target: Mapping[str, object], sample_hw: tuple[int, int]
) -> tuple[int, int]:
    """Resolve the evaluation canvas in priority order specified by the spec."""
    boxes = target.get("boxes")
    if isinstance(boxes, tv_tensors.BoundingBoxes):
        height, width = map(int, boxes.canvas_size)
        return height, width

    masks = target.get("masks")
    if isinstance(masks, torch.Tensor) and masks.ndim >= 2:
        return int(masks.shape[-2]), int(masks.shape[-1])

    return int(sample_hw[0]), int(sample_hw[1])


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return a plain Tensor without modifying or detaching the source value."""
    if isinstance(value, tv_tensors.BoundingBoxes):
        return value.as_subclass(torch.Tensor)
    return torch.as_tensor(value)


def build_alignment_targets(
    targets: Sequence[Mapping[str, object]],
    sample_hw: tuple[int, int],
) -> list[dict[str, object]]:
    """Convert absolute ``xyxy`` boxes to normalized ``cxcywh`` copies.

    Masks and all other target entries are shared with the raw target.  Any
    invalid converted box aborts evaluation instead of producing misleading
    alignment diagnostics.
    """
    converted: list[dict[str, object]] = []
    for target in targets:
        if "boxes" not in target:
            raise KeyError("evaluation target is missing boxes")
        source_boxes = target["boxes"]
        if not isinstance(source_boxes, torch.Tensor):
            raise TypeError("evaluation target boxes must be a Tensor")
        if isinstance(source_boxes, tv_tensors.BoundingBoxes):
            box_format = str(source_boxes.format).upper()
            if "XYXY" not in box_format:
                raise ValueError(
                    f"Evaluator expected absolute XYXY boxes, got {box_format}"
                )

        height, width = _resolve_canvas_hw(target, sample_hw)
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid alignment canvas: height={height}, width={width}")

        boxes = _plain_tensor(source_boxes).float()
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"expected target boxes [N,4], got {tuple(boxes.shape)}")
        boxes_cxcywh = torchvision.ops.box_convert(boxes, "xyxy", "cxcywh")
        scale = boxes_cxcywh.new_tensor([width, height, width, height])
        boxes_cxcywh = boxes_cxcywh / scale

        if boxes_cxcywh.numel():
            if not torch.isfinite(boxes_cxcywh).all():
                raise ValueError("alignment boxes contain NaN or Inf")
            minimum = float(boxes_cxcywh.min())
            maximum = float(boxes_cxcywh.max())
            if minimum < -1e-4:
                raise ValueError(f"alignment boxes have negative values: {minimum}")
            if maximum > 1.0001:
                raise ValueError(f"alignment boxes are not normalized: {maximum}")
            if (boxes_cxcywh[:, 2:] <= 0).any():
                raise ValueError("alignment boxes contain non-positive size")

        result = dict(target)
        result["boxes"] = boxes_cxcywh
        result["alignment_canvas_hw"] = torch.tensor(
            [height, width], dtype=torch.int64, device=boxes_cxcywh.device
        )
        converted.append(result)
    return converted


class AlignmentSanityAccumulator:
    """Aggregate the mandatory coordinate self-check section."""

    def __init__(self) -> None:
        self.images_checked = 0
        self.boxes_checked = 0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def update(self, targets: Sequence[Mapping[str, object]]) -> None:
        self.images_checked += len(targets)
        for target in targets:
            boxes = target.get("boxes")
            if not isinstance(boxes, torch.Tensor):
                raise TypeError("alignment target boxes must be a Tensor")
            self.boxes_checked += int(boxes.shape[0])
            if boxes.numel():
                minimum = float(boxes.min())
                maximum = float(boxes.max())
                self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)
                self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)

    def report(self) -> dict[str, object]:
        return {
            "source_box_format": "absolute_xyxy",
            "alignment_box_format": "normalized_cxcywh",
            "images_checked": self.images_checked,
            "boxes_checked": self.boxes_checked,
            "min_value": 0.0 if self.minimum is None else self.minimum,
            "max_value": 0.0 if self.maximum is None else self.maximum,
            "non_finite_count": 0,
            "non_positive_size_count": 0,
        }
