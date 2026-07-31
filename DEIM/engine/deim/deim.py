"""DEIM model with SQ-Align V3 auxiliary semantic branches."""

import torch.nn as nn

from ..core import register
from .continuous_semantic_alignment import (
    DefectnessHead,
    QuerySemanticProjection,
    SemanticPixelProjection,
    SemanticQualityHead,
    query_conditioned_mask_logits,
    roi_align_query_pixel_features,
)


__all__ = ["DEIM"]


@register()
class DEIM(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        enable_defect_auxiliary: bool = False,
        enable_query_conditioned_mask: bool = False,
        enable_semantic_quality: bool = False,
        semantic_in_channels: int = 256,
        defect_hidden_channels: int = 128,
        semantic_embed_dim: int = 64,
        semantic_pixel_downsample: bool = True,
        semantic_query_dim: int = 256,
        semantic_roi_size: int = 7,
        semantic_detach_boxes: bool = True,
        semantic_quality_query_dim: int = 32,
        semantic_quality_hidden_dim: int = 128,
        return_alignment_maps_in_eval: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.enable_defect_auxiliary = bool(enable_defect_auxiliary)
        self.enable_query_conditioned_mask = bool(enable_query_conditioned_mask)
        self.enable_semantic_quality = bool(enable_semantic_quality)
        self.semantic_roi_size = int(semantic_roi_size)
        self.semantic_detach_boxes = bool(semantic_detach_boxes)
        self.return_alignment_maps_in_eval = bool(return_alignment_maps_in_eval)

        if self.enable_query_conditioned_mask and not self.enable_defect_auxiliary:
            raise ValueError("query-conditioned masks require the V1 defect auxiliary stage")
        if self.enable_semantic_quality and not self.enable_query_conditioned_mask:
            raise ValueError("semantic quality requires query-conditioned masks")

        self.defectness_head = (
            DefectnessHead(semantic_in_channels, defect_hidden_channels)
            if self.enable_defect_auxiliary else None
        )
        self.semantic_pixel_projection = (
            SemanticPixelProjection(
                semantic_in_channels,
                semantic_embed_dim,
                downsample=semantic_pixel_downsample,
            )
            if self.enable_query_conditioned_mask else None
        )
        self.query_semantic_projection = (
            QuerySemanticProjection(semantic_query_dim, semantic_embed_dim)
            if self.enable_query_conditioned_mask else None
        )
        self.semantic_quality_head = (
            SemanticQualityHead(
                semantic_query_dim,
                query_projection_dim=semantic_quality_query_dim,
                hidden_dim=semantic_quality_hidden_dim,
            )
            if self.enable_semantic_quality else None
        )

    def forward(self, x, targets=None):
        features = self.encoder(self.backbone(x))
        defect_logits = (
            self.defectness_head(features[0]) if self.defectness_head is not None else None
        )
        pixel_embedding = (
            self.semantic_pixel_projection(features[0])
            if self.semantic_pixel_projection is not None else None
        )
        outputs = self.decoder(features, targets)

        if pixel_embedding is not None:
            query_features = outputs["pred_query_features"]
            query_embedding = self.query_semantic_projection(query_features)
            roi_pixels = roi_align_query_pixel_features(
                pixel_embedding,
                outputs["pred_boxes"],
                output_size=self.semantic_roi_size,
                detach_boxes=self.semantic_detach_boxes,
            )
            mask_logits = query_conditioned_mask_logits(query_embedding, roi_pixels)
            outputs["pred_query_mask_logits"] = mask_logits
            if self.semantic_quality_head is not None:
                outputs["pred_sem_quality"] = self.semantic_quality_head(
                    mask_logits, query_features
                )

        if defect_logits is not None and (
            self.training or self.return_alignment_maps_in_eval
        ):
            outputs["pred_defect_logits"] = defect_logits
        return outputs

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self
