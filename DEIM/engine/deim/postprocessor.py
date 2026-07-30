"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register
from .semantic_query_alignment import compose_dual_quality_scores


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        loc_quality_rerank=False,
        loc_quality_power=0.25,
        semantic_quality_rerank=False,
        semantic_quality_power=0.25,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.loc_quality_rerank = bool(loc_quality_rerank)
        self.loc_quality_power = float(loc_quality_power)
        self.semantic_quality_rerank = bool(semantic_quality_rerank)
        self.semantic_quality_power = float(semantic_quality_power)
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores = self._compose_scores(scores, outputs)
            topk = min(self.num_top_queries, scores.shape[1] * scores.shape[2])
            scores, index = torch.topk(scores.flatten(1), topk, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores = self._compose_scores(scores, outputs)
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def _compose_scores(self, class_scores, outputs):
        pred_loc_quality = None
        pred_sem_quality = None
        if self.loc_quality_rerank:
            if 'pred_loc_quality' not in outputs:
                raise KeyError("loc_quality_rerank requires pred_loc_quality")
            pred_loc_quality = outputs['pred_loc_quality']
        if self.semantic_quality_rerank:
            if 'pred_sem_quality' not in outputs:
                raise KeyError("semantic_quality_rerank requires pred_sem_quality")
            pred_sem_quality = outputs['pred_sem_quality']
        return compose_dual_quality_scores(
            class_scores,
            pred_loc_quality=pred_loc_quality,
            pred_sem_quality=pred_sem_quality,
            loc_quality_power=self.loc_quality_power,
            semantic_quality_power=self.semantic_quality_power,
        )


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
