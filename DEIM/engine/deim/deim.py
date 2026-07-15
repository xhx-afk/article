"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register
from .sqmal import DefectnessHead


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_defectness_head: bool = False,
        defectness_in_channels: int = 256,
        defectness_hidden_channels: int = 128,
        return_defect_map_in_eval: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_defectness_head = bool(use_defectness_head)
        self.return_defect_map_in_eval = bool(return_defect_map_in_eval)
        self.defectness_head = DefectnessHead(
            defectness_in_channels, defectness_hidden_channels
        ) if self.use_defectness_head else None

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        defect_logits = self.defectness_head(x[0]) if self.defectness_head is not None else None
        x = self.decoder(x, targets)
        if defect_logits is not None and (self.training or self.return_defect_map_in_eval):
            x['pred_defect_logits'] = defect_logits

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
