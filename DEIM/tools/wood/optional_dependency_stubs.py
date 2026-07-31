"""Minimal stubs for optional logging/evaluation packages absent in lean environments."""

from __future__ import annotations

import sys
import types


def install() -> None:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tensorboard_module = types.ModuleType("torch.utils.tensorboard")
        tensorboard_module.SummaryWriter = object
        sys.modules[tensorboard_module.__name__] = tensorboard_module
    try:
        import calflops  # noqa: F401
    except ImportError:
        calflops_module = types.ModuleType("calflops")
        calflops_module.calculate_flops = lambda **_kwargs: ("N/A", "N/A", "N/A")
        sys.modules[calflops_module.__name__] = calflops_module
    try:
        import faster_coco_eval  # noqa: F401
    except ImportError:
        from pycocotools import mask as pycoco_mask
        from pycocotools.coco import COCO

        faster_module = types.ModuleType("faster_coco_eval")
        faster_module.__path__ = []
        faster_module.COCO = COCO
        faster_module.COCOeval_faster = object
        faster_module.init_as_pycocotools = lambda: None
        faster_core = types.ModuleType("faster_coco_eval.core")
        faster_core.__path__ = []
        faster_mask = types.ModuleType("faster_coco_eval.core.mask")
        for attribute in dir(pycoco_mask):
            if not attribute.startswith("__"):
                setattr(faster_mask, attribute, getattr(pycoco_mask, attribute))
        sys.modules[faster_module.__name__] = faster_module
        sys.modules[faster_core.__name__] = faster_core
        sys.modules[faster_mask.__name__] = faster_mask

