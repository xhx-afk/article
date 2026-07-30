from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_alignment_module(name: str = "sqalign_test_core"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "engine" / "deim" / "semantic_query_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_optional_dependency_stubs() -> None:
    if "torch.utils.tensorboard" not in sys.modules:
        module = types.ModuleType("torch.utils.tensorboard")
        module.SummaryWriter = object
        sys.modules[module.__name__] = module
    if "calflops" not in sys.modules:
        module = types.ModuleType("calflops")
        module.calculate_flops = lambda **_kwargs: ("N/A", "N/A", "N/A")
        sys.modules[module.__name__] = module
    if "faster_coco_eval" not in sys.modules:
        from pycocotools import mask as pycoco_mask
        from pycocotools.coco import COCO

        faster = types.ModuleType("faster_coco_eval")
        faster.__path__ = []
        faster.COCO = COCO
        faster.COCOeval_faster = object
        faster.init_as_pycocotools = lambda: None
        core = types.ModuleType("faster_coco_eval.core")
        core.__path__ = []
        mask_module = types.ModuleType("faster_coco_eval.core.mask")
        for attribute in dir(pycoco_mask):
            if not attribute.startswith("__"):
                setattr(mask_module, attribute, getattr(pycoco_mask, attribute))
        sys.modules[faster.__name__] = faster
        sys.modules[core.__name__] = core
        sys.modules[mask_module.__name__] = mask_module

