"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import inspect

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes as TorchvisionSanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


_SANITIZE_INDEX_KEY = "_sanitize_instance_index"
_INSTANCE_FIELD_KEYS = ("labels", "area", "iscrowd", "mask_valid", "mixup")


def _get_target(inputs: Any):
    target = None
    if isinstance(inputs, dict):
        target = inputs
    elif isinstance(inputs, (tuple, list)) and len(inputs) > 1 and isinstance(inputs[1], dict):
        target = inputs[1]
    return target


def _replace_target(inputs: Any, target: Dict[str, Any]):
    if isinstance(inputs, dict):
        return target
    values = list(inputs)
    values[1] = target
    return tuple(values) if isinstance(inputs, tuple) else values


def _get_sanitize_index(inputs: Any):
    """旧版 torchvision 只接受单个 labels Tensor，因此使用临时实例索引。"""
    target = _get_target(inputs)
    return None if target is None else target.get(_SANITIZE_INDEX_KEY)


@register(name="SanitizeBoundingBoxes")
class SanitizeBoundingBoxes(TorchvisionSanitizeBoundingBoxes):
    """同步清理 boxes、masks 以及所有逐实例元数据。"""

    def __init__(self, min_size: float = 1.0, min_area: float = 1.0, labels_getter=None) -> None:
        self._sync_instance_fields = labels_getter is None
        kwargs = {
            "min_size": min_size,
            "labels_getter": _get_sanitize_index if self._sync_instance_fields else labels_getter,
        }
        try:
            supports_min_area = "min_area" in inspect.signature(
                TorchvisionSanitizeBoundingBoxes.__init__
            ).parameters
        except (TypeError, ValueError):
            supports_min_area = False
        if supports_min_area:
            kwargs["min_area"] = min_area
        elif min_area != 1.0:
            raise TypeError(
                "当前 torchvision 的 SanitizeBoundingBoxes 不支持 min_area；"
                "请使用默认 min_area=1.0 或升级 torchvision。"
            )
        super().__init__(**kwargs)

    def forward(self, *inputs: Any) -> Any:
        if not self._sync_instance_fields:
            return super().forward(*inputs)

        packed_inputs = inputs if len(inputs) > 1 else inputs[0]
        target = _get_target(packed_inputs)
        if target is None or not isinstance(target.get("boxes"), torch.Tensor):
            return super().forward(*inputs)

        instance_count = int(target["boxes"].shape[0])
        working_target = dict(target)
        working_target[_SANITIZE_INDEX_KEY] = torch.arange(
            instance_count, device=target["boxes"].device
        )
        outputs = super().forward(_replace_target(packed_inputs, working_target))
        output_target = _get_target(outputs)
        if output_target is None:
            return outputs

        keep_indices = output_target.pop(_SANITIZE_INDEX_KEY)
        for key in _INSTANCE_FIELD_KEYS:
            value = output_target.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == instance_count:
                output_target[key] = value[keep_indices]
        return outputs


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
