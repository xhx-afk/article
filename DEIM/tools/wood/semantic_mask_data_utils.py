"""Semantic-map 数据工具的共享函数。"""

from __future__ import annotations

# Shared semantic-mask data utilities used by SQ-Align.

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


SemanticValue = Union[int, Tuple[int, int, int]]


EXPECTED_CLASSES = (
    "Live_knot",
    "Dead_knot",
    "resin",
    "knot_with_crack",
    "Crack",
    "Marrow",
    "Quartzity",
    "Knot_missing",
    "Blue_stain",
)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


ALIASES: Dict[str, str] = {
    normalize_name(name): name for name in EXPECTED_CLASSES
}
ALIASES.update({
    "liveknot": "Live_knot",
    "livingknot": "Live_knot",
    "deathknot": "Dead_knot",
    "deadknot": "Dead_knot",
    "knotcrack": "knot_with_crack",
    "knotwithcrack": "knot_with_crack",
    "missingknot": "Knot_missing",
    "bluestain": "Blue_stain",
    "bluecolor": "Blue_stain",
    "background": "__background__",
    "bg": "__background__",
    "overgrown": "__overgrown__",
})


@dataclass(frozen=True)
class SemanticSpec:
    mode: str
    class_values: Dict[str, SemanticValue]
    background_values: Tuple[SemanticValue, ...] = ()
    overgrown_values: Tuple[SemanticValue, ...] = ()


def canonical_class_name(name: str) -> Optional[str]:
    return ALIASES.get(normalize_name(name))


def _parse_mapping_value(value: object) -> SemanticValue:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        numbers = [int(item) for item in re.findall(r"-?\d+", value)]
    elif isinstance(value, Sequence):
        numbers = [int(item) for item in value]
    else:
        raise ValueError(f"无法解析 semantic value: {value!r}")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 3:
        return tuple(numbers)
    raise ValueError(f"semantic value 必须是一个灰度值或三个 RGB 值: {value!r}")


def _finalize_spec(entries: Mapping[str, SemanticValue], source_lines: Iterable[str]) -> SemanticSpec:
    class_values: Dict[str, SemanticValue] = {}
    background: List[SemanticValue] = []
    overgrown: List[SemanticValue] = []
    for raw_name, value in entries.items():
        canonical = canonical_class_name(raw_name)
        if canonical is None:
            continue
        if canonical == "__background__":
            background.append(value)
        elif canonical == "__overgrown__":
            overgrown.append(value)
        else:
            class_values[canonical] = value

    value_lengths = {1 if isinstance(value, int) else len(value) for value in class_values.values()}
    if len(value_lengths) > 1:
        raise ValueError("semantic specification 同时包含灰度值与 RGB 值，无法确定 map 类型。")
    if not class_values or value_lengths not in ({1}, {3}):
        lines = "\n".join(source_lines)
        raise ValueError(
            "无法从 semantic specification 解析类别映射。已读取内容：\n"
            f"{lines}\n请使用 --class-map-json 提供显式映射。"
        )
    mode = "gray" if value_lengths == {1} else "rgb"
    return SemanticSpec(mode, class_values, tuple(background), tuple(overgrown))


def parse_semantic_spec(path: Path) -> SemanticSpec:
    """解析以空格、冒号、逗号或制表符分隔的灰度/RGB semantic spec。"""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    entries: Dict[str, SemanticValue] = {}
    alias_keys = sorted(ALIASES, key=len, reverse=True)
    for raw_line in lines:
        line = re.split(r"#|//", raw_line, maxsplit=1)[0].strip()
        if not line:
            continue
        normalized = normalize_name(re.sub(r"-?\d+", "", line))
        matched_alias = next((alias for alias in alias_keys if alias in normalized), None)
        if matched_alias is None:
            continue
        numbers = [int(item) for item in re.findall(r"-?\d+", line)]
        if len(numbers) not in (1, 3):
            continue
        entries[matched_alias] = numbers[0] if len(numbers) == 1 else tuple(numbers)
    return _finalize_spec(entries, lines)


def load_class_map_json(path: Path) -> SemanticSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--class-map-json 必须是 {类别名: 灰度值或 [R,G,B]}。")
    entries = {str(name): _parse_mapping_value(value) for name, value in raw.items()}
    return _finalize_spec(entries, [json.dumps(raw, ensure_ascii=False)])


def semantic_binary_mask(array: np.ndarray, value: SemanticValue, mode: str) -> np.ndarray:
    if mode == "gray":
        if array.ndim == 3:
            if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(array[..., 0], array[..., 2]):
                raise ValueError("spec 是灰度索引，但 semantic map 是非灰度 RGB 图。")
            array = array[..., 0]
        return array == int(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("spec 是 RGB 映射，但 semantic map 不是 RGB 图。")
    rgb = np.asarray(value, dtype=array.dtype)
    return np.all(array[..., :3] == rgb, axis=-1)


def encode_rle(mask: np.ndarray) -> Dict[str, object]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("需要 pycocotools：pip install pycocotools") from exc
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def decode_segmentation(segmentation: object, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("需要 pycocotools：pip install pycocotools") from exc
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    elif isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        raise TypeError(f"不支持的 segmentation 类型: {type(segmentation)}")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def find_by_stem(directory: Path, stem: str) -> Optional[Path]:
    candidates = [path for path in directory.iterdir() if path.is_file() and path.stem.lower() == stem.lower()]
    return sorted(candidates)[0] if candidates else None


def bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
