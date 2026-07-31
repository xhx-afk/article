from __future__ import annotations

import inspect
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image as PILImage
from pycocotools import mask as pycoco_mask


def _install_optional_dependency_stubs() -> None:
    if "torch.utils.tensorboard" not in sys.modules:
        module = types.ModuleType("torch.utils.tensorboard")
        module.SummaryWriter = object
        sys.modules[module.__name__] = module
    if "calflops" not in sys.modules:
        module = types.ModuleType("calflops")
        module.calculate_flops = lambda **_kwargs: ("N/A", "N/A", "N/A")
        sys.modules[module.__name__] = module
    if "faster_coco_eval" not in sys.modules:
        from pycocotools.coco import COCO
        faster = types.ModuleType("faster_coco_eval")
        faster.__path__ = []
        faster.COCO = COCO
        faster.COCOeval_faster = object
        faster.init_as_pycocotools = lambda: None
        core = types.ModuleType("faster_coco_eval.core")
        core.__path__ = []
        mask_module = types.ModuleType("faster_coco_eval.core.mask")
        for name in dir(pycoco_mask):
            if not name.startswith("__"):
                setattr(mask_module, name, getattr(pycoco_mask, name))
        sys.modules[faster.__name__] = faster
        sys.modules[core.__name__] = core
        sys.modules["faster_coco_eval.core.mask"] = mask_module


_install_optional_dependency_stubs()

from engine.data._misc import BoundingBoxes, BoundingBoxFormat, Image, Mask  # noqa: E402
from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask, convert_coco_poly_to_mask  # noqa: E402
from engine.data.transforms._transforms import SanitizeBoundingBoxes  # noqa: E402
from engine.data.transforms.mosaic import Mosaic  # noqa: E402


class SemanticMaskDataChainTest(unittest.TestCase):
    def test_mosaic_places_masks_on_full_canvas(self) -> None:
        width, height = 20, 10

        def make_sample():
            image = PILImage.new("RGB", (width, height), "white")
            mask = torch.zeros((1, height, width), dtype=torch.bool)
            mask[:, 1:5, 2:7] = True
            target = {
                "boxes": BoundingBoxes(
                    torch.tensor([[2, 1, 7, 5]], dtype=torch.float32),
                    format=BoundingBoxFormat.XYXY,
                    canvas_size=(height, width),
                ),
                "labels": torch.tensor([0]),
                "area": torch.tensor([20.0]),
                "iscrowd": torch.tensor([0]),
                "mask_valid": torch.tensor([True]),
                "masks": Mask(mask),
                "orig_size": torch.tensor([height, width]),
                "size": torch.tensor([height, width]),
                "image_id": torch.tensor([1]),
            }
            return image, target

        class OneSampleDataset:
            def __len__(self):
                return 1

            def load_item(self, _index):
                return make_sample()

        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                transform = Mosaic(
                    output_size=(height, width),
                    rotation_range=0,
                    translation_range=(0.0, 0.0),
                    scaling_range=(1.0, 1.0),
                    probability=1.0,
                    use_cache=use_cache,
                )
                image, target = make_sample()
                output_image, output_target, _ = transform((image, target, OneSampleDataset()))

                self.assertEqual(output_image.size, (width * 2, height * 2))
                self.assertEqual(tuple(output_target["masks"].shape), (4, height * 2, width * 2))
                self.assertEqual(tuple(output_target["boxes"].shape), (4, 4))
                self.assertEqual(output_target["labels"].shape[0], 4)
                expected_boxes = torch.tensor([
                    [2, 1, 7, 5],
                    [width + 2, 1, width + 7, 5],
                    [2, height + 1, 7, height + 5],
                    [width + 2, height + 1, width + 7, height + 5],
                ], dtype=torch.float32)
                torch.testing.assert_close(output_target["boxes"].as_subclass(torch.Tensor), expected_boxes)
                for index, (offset_x, offset_y) in enumerate(
                    ((0, 0), (width, 0), (0, height), (width, height))
                ):
                    self.assertTrue(
                        output_target["masks"][
                            index,
                            offset_y:offset_y + height,
                            offset_x:offset_x + width,
                        ].any()
                    )

    def test_sanitize_supports_old_torchvision_api(self) -> None:
        old_signature = inspect.Signature([
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("min_size", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1.0),
            inspect.Parameter("labels_getter", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="default"),
        ])
        with mock.patch(
            "engine.data.transforms._transforms.inspect.signature",
            return_value=old_signature,
        ):
            transform = SanitizeBoundingBoxes()
            with self.assertRaisesRegex(TypeError, "不支持 min_area"):
                SanitizeBoundingBoxes(min_area=2.0)

        original_getter = transform._labels_getter

        def tensor_only_getter(inputs):
            labels = original_getter(inputs)
            if labels is not None and not isinstance(labels, torch.Tensor):
                raise ValueError(f"old torchvision rejects {type(labels)}")
            return labels

        transform._labels_getter = tensor_only_getter
        image = Image(torch.zeros((3, 16, 16), dtype=torch.uint8))
        boxes = BoundingBoxes(
            torch.tensor([[1, 1, 8, 8], [2, 2, 2, 6]], dtype=torch.float32),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(16, 16),
        )
        target = {
            "boxes": boxes,
            "labels": torch.tensor([0, 1]),
            "area": torch.tensor([49.0, 0.0]),
            "iscrowd": torch.tensor([0, 0]),
            "mask_valid": torch.tensor([True, False]),
        }
        _, output, _ = transform((image, target, None))
        self.assertEqual(output["labels"].tolist(), [0])
        self.assertNotIn("_sanitize_instance_index", output)

    def test_polygon_compressed_rle_and_empty(self) -> None:
        height = width = 16
        binary = np.zeros((height, width), dtype=np.uint8)
        binary[2:8, 3:9] = 1
        rle = pycoco_mask.encode(np.asfortranarray(binary))
        compressed = {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}
        polygon = [[3, 2, 9, 2, 9, 8, 3, 8]]
        masks = convert_coco_poly_to_mask([polygon, compressed, []], height, width)
        self.assertEqual(tuple(masks.shape), (3, height, width))
        self.assertTrue(masks[0].any())
        self.assertTrue(masks[1].any())
        self.assertFalse(masks[2].any())

    def test_mask_valid_uses_same_box_keep(self) -> None:
        image = PILImage.new("RGB", (16, 16))
        annotations = [
            {"id": 1, "bbox": [2, 2, 6, 6], "category_id": 0, "area": 36, "iscrowd": 0,
             "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]], "mask_valid": 1},
            {"id": 2, "bbox": [5, 5, 0, 4], "category_id": 1, "area": 0, "iscrowd": 0,
             "segmentation": [], "mask_valid": 0},
        ]
        _, target = ConvertCocoPolysToMask(return_masks=True)(
            image, {"image_id": 1, "annotations": annotations}
        )
        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
        self.assertEqual(tuple(target["masks"].shape), (1, 16, 16))
        self.assertEqual(target["mask_valid"].tolist(), [True])

    def test_sanitize_synchronizes_all_instance_fields(self) -> None:
        image = Image(torch.zeros((3, 16, 16), dtype=torch.uint8))
        boxes = BoundingBoxes(
            torch.tensor([[1, 1, 8, 8], [2, 2, 2, 6]], dtype=torch.float32),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(16, 16),
        )
        target = {
            "boxes": boxes,
            "labels": torch.tensor([0, 1]),
            "area": torch.tensor([49.0, 0.0]),
            "iscrowd": torch.tensor([0, 0]),
            "mask_valid": torch.tensor([True, False]),
            "masks": Mask(torch.ones((2, 16, 16), dtype=torch.bool)),
        }
        _, output, _ = SanitizeBoundingBoxes()((image, target, None))
        for key in ("boxes", "labels", "area", "iscrowd", "mask_valid", "masks"):
            self.assertEqual(output[key].shape[0], 1, key)


if __name__ == "__main__":
    unittest.main()
