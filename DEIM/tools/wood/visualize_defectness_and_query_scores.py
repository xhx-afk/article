"""Visualize SQ-Align defectness heatmaps and top final-score queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optional_dependency_stubs import install  # noqa: E402

install()

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from engine.deim.semantic_query_alignment import compose_dual_quality_scores  # noqa: E402


def _checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        return checkpoint["ema"].get("module", checkpoint["ema"])
    return checkpoint.get("model", checkpoint)


def _load_matching(model, path: Path) -> None:
    current = model.state_dict()
    state = _checkpoint_state(path)
    compatible = {key: value for key, value in state.items() if key in current and value.shape == current[key].shape}
    model.load_state_dict(compatible, strict=False)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    config = YAMLConfig(args.config)
    config.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    config.yaml_cfg["val_dataloader"]["num_workers"] = args.num_workers
    device = torch.device(args.device)
    model = config.model
    _load_matching(model, Path(args.checkpoint))
    model = model.to(device).eval()
    if model.defectness_head is None:
        raise RuntimeError("visualization requires an R2/R3/R4 model with defectness enabled")
    model.return_defect_map_in_eval = True
    postprocessor = config.postprocessor
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, (samples, targets) in enumerate(config.val_dataloader):
        if batch_index >= args.num_images:
            break
        images = samples.to(device)
        outputs = model(images)
        class_scores = outputs["pred_logits"].sigmoid()
        final_scores = compose_dual_quality_scores(
            class_scores,
            outputs.get("pred_loc_quality") if postprocessor.loc_quality_rerank else None,
            outputs.get("pred_sem_quality") if postprocessor.semantic_quality_rerank else None,
            postprocessor.loc_quality_power,
            postprocessor.semantic_quality_power,
        )
        query_scores, query_labels = final_scores[0].max(-1)
        top_indices = query_scores.topk(min(args.top_queries, query_scores.numel())).indices
        loc_quality = outputs["pred_loc_quality"].sigmoid()[0, :, 0]
        semantic_quality = outputs.get("pred_sem_quality")
        semantic_quality = semantic_quality[0, :, 0] if semantic_quality is not None else None
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0]).clamp(0, 1)

        image = images[0].detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        heatmap = F.interpolate(
            outputs["pred_defect_logits"].sigmoid(), size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].float().cpu().numpy()
        height, width = image.shape[:2]
        figure, axis = plt.subplots(figsize=(12, 12))
        axis.imshow(image)
        axis.imshow(heatmap, cmap="inferno", alpha=0.42, vmin=0, vmax=1)
        colors = plt.cm.viridis(torch.linspace(0, 1, len(top_indices)).numpy())
        for rank, (query_index, color) in enumerate(zip(top_indices.tolist(), colors), start=1):
            x1, y1, x2, y2 = boxes[query_index].float().cpu().tolist()
            rectangle = patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            axis.add_patch(rectangle)
            axis.text(
                x1 * width,
                y1 * height,
                f"#{rank} q{query_index} c{int(query_labels[query_index])}\n"
                f"S={float(query_scores[query_index]):.3f} "
                f"L={float(loc_quality[query_index]):.3f} "
                f"M={'n/a' if semantic_quality is None else f'{float(semantic_quality[query_index]):.3f}'}",
                color="white",
                fontsize=7,
                bbox={"facecolor": color, "alpha": 0.65, "pad": 1},
            )
        image_id = int(targets[0]["image_id"].reshape(-1)[0])
        axis.set_title(f"image={image_id}: defectness + class × loc × semantic top queries")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / f"{image_id}_sqalign.png", dpi=180)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--top-queries", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
