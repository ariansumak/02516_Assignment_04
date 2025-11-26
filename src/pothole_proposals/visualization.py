from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

from .dataset import ImageRecord
from .labeling import compute_iou



def save_visualizations(
    records: Sequence[ImageRecord],
    output_dir: Path,
    *,
    limit: int = 6,
    random_seed: int | None = 13,
) -> List[Path]:
    """Save a few dataset examples with their ground-truth boxes overlaid."""

    if not records:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(records)
    if random_seed is not None:
        random.Random(random_seed).shuffle(chosen)

    saved_paths: List[Path] = []
    for record in chosen[:limit]:
        image = Image.open(record.image_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(record.image_id)

        # ---- draw GT boxes (original behaviour) ----
        for box in record.annotations:
            rect = patches.Rectangle(
                (box.x_min, box.y_min),
                box.x_max - box.x_min,
                box.y_max - box.y_min,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                box.x_min,
                box.y_min - 2,
                box.label,
                color="lime",
                fontsize=9,
                backgroundcolor="black",
            )

        output_path = output_dir / f"{record.image_id}.png"
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def save_proposal_visualizations(
    records: Sequence[ImageRecord],
    proposals_dir: Path,
    output_dir: Path,
    *,
    limit: int = 6,
    random_seed: int | None = 13,
    min_iou: float = 0.3,
    max_props_per_object: int = 5,
) -> List[Path]:
    """
    Visualise proposals instead of GT, keeping only a few per object.

    For each image:
      - loads the labelled proposals JSON
      - for each ground-truth object in that image:
          * recomputes IoU between each proposal and that GT box
          * keeps up to `max_props_per_object` proposals with IoU >= min_iou
      - draws those proposals in red (no GT boxes drawn).
    """

    if not records:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(records)
    if random_seed is not None:
        random.Random(random_seed).shuffle(chosen)

    saved_paths: List[Path] = []
    for record in chosen[:limit]:
        json_path = proposals_dir / f"{record.image_id}.json"
        if not json_path.exists():
            # skip images without proposals
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        proposals = data.get("proposals", [])
        if not proposals or not record.annotations:
            continue

        image = Image.open(record.image_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"{record.image_id} – proposals (top {max_props_per_object}/object)")

        # For each GT object, keep its best proposals
        for obj_idx, gt in enumerate(record.annotations):
            gt_box = (gt.x_min, gt.y_min, gt.x_max, gt.y_max)

            candidates = []
            for p in proposals:
                bbox = tuple(p["bbox"])
                iou = compute_iou(bbox, gt_box)
                if iou >= min_iou:
                    candidates.append({"bbox": bbox, "iou": iou, "obj_idx": obj_idx})

            # sort by IoU and keep only top-k proposals for this object
            candidates.sort(key=lambda c: c["iou"], reverse=True)
            candidates = candidates[:max_props_per_object]

            for cand in candidates:
                x_min, y_min, x_max, y_max = cand["bbox"]
                iou = cand["iou"]

                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x_min,
                    y_min - 2,
                    f"obj{obj_idx} ({iou:.2f})",
                    color="white",
                    fontsize=8,
                    backgroundcolor="red",
                )

        output_path = output_dir / f"{record.image_id}_proposals.png"
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths
