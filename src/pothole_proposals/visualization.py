from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

from .dataset import ImageRecord


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
