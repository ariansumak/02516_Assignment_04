# src/part2/proposal_dataset.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from pothole_proposals.dataset import DatasetIndex  # your existing index


@dataclass(frozen=True)
class ProposalSample:
    image_path: Path
    bbox: Tuple[int, int, int, int]
    label: int  # 0 = background, 1 = pothole


class ProposalClassificationDataset(Dataset):
    """
    Dataset of cropped proposal regions for classification (background vs pothole).

    It reads the labelled proposals produced in Part 1:
      outputs/part1/training_proposals/<split>/*.json

    Each JSON should look like:
      {
        "image_id": "potholes0",
        "proposals": [
          {"bbox": [x_min, y_min, x_max, y_max], "label": "background" or "pothole", "iou": 0.53},
          ...
        ]
      }
    """

    def __init__(
        self,
        dataset_root: Path,
        proposals_root: Path,
        *,
        split: str = "train",
        resize_size: Tuple[int, int] = (224, 224),
        balance_background: bool = True,
        max_bg_ratio: float = 3.0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.dataset_index = DatasetIndex(dataset_root, split=split)
        self.image_map: Dict[str, Path] = {
            rec.image_id: rec.image_path for rec in self.dataset_index.records
        }

        self.transform = T.Compose(
            [
                T.Resize(resize_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        rng = random.Random(seed)
        self.samples: List[ProposalSample] = []

        split_dir = proposals_root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Proposals directory for split '{split}' not found at {split_dir}. "
                f"Did you run run_part1_pipeline.py with --split {split}?"
            )

        # ---- Load all proposals for this split ----
        for json_path in sorted(split_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_id: str = data["image_id"]
            image_path = self.image_map[image_id]

            for prop in data["proposals"]:
                bbox = tuple(prop["bbox"])
                label_str = prop["label"]

                if label_str == "background":
                    label = 0
                else:
                    # Any pothole-like label becomes foreground
                    label = 1

                self.samples.append(
                    ProposalSample(image_path=image_path, bbox=bbox, label=label)
                )

        if not self.samples:
            raise RuntimeError(f"No proposals found in {split_dir}")

        # ---- Optional: balance background vs pothole ----
        if balance_background:
            pos = [s for s in self.samples if s.label == 1]
            neg = [s for s in self.samples if s.label == 0]

            if len(pos) > 0 and len(neg) > 0:
                max_neg = int(max_bg_ratio * len(pos))
                if len(neg) > max_neg:
                    neg = rng.sample(neg, k=max_neg)

                self.samples = pos + neg
                rng.shuffle(self.samples)

        print(
            f"[ProposalClassificationDataset] split={split}, "
            f"samples={len(self.samples)} "
            f"(pos={sum(s.label == 1 for s in self.samples)}, "
            f"bg={sum(s.label == 0 for s in self.samples)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        x_min, y_min, x_max, y_max = sample.bbox

        crop = img.crop((x_min, y_min, x_max, y_max))
        crop = self.transform(crop)

        label = torch.tensor(sample.label, dtype=torch.long)
        return crop, label
