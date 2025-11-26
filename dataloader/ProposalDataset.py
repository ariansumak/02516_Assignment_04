from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as T


ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class ProposalSample:
    image_id: str
    image_path: Path
    bbox: Tuple[int, int, int, int]
    label: str
    gt_bbox: Optional[Tuple[int, int, int, int]]


class ProposalDataset(Dataset):
    """
    Dataset that converts labelled proposals (generated in Part 1) into cropped tensors.
    Each sample corresponds to one region proposal paired with a class/background label.
    """

    def __init__(
        self,
        dataset_root: Path,
        proposals_dir: Path,
        *,
        image_subdir: str = "images",
        transform: Optional[ImageTransform] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        image_size: Tuple[int, int] = (128, 128),
        split: Optional[str] = None,
        splits_path: Optional[Path] = None,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / image_subdir
        self.proposals_dir = Path(proposals_dir)
        self.annotations_dir = self.dataset_root / "annotations"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory '{self.images_dir}' does not exist")
        if not self.proposals_dir.exists():
            raise FileNotFoundError(f"Proposals directory '{self.proposals_dir}' does not exist")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory '{self.annotations_dir}' does not exist")

        self.allowed_ids: Optional[set[str]] = None
        if split:
            resolved = splits_path or (self.dataset_root / "splits.json")
            self.allowed_ids = set(self._load_split_ids(resolved, split))
            if not self.allowed_ids:
                raise RuntimeError(f"Split '{split}' is empty or missing in {resolved}")
        if allowed_ids is not None:
            allowed_set = {Path(item).stem for item in allowed_ids}
            self.allowed_ids = allowed_set if self.allowed_ids is None else (self.allowed_ids & allowed_set)

        self.gt_annotations = self._load_gt_annotations()
        self.samples = self._load_samples()
        if self.allowed_ids is not None:
            self.samples = [sample for sample in self.samples if sample.image_id in self.allowed_ids]
        if not self.samples:
            raise RuntimeError(f"No proposals were found inside {self.proposals_dir}")

        self.class_to_idx = class_to_idx or self._build_class_index()
        missing = {sample.label for sample in self.samples} - set(self.class_to_idx)
        if missing:
            raise ValueError(f"Provided class mapping is missing labels: {missing}")
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.labels: List[int] = [self.class_to_idx[sample.label] for sample in self.samples]
        self.num_classes = len(self.class_to_idx)

        default_transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transform or default_transform

    def _load_samples(self) -> List[ProposalSample]:
        samples: List[ProposalSample] = []
        json_files = sorted(self.proposals_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files were found in {self.proposals_dir}. Run run_part1_pipeline.py first."
            )

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            image_id = payload.get("image_id")
            image_path = self._resolve_image_path(image_id)
            gt_boxes = self.gt_annotations.get(str(image_id), [])
            for entry in payload.get("proposals", []):
                bbox = entry.get("bbox")
                label = entry.get("label")
                if bbox is None or label is None:
                    continue
                if len(bbox) != 4:
                    continue
                bbox_tuple = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                )
                best_gt = self._match_gt_box(bbox_tuple, gt_boxes)
                samples.append(
                    ProposalSample(
                        image_id=str(image_id),
                        image_path=image_path,
                        bbox=bbox_tuple,
                        label=str(label),
                        gt_bbox=best_gt,
                    )
                )
        return samples

    def _resolve_image_path(self, image_id: Optional[str]) -> Path:
        if not image_id:
            raise ValueError("Proposal file is missing the 'image_id' field")
        candidates = list(self.images_dir.glob(f"{image_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"No image found for id '{image_id}' in {self.images_dir}")
        return candidates[0]

    def _build_class_index(self) -> Dict[str, int]:
        classes = sorted({sample.label for sample in self.samples})
        if "background" in classes:
            classes = ["background"] + [cls for cls in classes if cls != "background"]
        return {label: idx for idx, label in enumerate(classes)}

    @staticmethod
    def _load_split_ids(path: Path, split_name: str) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Splits file '{path}' was not found")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        split_entries = data.get(split_name)
        if split_entries is None:
            raise KeyError(f"Split '{split_name}' missing from {path}")
        if isinstance(split_entries, dict) and "images" in split_entries:
            split_entries = split_entries["images"]
        return [str(item) for item in split_entries]

    def _load_gt_annotations(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        gt: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for xml_path in sorted(self.annotations_dir.glob("*.xml")):
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            boxes: List[Tuple[int, int, int, int]] = []
            for obj in root.findall("object"):
                bbox_node = obj.find("bndbox")
                if bbox_node is None:
                    continue
                coords: List[int] = []
                for tag in ("xmin", "ymin", "xmax", "ymax"):
                    node = bbox_node.find(tag)
                    if node is None or node.text is None:
                        coords = []
                        break
                    coords.append(int(float(node.text)))
                if len(coords) != 4:
                    continue
                boxes.append(tuple(coords))  # type: ignore[arg-type]
            gt[xml_path.stem] = boxes
        return gt

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _match_gt_box(
        self,
        proposal: Tuple[int, int, int, int],
        gt_boxes: Sequence[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = self._compute_iou(proposal, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_box = gt_box
        return best_box

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        x_min, y_min, x_max, y_max = sample.bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = max(x_min + 1, x_max)
        y_max = max(y_min + 1, y_max)
        crop = image.crop((x_min, y_min, x_max, y_max))

        tensor = self.transform(crop) if self.transform else T.ToTensor()(crop)
        label_idx = self.class_to_idx[sample.label]

        proposal_box = torch.tensor(sample.bbox, dtype=torch.float32)
        if sample.gt_bbox is not None:
            target_box = torch.tensor(sample.gt_bbox, dtype=torch.float32)
            has_target = torch.tensor(1, dtype=torch.float32)
        else:
            target_box = torch.zeros(4, dtype=torch.float32)
            has_target = torch.tensor(0, dtype=torch.float32)
        return tensor, label_idx, proposal_box, target_box, has_target


class BalancedProposalSampler(Sampler[List[int]]):
    """
    Sampler that maintains a fixed background/foreground ratio per batch.
    """

    def __init__(
        self,
        dataset: ProposalDataset,
        batch_size: int,
        *,
        background_label: str = "background",
        background_fraction: float = 0.75,
        seed: int = 0,
    ) -> None:
        if batch_size < 2:
            raise ValueError("BalancedProposalSampler requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.background_fraction = background_fraction
        self.seed = seed

        bg_idx = dataset.class_to_idx.get(background_label)
        if bg_idx is None:
            raise ValueError(f"Dataset does not contain the '{background_label}' class")

        self.bg_indices = [i for i, lbl in enumerate(dataset.labels) if lbl == bg_idx]
        self.fg_indices = [i for i, lbl in enumerate(dataset.labels) if lbl != bg_idx]
        if not self.bg_indices or not self.fg_indices:
            raise ValueError("Balanced sampling requires at least one background and one foreground sample")

        self.bg_per_batch = max(1, int(round(batch_size * background_fraction)))
        self.fg_per_batch = max(1, batch_size - self.bg_per_batch)
        self.num_batches = max(1, math.ceil(len(dataset) / batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        bg_cycle = self._cycle(self.bg_indices, rng)
        fg_cycle = self._cycle(self.fg_indices, rng)
        for _ in range(self.num_batches):
            batch: List[int] = []
            for _ in range(self.bg_per_batch):
                batch.append(next(bg_cycle))
            for _ in range(self.fg_per_batch):
                batch.append(next(fg_cycle))
            rng.shuffle(batch)
            yield batch

    @staticmethod
    def _cycle(indices: Sequence[int], rng: random.Random) -> Iterator[int]:
        pool = list(indices)
        while True:
            rng.shuffle(pool)
            for idx in pool:
                yield idx
