from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
import xml.etree.ElementTree as ET


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Simple Pascal-VOC bounding box expressed as inclusive pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str


@dataclass(frozen=True)
class ImageRecord:
    """Holds all metadata required for proposal extraction."""

    image_id: str
    image_path: Path
    annotation_path: Optional[Path]
    width: int
    height: int
    annotations: Sequence[BoundingBox]


def parse_voc_annotation(xml_path: Path) -> tuple[int, int, List[BoundingBox]]:
    """Parse a Pascal VOC style XML file and return (width, height, boxes)."""

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    width = _safe_int(size_node.findtext("width") if size_node is not None else None)
    height = _safe_int(size_node.findtext("height") if size_node is not None else None)

    boxes: List[BoundingBox] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="pothole").strip().lower()
        bbox_node = obj.find("bndbox")
        if bbox_node is None:
            continue
        coords = [
            _safe_int(bbox_node.findtext(tag))
            for tag in ("xmin", "ymin", "xmax", "ymax")
        ]
        if None in coords:
            continue
        x_min, y_min, x_max, y_max = coords  # type: ignore
        boxes.append(
            BoundingBox(
                x_min=int(x_min),
                y_min=int(y_min),
                x_max=int(x_max),
                y_max=int(y_max),
                label=name,
            )
        )

    return width, height, boxes


def _safe_int(value: Optional[str]) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _resolve_directory(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def discover_dataset(
    dataset_root: Path,
    *,
    images_subdir: Optional[str] = None,
    annotations_subdir: Optional[str] = None,
    splits_filename: Optional[str] = "splits.json",
) -> Dict[str, Path]:
    """
    Locate the canonical dataset folders. The function accepts optional overrides but
    will try to infer a sensible default (e.g. `images` vs `JPEGImages`).
    """

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist")

    images_dir = (
        dataset_root / images_subdir
        if images_subdir
        else _resolve_directory(dataset_root, ("images", "JPEGImages"))
    )
    annotations_dir = (
        dataset_root / annotations_subdir
        if annotations_subdir
        else _resolve_directory(dataset_root, ("annotations", "Annotations"))
    )

    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError("Could not locate the images directory inside the dataset")
    if annotations_dir is None or not annotations_dir.exists():
        LOGGER.warning(
            "Annotations directory was not found, proceeding without ground-truth boxes"
        )

    splits_path = None
    if splits_filename:
        candidate = dataset_root / splits_filename
        splits_path = candidate if candidate.exists() else None

    return {
        "root": dataset_root,
        "images": images_dir,
        "annotations": annotations_dir if annotations_dir else dataset_root,
        "splits": splits_path if splits_path else dataset_root / splits_filename,
    }


def load_split_ids(splits_path: Path, split_name: str) -> Optional[List[str]]:
    """Return the identifiers belonging to `split_name` from the JSON file."""

    if not splits_path.exists():
        return None

    with open(splits_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    split_data = data.get(split_name)
    if split_data is None:
        return None
    if isinstance(split_data, dict):
        # Some files wrap the list inside an "images" key.
        if "images" in split_data:
            split_data = split_data["images"]

    if not isinstance(split_data, Iterable):
        return None

    return [str(item) for item in split_data]


class DatasetIndex:
    """Convenience wrapper over the discovered dataset folders."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        split: str = "train",
        images_subdir: Optional[str] = None,
        annotations_subdir: Optional[str] = None,
        splits_filename: Optional[str] = "splits.json",
    ):
        paths = discover_dataset(
            dataset_root,
            images_subdir=images_subdir,
            annotations_subdir=annotations_subdir,
            splits_filename=splits_filename,
        )
        self.root = paths["root"]
        self.images_dir = paths["images"]
        self.annotations_dir = paths["annotations"]
        self.splits_path = paths["splits"]
        self.split = split

        self.records = self._build_records()

    def _build_records(self) -> List[ImageRecord]:
        image_files = {
            image.stem: image
            for image in self.images_dir.glob("*")
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

        selected_ids = load_split_ids(self.splits_path, self.split)
        if not selected_ids:
            LOGGER.warning(
                "Split '%s' missing in %s – defaulting to every discovered image",
                self.split,
                self.splits_path,
            )
            selected_ids = sorted(image_files.keys())

        records: List[ImageRecord] = []
        for image_id in selected_ids:
            img_path = image_files.get(image_id)
            if img_path is None:
                LOGGER.warning("Image '%s' from split not found on disk", image_id)
                continue
            ann_path = self._find_annotation(image_id)

            width = height = 0
            annotations: Sequence[BoundingBox] = ()
            if ann_path and ann_path.exists():
                width, height, annotations = parse_voc_annotation(ann_path)

            records.append(
                ImageRecord(
                    image_id=image_id,
                    image_path=img_path,
                    annotation_path=ann_path,
                    width=width,
                    height=height,
                    annotations=annotations,
                )
            )

        return records

    def _find_annotation(self, image_id: str) -> Optional[Path]:
        xml_path = self.annotations_dir / f"{image_id}.xml"
        if xml_path.exists():
            return xml_path
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    def iter_annotations(self) -> Iterator[BoundingBox]:
        for record in self.records:
            for box in record.annotations:
                yield box
