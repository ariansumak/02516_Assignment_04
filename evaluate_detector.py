from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Ensure local modules are importable (pothole_proposals lives in src/)
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pothole_proposals.dataset import DatasetIndex
from pothole_proposals.evaluation import compute_iou
from pothole_proposals.proposals import generate_selective_search_proposals
from train import build_model

DEFAULT_DATASET_ROOT = Path(r"/home/arian-sumak/Documents/DTU/computer vision/potholes_local_copy")

LOGGER = logging.getLogger("detector_eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained CNN on the test split, apply NMS, and compute AP.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root folder containing images/ and annotations/")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--checkpoint", type=Path, default=Path("eval/best_model"), help="Path to best model checkpoint (file or extracted directory)")
    parser.add_argument("--proposals-dir", type=Path, default=None, help="Optional directory with pre-computed proposals (.json per image).")
    parser.add_argument("--proposal-mode", choices=["fast", "quality"], default="fast", help="Selective Search mode when proposals are generated on the fly")
    parser.add_argument("--proposal-limit", type=int, default=2000, help="Maximum proposals per image")
    parser.add_argument("--min-box-size", type=int, default=20, help="Minimum proposal size (pixels)")
    parser.add_argument("--image-size", type=int, default=128, help="Proposal crop size (must match training)")
    parser.add_argument("--batch-size", type=int, default=128, help="Proposals processed per forward pass")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum pothole probability to keep a detection")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="IoU threshold for non-maximum suppression")
    parser.add_argument("--ap-iou", type=float, default=0.5, help="IoU threshold used for Average Precision")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device")
    parser.add_argument("--images-subdir", default="images", help="Dataset images subfolder")
    parser.add_argument("--annotations-subdir", default="annotations", help="Dataset annotations subfolder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"), help="Where to save detections and metrics")
    return parser.parse_args()


def build_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _pack_torch_directory(checkpoint_dir: Path) -> Path:
    """
    Torch checkpoints saved with the default zip format keep all contents in a
    top-level folder (matching the file name). If only the extracted directory
    is available, repack it temporarily so torch.load can read it.
    """

    temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    temp_file.close()
    prefix = checkpoint_dir.name
    with zipfile.ZipFile(temp_file.name, "w") as zf:
        for file_path in checkpoint_dir.rglob("*"):
            if not file_path.is_file():
                continue
            arcname = Path(prefix) / file_path.relative_to(checkpoint_dir)
            zf.write(file_path, arcname=str(arcname))
    return Path(temp_file.name)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' was not found")
    if path.is_file():
        return torch.load(path, map_location=device)
    packed = _pack_torch_directory(path)
    try:
        return torch.load(packed, map_location=device)
    finally:
        packed.unlink(missing_ok=True)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor, *, image_size: Tuple[int, int]) -> torch.Tensor:
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    pred_ctr_x = deltas[:, 0] * widths + ctr_x
    pred_ctr_y = deltas[:, 1] * heights + ctr_y
    pred_w = widths * deltas[:, 2].exp()
    pred_h = heights * deltas[:, 3].exp()

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    width, height = image_size
    x1 = x1.clamp(0, max(width - 1, 1))
    y1 = y1.clamp(0, max(height - 1, 1))
    x2 = x2.clamp(0, max(width - 1, 1))
    y2 = y2.clamp(0, max(height - 1, 1))
    return torch.stack((x1, y1, x2, y2), dim=1)


def box_iou_tensor(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    intersection = inter_w * inter_h

    area_a = torch.clamp(box[2] - box[0], min=0) * torch.clamp(box[3] - box[1], min=0)
    area_b = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0)
    union = area_a + area_b - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    if boxes.numel() == 0:
        return []
    keep: List[int] = []
    order = scores.argsort(descending=True)

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_tensor(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def load_proposals(image_id: str, proposals_dir: Path) -> List[Box]:
    json_path = proposals_dir / f"{image_id}.json"
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    boxes: List[Sequence[int]] = []
    if "boxes" in payload:
        boxes = payload["boxes"]
    elif "proposals" in payload:
        for entry in payload["proposals"]:
            if isinstance(entry, dict) and "bbox" in entry:
                boxes.append(entry["bbox"])
            elif isinstance(entry, (list, tuple)) and len(entry) == 4:
                boxes.append(entry)
    return [tuple(map(int, box)) for box in boxes if len(box) == 4]


def run_detector_on_image(
    model: torch.nn.Module,
    image: Image.Image,
    proposals: Sequence[Box],
    *,
    transform: T.Compose,
    device: torch.device,
    background_idx: int,
    score_threshold: float,
    nms_iou: float,
    batch_size: int,
) -> List[Tuple[Box, float]]:
    if not proposals:
        return []

    width, height = image.size
    detections: List[Tuple[Box, float]] = []
    model.eval()

    for start in range(0, len(proposals), batch_size):
        batch_props = proposals[start : start + batch_size]
        crops = [image.crop((x1, y1, x2, y2)).convert("RGB") for x1, y1, x2, y2 in batch_props]
        batch = torch.stack([transform(crop) for crop in crops]).to(device)
        prop_tensor = torch.tensor(batch_props, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, deltas = model(batch)
            probs = logits.softmax(dim=1)
            scores, labels = probs.max(dim=1)

        keep_mask = (labels != background_idx) & (scores >= score_threshold)
        if keep_mask.any():
            kept_props = prop_tensor[keep_mask]
            kept_deltas = deltas[keep_mask]
            kept_scores = scores[keep_mask]
            decoded = decode_boxes(kept_props, kept_deltas, image_size=(width, height))
            detections.extend(
                [
                    (tuple(map(int, decoded[i].tolist())), float(kept_scores[i].item()))
                    for i in range(decoded.size(0))
                ]
            )

    if not detections:
        return []

    det_boxes = torch.tensor([det[0] for det in detections], dtype=torch.float32)
    det_scores = torch.tensor([det[1] for det in detections], dtype=torch.float32)
    keep_indices = non_max_suppression(det_boxes, det_scores, nms_iou)
    return [detections[i] for i in keep_indices]


def compute_average_precision(
    detections: Sequence[Dict[str, object]],
    ground_truth: Dict[str, List[Box]],
    *,
    iou_threshold: float,
) -> Tuple[float, List[float], List[float]]:
    """
    Simple 11-point interpolated Average Precision for a single class detector.
    Returns (AP, precision curve, recall curve).
    """

    if not ground_truth:
        return 0.0, [], []
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    if total_gt == 0:
        return 0.0, [], []

    detections_sorted = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    matched: Dict[str, List[bool]] = {img_id: [False] * len(boxes) for img_id, boxes in ground_truth.items()}

    tp: List[float] = []
    fp: List[float] = []
    for det in detections_sorted:
        image_id = str(det["image_id"])
        box = tuple(det["box"])  # type: ignore[arg-type]
        gt_boxes = ground_truth.get(image_id, [])

        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0 and not matched[image_id][best_idx]:
            tp.append(1.0)
            fp.append(0.0)
            matched[image_id][best_idx] = True
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0, [], []

    tp_cum = torch.tensor(tp).cumsum(dim=0)
    fp_cum = torch.tensor(fp).cumsum(dim=0)
    recall = (tp_cum / total_gt).tolist()
    precision = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-8)).tolist()

    ap = 0.0
    for r_thresh in [i / 10 for i in range(11)]:  # 0.0 ... 1.0
        p_at_r = max((p for p, r in zip(precision, recall) if r >= r_thresh), default=0.0)
        ap += p_at_r / 11.0

    return ap, precision, recall


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)
    LOGGER.info("Evaluating on %s", device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_to_idx = checkpoint.get("class_to_idx", {"background": 0, "pothole": 1})
    background_idx = class_to_idx.get("background", 0)
    model = build_model(num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transform(args.image_size)
    dataset = DatasetIndex(
        args.dataset_root,
        split=args.split,
        images_subdir=args.images_subdir,
        annotations_subdir=args.annotations_subdir,
    )
    LOGGER.info("Loaded %d %s images", len(dataset), args.split)

    detections: List[Dict[str, object]] = []
    gt_map: Dict[str, List[Box]] = {
        record.image_id: [
            (box.x_min, box.y_min, box.x_max, box.y_max) for box in record.annotations if box.label != "background"
        ]
        for record in dataset.records
    }

    for record in tqdm(dataset.records, desc=f"Running detector on {args.split}"):
        image = Image.open(record.image_path).convert("RGB")
        width = record.width or image.width
        height = record.height or image.height

        proposals: List[Box] = []
        if args.proposals_dir:
            proposals = load_proposals(record.image_id, args.proposals_dir)
        if not proposals:
            proposals = generate_selective_search_proposals(
                record.image_path,
                mode=args.proposal_mode,
                max_proposals=args.proposal_limit,
                min_size=args.min_box_size,
            )
        proposals = proposals[: args.proposal_limit]

        per_image = run_detector_on_image(
            model,
            image,
            proposals,
            transform=transform,
            device=device,
            background_idx=background_idx,
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
            batch_size=args.batch_size,
        )
        for box, score in per_image:
            clipped_box = (
                max(0, min(int(box[0]), width)),
                max(0, min(int(box[1]), height)),
                max(0, min(int(box[2]), width)),
                max(0, min(int(box[3]), height)),
            )
            detections.append({"image_id": record.image_id, "box": clipped_box, "score": float(score)})

    ap, precision, recall = compute_average_precision(detections, gt_map, iou_threshold=args.ap_iou)
    metrics = {
        "split": args.split,
        "ap": ap,
        "ap_iou_threshold": args.ap_iou,
        "num_images": len(dataset),
        "num_detections": len(detections),
    }

    det_path = args.output_dir / f"{args.split}_detections.json"
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    with open(det_path, "w", encoding="utf-8") as handle:
        json.dump(detections, handle, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    LOGGER.info("AP@%.2f: %.4f over %d images (%d detections)", args.ap_iou, ap, len(dataset), len(detections))
    LOGGER.info("Saved detections to %s", det_path)
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
