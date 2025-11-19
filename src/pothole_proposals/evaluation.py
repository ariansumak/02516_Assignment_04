from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .dataset import ImageRecord

Box = Tuple[int, int, int, int]


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-Union for axis-aligned bounding boxes."""

    ax_min, ay_min, ax_max, ay_max = box_a
    bx_min, by_min, bx_max, by_max = box_b

    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_a = max(0, ax_max - ax_min) * max(0, ay_max - ay_min)
    area_b = max(0, bx_max - bx_min) * max(0, by_max - by_min)
    union = area_a + area_b - intersection
    if union == 0:
        return 0.0
    return intersection / union


def recall_at_k(
    proposals: Sequence[Box],
    ground_truth: Sequence[Box],
    *,
    top_k: int,
    iou_threshold: float,
) -> float:
    """Return the fraction of ground-truth boxes recalled by the top-k proposals."""

    if not ground_truth:
        return 1.0
    if not proposals:
        return 0.0

    proposals_limited = proposals[:top_k]
    hits = 0
    for gt_box in ground_truth:
        best_iou = max(compute_iou(gt_box, prop) for prop in proposals_limited)
        if best_iou >= iou_threshold:
            hits += 1
    return hits / len(ground_truth)


def evaluate_recall_curve(
    records: Sequence[ImageRecord],
    proposals_map: Mapping[str, Sequence[Box]],
    *,
    top_k_values: Iterable[int] = (50, 100, 200, 500, 1000, 2000),
    iou_thresholds: Iterable[float] = (0.5, 0.7),
) -> Dict[str, Dict[int, float]]:
    """
    Compute average recall for multiple IoU thresholds and proposal counts.
    Returns:
        {threshold: {k: average recall}}
    """

    recalls: Dict[float, Dict[int, List[float]]] = {
        thr: {k: [] for k in top_k_values} for thr in iou_thresholds
    }

    for record in records:
        proposals = proposals_map.get(record.image_id, [])
        gts = [tuple((box.x_min, box.y_min, box.x_max, box.y_max)) for box in record.annotations]
        for thr in iou_thresholds:
            for k in top_k_values:
                recalls[thr][k].append(
                    recall_at_k(proposals, gts, top_k=k, iou_threshold=thr)
                )

    averaged: Dict[str, Dict[int, float]] = {}
    for thr, stats in recalls.items():
        averaged[str(thr)] = {k: (mean(values) if values else 0.0) for k, values in stats.items()}
    return averaged
