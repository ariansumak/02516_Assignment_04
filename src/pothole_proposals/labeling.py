from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .dataset import BoundingBox
from .evaluation import compute_iou

Box = Tuple[int, int, int, int]


def assign_labels(
    proposals: Sequence[Box],
    annotations: Sequence[BoundingBox],
    *,
    positive_iou: float = 0.5,
    negative_iou: float = 0.3,
) -> List[dict]:
    """
    Assign foreground/background labels to Selective Search proposals. The function
    returns a list of dictionaries containing the bounding box, label and IoU. Boxes
    in the ambiguous range between the thresholds are discarded.
    """

    labelled: List[dict] = []
    for box in proposals:
        best_iou = 0.0
        best_label = "background"
        for annotation in annotations:
            gt_box = (annotation.x_min, annotation.y_min, annotation.x_max, annotation.y_max)
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_label = annotation.label

        if best_iou >= positive_iou:
            label = best_label
        elif best_iou < negative_iou:
            label = "background"
        else:
            continue  # Ignore uncertain samples

        labelled.append({"bbox": box, "label": label, "iou": best_iou})

    return labelled
