"""
Utility helpers for DTU 02516 Assignment 3 – Part 1.

The submodules provide the following functionality:

```
dataset      – Discover the pothole dataset and parse Pascal-VOC annotations.
visualization– Plot images with their ground-truth bounding boxes.
proposals    – Selective Search proposal extraction with OpenCV and pure python fallback.
evaluation   – IoU utilities and recall/coverage statistics for proposal quality checks.
labeling     – Assign foreground/background labels to proposals for detector training.
```

These modules are intentionally lightweight so they can be imported from Jupyter
notebooks or the provided `run_part1_pipeline.py` script.
"""

from .dataset import (
    BoundingBox,
    DatasetIndex,
    ImageRecord,
    discover_dataset,
    load_split_ids,
    parse_voc_annotation,
)
from .evaluation import (
    compute_iou,
    evaluate_recall_curve,
    recall_at_k,
)
from .labeling import assign_labels
from .proposals import generate_selective_search_proposals
from .visualization import save_visualizations

__all__ = [
    "BoundingBox",
    "DatasetIndex",
    "ImageRecord",
    "discover_dataset",
    "load_split_ids",
    "parse_voc_annotation",
    "compute_iou",
    "evaluate_recall_curve",
    "recall_at_k",
    "assign_labels",
    "generate_selective_search_proposals",
    "save_visualizations",
]
