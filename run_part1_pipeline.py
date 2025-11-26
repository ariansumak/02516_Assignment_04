from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATASET_ROOT = Path(r"/home/arian-sumak/Documents/DTU/computer vision/potholes_local_copy")

from tqdm import tqdm

from pothole_proposals.dataset import DatasetIndex
from pothole_proposals.evaluation import evaluate_recall_curve
from pothole_proposals.labeling import assign_labels
from pothole_proposals.proposals import generate_selective_search_proposals
from pothole_proposals.visualization import save_visualizations

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("part1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTU 02516 – Proposal pipeline (Part 1)")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        type=Path,
        help="Path to the potholes dataset root (defaults to /home/arian-sumak/Documents/DTU)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--output-dir", default=Path("outputs/part1"), type=Path, help="Destination for generated files")
    parser.add_argument("--visualization-count", type=int, default=6, help="How many images to visualise for Task 1")
    parser.add_argument(
        "--proposal-mode",
        choices=["fast", "quality"],
        default="fast",
        help="Selective Search mode – fast is recommended for the full dataset",
    )
    parser.add_argument(
        "--proposal-limit",
        type=int,
        default=2000,
        help="Maximum number of proposals retained per image",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=20,
        help="Filter proposals smaller than this edge length",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000],
        help="Proposal counts used for the recall evaluation curve",
    )
    parser.add_argument(
        "--eval-iou",
        type=float,
        nargs="+",
        default=[0.5, 0.7],
        help="IoU thresholds used for the recall evaluation",
    )
    parser.add_argument(
        "--positive-iou",
        type=float,
        default=0.5,
        help="IoU threshold for labelling a proposal as foreground",
    )
    parser.add_argument(
        "--negative-iou",
        type=float,
        default=0.3,
        help="IoU threshold for labelling a proposal as background",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DatasetIndex(args.dataset_root, split=args.split)
    LOGGER.info("Loaded split '%s' with %d images", args.split, len(dataset))

    output_dir = args.output_dir
    visualization_dir = output_dir / "visualizations"
    proposals_dir = output_dir / "proposals" / args.split
    evaluation_dir = output_dir / "evaluation"
    training_dir = output_dir / "training_proposals" / args.split

    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Task 1 – Visualise data with GT boxes
    saved = save_visualizations(
        dataset.records, visualization_dir, limit=args.visualization_count
    )
    LOGGER.info("Saved %d visualisations to %s", len(saved), visualization_dir)

    # Task 2 – Extract proposals
    proposals_map: Dict[str, List[tuple[int, int, int, int]]] = {}
    for record in tqdm(dataset, desc="Extracting proposals"):
        boxes = generate_selective_search_proposals(
            record.image_path,
            mode=args.proposal_mode,
            max_proposals=args.proposal_limit,
            min_size=args.min_box_size,
        )
        proposals_map[record.image_id] = boxes
        with open(proposals_dir / f"{record.image_id}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "image_id": record.image_id,
                    "boxes": [list(map(int, box)) for box in boxes],
                },
                handle,
            )

    # Task 3 – Evaluate recall curve
    recall = evaluate_recall_curve(
        dataset.records,
        proposals_map,
        top_k_values=args.eval_topk,
        iou_thresholds=args.eval_iou,
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with open(evaluation_dir / "recall.json", "w", encoding="utf-8") as handle:
        json.dump(recall, handle, indent=2)
    LOGGER.info("Evaluation results saved to %s", evaluation_dir / "recall.json")
    for thr, values in recall.items():
        LOGGER.info("IoU %s: %s", thr, ", ".join(f"k={k}: {v:.3f}" for k, v in values.items()))

    # Task 4 – Prepare training proposals
    for record in tqdm(dataset, desc="Labelling proposals"):
        proposals = proposals_map.get(record.image_id, [])
        labelled = assign_labels(
            proposals,
            record.annotations,
            positive_iou=args.positive_iou,
            negative_iou=args.negative_iou,
        )
        with open(training_dir / f"{record.image_id}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "image_id": record.image_id,
                    "proposals": [
                        {"bbox": list(map(int, entry["bbox"])), "label": entry["label"], "iou": entry["iou"]}
                        for entry in labelled
                    ],
                },
                handle,
            )
    LOGGER.info("Training proposals saved to %s", training_dir)


if __name__ == "__main__":
    main()
