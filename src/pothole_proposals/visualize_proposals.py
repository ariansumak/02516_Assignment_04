from pathlib import Path
import argparse

from src.pothole_proposals.dataset import DatasetIndex
from src.pothole_proposals.visualization import save_proposal_visualizations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data"
PROPOSALS_ROOT = PROJECT_ROOT / "outputs" / "part1" / "training_proposals" / "train"
OUT_DIR = PROJECT_ROOT / "outputs" / "part1" / "proposal_visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pothole proposals for a subset of the dataset.",
    )
    parser.add_argument(
        "--max-props-per-object",
        type=int,
        default=5,
        help="Maximum number of proposals to keep per ground-truth object (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # use the train split (or "test" if you prefer)
    dataset = DatasetIndex(DATASET_ROOT, split="train")
    saved = save_proposal_visualizations(
        dataset.records,
        proposals_dir=PROPOSALS_ROOT,
        output_dir=OUT_DIR,
        limit=6,          # how many images
        min_iou=0.3,      # how strict you want the overlap
        max_props_per_object=args.max_props_per_object,
    )
    print(f"Saved {len(saved)} proposal visualisations to {OUT_DIR}")


if __name__ == "__main__":
    main()
