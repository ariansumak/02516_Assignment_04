import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.PotholeDataset import PotholeDataset, collate_fn

# Allow overriding the dataset location via environment variable.
DATA_ROOT = Path(os.environ.get("POTHOLE_DATA_ROOT", Path(__file__).resolve().parent / "data"))

def show_image_with_boxes(img, target):
    """
    Display an image with bounding boxes using matplotlib.
    """
    plt.imshow(img.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    boxes = target["boxes"]
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "r")
    plt.show()


def test_dataset(data_root=DATA_ROOT, max_images=1, shuffle=False):
    """
    Load the dataset, iterate through it, and display one or more images with their boxes.
    """
    data_root = Path(data_root)
    if max_images < 1:
        raise ValueError("max_images must be >= 1")

    img_dir = data_root / "images"
    ann_dir = data_root / "annotations"

    dataset = PotholeDataset(img_dir=str(img_dir), ann_dir=str(ann_dir))
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        print(f"Loaded image {batch_idx} with annotations")
        print(f"Image shape: {images[0].shape}")  # Shape of the image tensor (C, H, W)
        print(f"Bounding boxes: {targets[0]['boxes']}")
        print(f"Labels: {targets[0]['labels']}")
        print(f"Image filename: {targets[0]['filename']}")

        show_image_with_boxes(images[0], targets[0])

        if batch_idx >= max_images:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Preview pothole dataset samples.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root folder containing images/ and annotations/ (default: script directory / data)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1,
        help="How many images to display (default: 1)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before sampling images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_dataset(
        data_root=args.data_root,
        max_images=args.max_images,
        shuffle=args.shuffle,
    )
