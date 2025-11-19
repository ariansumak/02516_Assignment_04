import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.PotholeDataset import PotholeDataset, collate_fn

# Allow overriding the dataset location via environment variable.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
DATA_ROOT = Path(os.environ.get("POTHOLE_DATA_ROOT", DEFAULT_DATA_ROOT))

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


def test_dataset():
    """
    Load the dataset, iterate through it, and display the first image with its boxes.
    """
    # Paths to your dataset
    img_dir = os.path.join(DATA_ROOT, "images")
    ann_dir = os.path.join(DATA_ROOT, "annotations")
    
    # Create the dataset and DataLoader
    dataset = PotholeDataset(img_dir=img_dir, ann_dir=ann_dir)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Iterate over the first batch
    for images, targets in data_loader:
        print("Loaded 1 image with annotations")
        print(f"Image shape: {images[0].shape}")  # Shape of the image tensor (C, H, W)
        print(f"Bounding boxes: {targets[0]['boxes']}")
        print(f"Labels: {targets[0]['labels']}")
        print(f"Image filename: {targets[0]['filename']}")

        # Show the image with bounding boxes
        show_image_with_boxes(images[0], targets[0])

        break  # Just show the first batch for testing


if __name__ == "__main__":
    test_dataset()
