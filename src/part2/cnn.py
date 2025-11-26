# src/part2/cnn.py
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Make sure "src" is on the path when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from part2.proposal_dataset import ProposalClassificationDataset  # noqa: E402


# ---- Model ----
def build_model(num_classes: int = 2) -> nn.Module:
    """
    Simple classifier for proposal crops: background vs pothole.
    Uses a pretrained ResNet-18 as backbone.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---- Training / evaluation helpers ----
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main() -> None:
    # ---- Paths ----
    DATASET_ROOT = PROJECT_ROOT / "data"
    PROPOSALS_ROOT = PROJECT_ROOT / "outputs" / "part1" / "training_proposals"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Datasets and loaders ----
    # IMPORTANT:
    # * You must have run run_part1_pipeline.py with --split train
    #   (creates outputs/part1/training_proposals/train)
    # * For validation, either:
    #   - run run_part1_pipeline.py again with --split test, or
    #   - reuse train and manually change split here.
    train_dataset = ProposalClassificationDataset(
        dataset_root=DATASET_ROOT,
        proposals_root=PROPOSALS_ROOT,
        split="train",
        balance_background=True,
        max_bg_ratio=3.0,
    )

    # If you don't have proposals for "test" split yet, run:
    #   python run_part1_pipeline.py --split test
    val_dataset = ProposalClassificationDataset(
        dataset_root=DATASET_ROOT,
        proposals_root=PROPOSALS_ROOT,
        split="test",
        balance_background=False,  # don't rebalance on validation
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # ---- Model, loss, optimizer ----
    model = build_model(num_classes=2).to(device)

    # Simple CrossEntropy; if you want, you can add class weights here.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # Optionally save the model
    out_path = PROJECT_ROOT / "outputs" / "part2"
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path / "proposal_classifier_resnet18.pth")
    print(f"Saved model to {out_path / 'proposal_classifier_resnet18.pth'}")


if __name__ == "__main__":
    main()
