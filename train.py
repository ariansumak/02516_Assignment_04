
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from dataloader.ProposalDataset import BalancedProposalSampler, ProposalDataset

DEFAULT_DATASET_ROOT = os.getenv('DEFAULT_DATASET_ROOT', Path(r"/home/arian-sumak/Documents/DTU/computer vision/potholes_local_copy"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("proposal_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a proposal-level classifier (Part 2)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root folder containing the images/")
    parser.add_argument(
        "--images-subdir",
        default="images",
        help="Subdirectory inside dataset-root with the RGB images",
    )
    parser.add_argument(
        "--proposals-root",
        type=Path,
        default=Path("outputs/part1/training_proposals"),
        help="Directory containing labelled proposals. If your data is unsplit, point to this root and use --train-split/--val-split.",
    )
    parser.add_argument(
        "--train-proposals-dir",
        type=Path,
        default=None,
        help="Optional override for training proposals directory if already split.",
    )
    parser.add_argument(
        "--val-proposals-dir",
        type=Path,
        default=None,
        help="Optional override for validation proposals directory if already split.",
    )
    parser.add_argument("--splits-file", type=Path, default=None, help="Path to splits.json (defaults to dataset-root/splits.json)")
    parser.add_argument("--train-split", default="train", help="Split name for training proposals (used when directories are shared)")
    parser.add_argument("--val-split", default="val", help="Split name for validation proposals (used when directories are shared)")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for automatic splitting when no splits file is available",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=13,
        help="Random seed for automatic train/val splitting",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam")
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.75,
        help="Background proportion enforced per mini-batch",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/part2/classifier"),
        help="Where to store checkpoints and logs",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device – defaults to auto-detecting CUDA",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (only effective on CUDA)",
    )
    return parser.parse_args()


class ProposalNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cls_head = nn.Linear(feature_dim, num_classes)
        self.box_head = nn.Linear(feature_dim, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.cls_head(features)
        bbox = self.box_head(features)
        return logits, bbox


def build_model(num_classes: int, pretrained: bool = True) -> ProposalNet:
    return ProposalNet(num_classes=num_classes, pretrained=pretrained)


def encode_boxes(proposals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Encode target boxes relative to proposals (dx, dy, dw, dh)."""
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    target_widths = (targets[:, 2] - targets[:, 0]).clamp(min=1.0)
    target_heights = (targets[:, 3] - targets[:, 1]).clamp(min=1.0)
    target_ctr_x = targets[:, 0] + 0.5 * target_widths
    target_ctr_y = targets[:, 1] + 0.5 * target_heights

    dx = (target_ctr_x - ctr_x) / widths
    dy = (target_ctr_y - ctr_y) / heights
    dw = torch.log(target_widths / widths)
    dh = torch.log(target_heights / heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def train_one_epoch(
    model: ProposalNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    background_idx: int,
    *,
    use_amp: bool,
    scaler: Optional[GradScaler],
) -> Dict[str, float]:
    model.train()
    running_cls = 0.0
    running_bbox = 0.0
    running_total = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, labels, proposals, targets, has_target in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        proposals = proposals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        has_target = has_target.to(device, non_blocking=True) > 0.5

        optimiser.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits, bbox_preds = model(images)
            cls_loss = criterion(logits, labels)
            reg_targets = encode_boxes(proposals, targets)
            positive_mask = (labels != background_idx) & has_target
            if positive_mask.any():
                bbox_loss = F.smooth_l1_loss(bbox_preds[positive_mask], reg_targets[positive_mask])
            else:
                bbox_loss = torch.tensor(0.0, device=device)
            loss = cls_loss + bbox_loss

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        cls_val = float(cls_loss.detach())
        bbox_val = float(bbox_loss.detach())
        total_val = cls_val + bbox_val

        running_cls += cls_val * images.size(0)
        running_bbox += bbox_val * images.size(0)
        running_total += total_val * images.size(0)
        progress.set_postfix(loss=total_val, cls=cls_val, bbox=bbox_val)
    dataset_size = max(1, len(loader.dataset))
    return {
        "loss": running_total / dataset_size,
        "cls_loss": running_cls / dataset_size,
        "bbox_loss": running_bbox / dataset_size,
    }


@torch.no_grad()
def evaluate(
    model: ProposalNet,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str],
    background_idx: int,
    *,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}
    running_cls = 0.0
    running_bbox = 0.0

    progress = tqdm(loader, desc="Validation", leave=False)
    for images, labels, proposals, targets, has_target in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        proposals = proposals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        has_target = has_target.to(device, non_blocking=True) > 0.5

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits, bbox_preds = model(images)
        preds = logits.argmax(dim=1)
        matches = preds == labels
        correct += matches.sum().item()
        total += labels.numel()
        for label, match in zip(labels.cpu().tolist(), matches.cpu().tolist()):
            per_class_total[label] = per_class_total.get(label, 0) + 1
            if match:
                per_class_correct[label] = per_class_correct.get(label, 0) + 1

        with autocast(enabled=use_amp and device.type == "cuda"):
            cls_loss = F.cross_entropy(logits, labels, reduction="mean")
            reg_targets = encode_boxes(proposals, targets)
            positive_mask = (labels != background_idx) & has_target
            if positive_mask.any():
                bbox_loss = F.smooth_l1_loss(bbox_preds[positive_mask], reg_targets[positive_mask])
            else:
                bbox_loss = torch.tensor(0.0, device=device)
        running_cls += float(cls_loss.detach()) * images.size(0)
        running_bbox += float(bbox_loss.detach()) * images.size(0)

    dataset_size = max(1, len(loader.dataset))
    per_class_accuracy = {
        idx_to_class[label]: per_class_correct.get(label, 0) / max(1, per_class_total.get(label, 0))
        for label in sorted(per_class_total)
    }
    return {
        "accuracy": correct / max(1, total),
        "per_class_accuracy": per_class_accuracy,
        "cls_loss": running_cls / dataset_size,
        "bbox_loss": running_bbox / dataset_size,
    }


def auto_split_ids(proposals_dir: Path, val_fraction: float, seed: int) -> Tuple[Sequence[str], Sequence[str]]:
    json_files = sorted(proposals_dir.glob("*.json"))
    if len(json_files) < 2:
        raise RuntimeError("Automatic split requires at least two proposal files")
    image_ids = [path.stem for path in json_files]
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * val_fraction)))
    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]
    if not train_ids:
        raise RuntimeError("Automatic split produced an empty training set – adjust val_fraction")
    return train_ids, val_ids


def count_json_files(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.json"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_file = args.splits_file or (args.dataset_root / "splits.json")
    splits_available = splits_file.exists()

    def resolve_dir(override: Optional[Path], split_name: str) -> tuple[Path, Optional[str]]:
        if override is not None:
            if not override.exists():
                raise FileNotFoundError(f"Proposals directory '{override}' does not exist")
            return override, None
        base_dir = args.proposals_root
        if not base_dir.exists():
            raise FileNotFoundError(f"Proposals root '{base_dir}' does not exist")
        candidate = base_dir / split_name if split_name else None
        if candidate is not None and candidate.exists():
            return candidate, None
        return base_dir, split_name

    train_dir, train_split = resolve_dir(args.train_proposals_dir, args.train_split)
    val_dir, val_split = resolve_dir(args.val_proposals_dir, args.val_split)
    auto_train_ids: Optional[Sequence[str]] = None
    auto_val_ids: Optional[Sequence[str]] = None
    val_json_count = count_json_files(val_dir)
    train_json_count = count_json_files(train_dir)
    needs_auto_split = (
        not splits_available
        and (train_dir == val_dir or val_json_count < 2)
    )
    if needs_auto_split:
        split_source = train_dir if train_json_count >= 2 else val_dir
        LOGGER.warning(
            "Splits file %s not found or val directory empty; generating an automatic %.0f/%.0f train/val split from %s",
            splits_file,
            (1 - args.val_fraction) * 100,
            args.val_fraction * 100,
            split_source,
        )
        auto_train_ids, auto_val_ids = auto_split_ids(split_source, args.val_fraction, args.split_seed)
        train_dir = split_source
        val_dir = split_source

    LOGGER.info("Loading proposal datasets...")
    train_dataset = ProposalDataset(
        dataset_root=args.dataset_root,
        proposals_dir=train_dir,
        image_subdir=args.images_subdir,
        split=train_split if splits_available else None,
        splits_path=splits_file if (splits_available and train_split) else None,
        allowed_ids=auto_train_ids,
    )
    val_dataset = ProposalDataset(
        dataset_root=args.dataset_root,
        proposals_dir=val_dir,
        image_subdir=args.images_subdir,
        class_to_idx=train_dataset.class_to_idx,
        split=val_split if splits_available else None,
        splits_path=splits_file if (splits_available and val_split) else None,
        allowed_ids=auto_val_ids,
    )

    train_sampler = BalancedProposalSampler(
        train_dataset,
        batch_size=args.batch_size,
        background_fraction=args.background_fraction,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        device = torch.device(args.device)
    LOGGER.info("Training on %s", device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        LOGGER.warning("AMP requested but CUDA not available; falling back to full precision on CPU")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(train_dataset.num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    background_idx = train_dataset.class_to_idx.get("background", 0)
    scaler = GradScaler(enabled=use_amp)

    best_val = 0.0
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            epoch,
            background_idx,
            use_amp=use_amp,
            scaler=scaler,
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            train_dataset.idx_to_class,
            background_idx,
            use_amp=use_amp,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_cls_loss": train_stats["cls_loss"],
                "train_bbox_loss": train_stats["bbox_loss"],
            }
        )
        history.append(metrics)

        LOGGER.info(
            "Epoch %d/%d - train loss %.4f (cls %.4f, bbox %.4f) - val acc %.3f",
            epoch,
            args.epochs,
            train_stats["loss"],
            train_stats["cls_loss"],
            train_stats["bbox_loss"],
            metrics["accuracy"],
        )

        if metrics["accuracy"] > best_val:
            best_val = metrics["accuracy"]
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimiser.state_dict(),
                "epoch": epoch,
                "val_accuracy": best_val,
                "class_to_idx": train_dataset.class_to_idx,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if history:
        epochs = [entry["epoch"] for entry in history]
        train_loss_values = [entry["train_loss"] for entry in history]
        val_acc_values = [entry["accuracy"] for entry in history]
        fig, ax1 = plt.subplots()
        ax1.plot(epochs, train_loss_values, "b-o", label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_acc_values, "r-s", label="Val Accuracy")
        ax2.set_ylabel("Validation Accuracy", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        fig.tight_layout()
        curve_path = output_dir / "training_curves.png"
        fig.savefig(curve_path)
        plt.close(fig)
        LOGGER.info("Saved training/validation curves to %s", curve_path)

    LOGGER.info("Best validation accuracy: %.3f", best_val)
    LOGGER.info("Saved metrics to %s", output_dir / "history.json")


if __name__ == "__main__":
    main()
