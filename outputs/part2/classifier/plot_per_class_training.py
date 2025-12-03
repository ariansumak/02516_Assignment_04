import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    return history


def plot_per_class_curves(history, class_name, output_path):
    """
    Plot train loss vs per-class validation accuracy for a given class.
    """
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    class_acc = [entry["per_class_accuracy"][class_name] for entry in history]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left y-axis: training loss
    ax1.plot(epochs, train_loss, marker="o", linestyle="-", color="tab:blue", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right y-axis: per-class validation accuracy
    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        class_acc,
        marker="s",
        linestyle="-",
        color="tab:red",
        label=f"Val accuracy ({class_name})",
    )
    ax2.set_ylabel("Validation accuracy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    title_class = class_name.capitalize()
    ax1.set_title(f"Training loss and validation accuracy ({title_class})")

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    # Path to your history.json (adjust if needed)
    history_path = "./history.json"

    history = load_history(history_path)

    # Background graph
    plot_per_class_curves(
        history,
        class_name="background",
        output_path="training_curves_background.png",
    )

    # Pothole graph
    plot_per_class_curves(
        history,
        class_name="pothole",
        output_path="training_curves_pothole.png",
    )
