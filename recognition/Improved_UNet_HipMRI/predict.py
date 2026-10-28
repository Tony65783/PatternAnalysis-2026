"""Visualise HipMRI segmentation predictions from trained U-Net models."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

from dataset import HipMRIDataset
from modules import UNet, ImprovedUNet


NUM_CLASSES = 6

# Class colours are for visual distinction, not anatomical identification.
COLORS = ["black", "red", "green", "blue", "orange", "purple"]
MASK_CMAP = ListedColormap(COLORS)
MASK_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5), NUM_CLASSES)


def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = model_class(in_channels=1, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_mask(model, image, device):
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def save_comparison(index, image, truth, baseline, improved, output_dir):
    # False positive: prediction labels foreground where truth is background.
    false_positive = (improved != 0) & (truth == 0)
    # False negative: prediction labels background where truth is foreground.
    false_negative = (improved == 0) & (truth != 0)
    # Other mistakes: incorrect foreground class.
    wrong_class = (
        (improved != truth)
        & (improved != 0)
        & (truth != 0)
    )

    error_overlay = np.zeros((*truth.shape, 4), dtype=np.float32)
    error_overlay[false_positive] = [1, 0, 0, 0.8]
    error_overlay[false_negative] = [0, 0.4, 1, 0.8]
    error_overlay[wrong_class] = [1, 1, 0, 0.8]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI input")

    axes[1].imshow(truth, cmap=MASK_CMAP, norm=MASK_NORM)
    axes[1].set_title("Ground truth")

    axes[2].imshow(baseline, cmap=MASK_CMAP, norm=MASK_NORM)
    axes[2].set_title("Baseline U-Net")

    axes[3].imshow(improved, cmap=MASK_CMAP, norm=MASK_NORM)
    axes[3].set_title("Improved U-Net")

    axes[4].imshow(image, cmap="gray")
    axes[4].imshow(error_overlay)
    axes[4].set_title("Improved U-Net errors")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"HipMRI test slice {index}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_{index:04d}.png", dpi=160)
    plt.close(fig)

    return {
        "index": index,
        "baseline_errors": int((baseline != truth).sum()),
        "improved_errors": int((improved != truth).sum()),
        "false_positive": int(false_positive.sum()),
        "false_negative": int(false_negative.sum()),
        "wrong_class": int(wrong_class.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 100, 250, 400, 539])
    parser.add_argument("--output-dir", default="evaluation/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HipMRIDataset(split="test")

    baseline_model = load_model(
        UNet, "runs/baseline_v1/best_model.pth", device
    )
    improved_model = load_model(
        ImprovedUNet, "runs/improved_v1/best_model.pth", device
    )

    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise ValueError(f"Invalid index {index}; dataset has {len(dataset)} images")

        image, mask = dataset[index]
        truth = mask.numpy()
        baseline = predict_mask(baseline_model, image, device)
        improved = predict_mask(improved_model, image, device)

        result = save_comparison(
            index,
            image.squeeze(0).numpy(),
            truth,
            baseline,
            improved,
            output_dir,
        )
        print(result, flush=True)

    print(f"Figures saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
