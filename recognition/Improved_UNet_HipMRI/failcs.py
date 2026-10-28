import csv
from pathlib import Path

import torch

from dataset import HipMRIDataset
from modules import ImprovedUNet


NUM_CLASSES = 6


def class_dice(prediction, target, label):
    pred = prediction == label
    true = target == label

    denominator = pred.sum().item() + true.sum().item()

    if denominator == 0:
        return None

    intersection = (pred & true).sum().item()
    return 2 * intersection / denominator


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = HipMRIDataset(split="test")

    checkpoint = torch.load(
        "runs/improved_v1/best_model.pth",
        map_location="cpu",
        weights_only=True,
    )

    model = ImprovedUNet(
        in_channels=1,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    results = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).squeeze(0).cpu()

            row = {
                "index": index,
                "filename": dataset.pairs[index][0].name,
            }

            scores = []

            for label in range(1, NUM_CLASSES):
                dice = class_dice(prediction, mask, label)
                row[f"dice_class_{label}"] = dice

                if dice is not None:
                    scores.append(dice)

            row["mean_foreground_dice"] = (
                sum(scores) / len(scores) if scores else None
            )

            results.append(row)

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "slice_dice_results.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    ranked = [
        row for row in results
        if row["mean_foreground_dice"] is not None
    ]
    ranked.sort(key=lambda row: row["mean_foreground_dice"])

    print("\nWorst 10 test slices by foreground Dice:")
    for row in ranked[:10]:
        print(
            f"Index {row['index']:3d} | "
            f"Dice {row['mean_foreground_dice']:.4f} | "
            f"{row['filename']}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

