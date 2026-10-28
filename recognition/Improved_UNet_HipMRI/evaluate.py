import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet, ImprovedUNet


NUM_CLASSES = 6


def evaluate(model, loader, device):
    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    predicted = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    actual = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    total_time = 0.0
    total_images = 0

    model.eval()

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start
            total_images += images.shape[0]

            for c in range(NUM_CLASSES):
                p = predictions == c
                t = masks == c

                intersection[c] += (p & t).sum().item()
                predicted[c] += p.sum().item()
                actual[c] += t.sum().item()

    dice = []
    iou = []

    for c in range(NUM_CLASSES):
        inter = intersection[c].item()
        pred = predicted[c].item()
        true = actual[c].item()

        dice_den = pred + true
        iou_den = pred + true - inter

        dice.append(
            2 * inter / dice_den if dice_den else float("nan")
        )
        iou.append(
            inter / iou_den if iou_den else float("nan")
        )

    valid_dice = [x for x in dice if x == x]
    valid_iou = [x for x in iou if x == x]

    return {
        "mean_dice": sum(valid_dice) / len(valid_dice),
        "mean_iou": sum(valid_iou) / len(valid_iou),
        "dice": dice,
        "iou": iou,
        "inference_ms_per_image": total_time / total_images * 1000,
        "images": total_images,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = HipMRIDataset(split="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    results = []

    for name, model_class in [
        ("baseline", UNet),
        ("improved", ImprovedUNet),
    ]:
        checkpoint_path = (
            Path("runs") / f"{name}_v1" / "best_model.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model = model_class(
            in_channels=1,
            num_classes=NUM_CLASSES,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        metrics = evaluate(model, loader, device)

        print(f"\n{name.upper()}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Test images: {metrics['images']}")
        print(f"Mean Dice: {metrics['mean_dice']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(
            "Per-class Dice:",
            [round(x, 4) for x in metrics["dice"]],
        )
        print(
            "Per-class IoU:",
            [round(x, 4) for x in metrics["iou"]],
        )
        print(
            f"Inference: "
            f"{metrics['inference_ms_per_image']:.2f} ms/image"
        )

        row = {
            "model": name,
            "checkpoint_epoch": checkpoint["epoch"],
            "test_images": metrics["images"],
            "mean_dice": metrics["mean_dice"],
            "mean_iou": metrics["mean_iou"],
            "inference_ms_per_image":
                metrics["inference_ms_per_image"],
        }

        for c in range(NUM_CLASSES):
            row[f"dice_class_{c}"] = metrics["dice"][c]
            row[f"iou_class_{c}"] = metrics["iou"][c]

        results.append(row)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(
        output_dir / "test_results.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to evaluation/test_results.csv")


if __name__ == "__main__":
    main()
