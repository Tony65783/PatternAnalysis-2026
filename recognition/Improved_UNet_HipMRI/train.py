
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet, ImprovedUNet


NUM_CLASSES = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(outputs, masks):
    """Return intersection, predicted count, and actual count per class."""
    predictions = torch.argmax(outputs, dim=1)

    intersections = []
    predicted_counts = []
    actual_counts = []

    for class_id in range(NUM_CLASSES):
        predicted = predictions == class_id
        actual = masks == class_id

        intersections.append((predicted & actual).sum().item())
        predicted_counts.append(predicted.sum().item())
        actual_counts.append(actual.sum().item())

    return intersections, predicted_counts, actual_counts


def evaluate(model, loader, criterion, device):
    """Evaluate validation loss, Dice and IoU."""
    model.eval()

    total_loss = 0.0
    intersections = [0] * NUM_CLASSES
    predicted_counts = [0] * NUM_CLASSES
    actual_counts = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item() * images.size(0)

            batch_inter, batch_pred, batch_actual = calculate_metrics(
                outputs, masks
            )

            for i in range(NUM_CLASSES):
                intersections[i] += batch_inter[i]
                predicted_counts[i] += batch_pred[i]
                actual_counts[i] += batch_actual[i]

    avg_loss = total_loss / len(loader.dataset)

    dice_scores = []
    iou_scores = []

    for i in range(NUM_CLASSES):
        intersection = intersections[i]
        predicted = predicted_counts[i]
        actual = actual_counts[i]

        dice_denominator = predicted + actual
        iou_denominator = predicted + actual - intersection

        dice = (
            2 * intersection / dice_denominator
            if dice_denominator > 0 else float("nan")
        )

        iou = (
            intersection / iou_denominator
            if iou_denominator > 0 else float("nan")
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

    return avg_loss, dice_scores, iou_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["baseline", "improved"],
        default="baseline"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.model}_{int(time.time())}"

    if not all(c.isalnum() or c in "-_" for c in run_name):
        parser.error("Run name may only contain letters, numbers, - and _.")

    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Device:", device)
    print("Run:", run_name)

    # Load datasets
    train_dataset = HipMRIDataset(split="train")
    val_dataset = HipMRIDataset(split="validate")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # Choose model
    if args.model == "baseline":
        model = UNet()
    else:
        model = ImprovedUNet()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Model:", args.model)
    print("Parameters:", parameter_count)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Seed:", args.seed)

    log_path = os.path.join(output_dir, "metrics.csv")
    checkpoint_path = os.path.join(output_dir, "best_model.pth")

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "mean_dice",
        "mean_iou",
    ]

    for i in range(NUM_CLASSES):
        fieldnames.append(f"dice_class_{i}")
        fieldnames.append(f"iou_class_{i}")

    best_dice = -1.0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0.0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

            avg_train_loss = total_train_loss / len(train_dataset)

            val_loss, dice_scores, iou_scores = evaluate(
                model, val_loader, criterion, device
            )

            valid_dice = [
                dice_scores[i]
                for i in range(NUM_CLASSES)
                if dice_scores[i] == dice_scores[i]
            ]

            valid_iou = [
                iou_scores[i]
                for i in range(NUM_CLASSES)
                if iou_scores[i] == iou_scores[i]
            ]

            mean_dice = (
                sum(valid_dice) / len(valid_dice)
                if valid_dice else float("nan")
            )

            mean_iou = (
                sum(valid_iou) / len(valid_iou)
                if valid_iou else float("nan")
            )

            row = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "mean_dice": mean_dice,
                "mean_iou": mean_iou,
            }

            for i in range(NUM_CLASSES):
                row[f"dice_class_{i}"] = dice_scores[i]
                row[f"iou_class_{i}"] = iou_scores[i]

            writer.writerow(row)
            log_file.flush()

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Mean Dice: {mean_dice:.4f} | "
                f"Mean IoU: {mean_iou:.4f}",
                flush=True
            )

            print(
                "Class Dice:",
                [round(score, 4) for score in dice_scores],
                flush=True
            )

            if mean_dice > best_dice:
                best_dice = mean_dice

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_dice,
                        "dice_scores": dice_scores,
                        "iou_scores": iou_scores,
                        "args": vars(args),
                    },
                    checkpoint_path
                )

                print(
                    f"Saved best checkpoint (Dice: {best_dice:.4f})",
                    flush=True
                )

    if device.type == "cuda":
        print(
            "Peak GPU memory (MB):",
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        )

    print("Training completed.")
    print("Best validation Dice:", round(best_dice, 4))
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()

