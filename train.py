"""
FINAL TRAINING SCRIPT - Retinal Disease Detection Using OCT Images

Run:
    python train.py
    python train.py --model resnet18 --epochs 5
    python train.py --model resnet18 --epochs 1 --max-train-samples 1000 --batch-size 16

Saves:
    models/<model>_best.pth
    outputs/<model>_metrics.json
    outputs/<model>_confusion_matrix.png
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import IMAGE_SIZE, MODEL_DIR, OUTPUT_DIR
from data_utils import (
    HuggingFaceOCTDataset, get_class_names, load_oct_dataset, set_seed,
)
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an OCT retinal disease classifier."
    )
    parser.add_argument(
        "--model", default="resnet18",
        choices=["custom_cnn", "resnet18", "vgg16", "mobilenet_v2", "densenet121"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="0 uses the full training split; otherwise uses a stratified subset.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use class-weighted loss for Custom CNN only (DRUSEN experiment).",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Class-weight strength: 1.0=inverse frequency; 0.5=gentler.",
    )
    return parser.parse_args()


def create_loaders(dataset, args, device):
    train_split = dataset["train"]
    test_split = dataset["test"]
    labels = np.asarray(train_split["label"])
    all_indices = np.arange(len(labels))

    # Correctly select the requested number of samples, stratified by class.
    if 0 < args.max_train_samples < len(all_indices):
        if args.max_train_samples < len(np.unique(labels)):
            raise ValueError("--max-train-samples must be at least the number of classes.")
        selected, _ = train_test_split(
            all_indices,
            train_size=args.max_train_samples,
            stratify=labels,
            random_state=args.seed,
        )
    else:
        selected = all_indices

    selected_labels = labels[selected]
    train_indices, val_indices = train_test_split(
        selected,
        test_size=0.15,
        stratify=selected_labels,
        random_state=args.seed,
    )

    train_ds = HuggingFaceOCTDataset(
        train_split, indices=train_indices,
        image_size=args.image_size, training=True,
    )
    val_ds = HuggingFaceOCTDataset(
        train_split, indices=val_indices,
        image_size=args.image_size, training=False,
    )
    test_ds = HuggingFaceOCTDataset(
        test_split, image_size=args.image_size, training=False,
    )

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    print(f"Training samples:   {len(train_ds):,}")
    print(f"Validation samples: {len(val_ds):,}")
    print(f"Testing samples:    {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_count

    return total_loss / max(total, 1), correct / max(total, 1)


def calculate_loss(model, loader, criterion, device):
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
    return loss_sum / max(count, 1)


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            outputs = model(images.to(device))
            predicted.extend(outputs.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    return actual, predicted


def save_confusion_matrix(actual, predicted, class_names, output_path, model_name):
    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(class_names)))
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix - {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def calculate_class_weights(dataset, class_names, weight_power=1.0):
    """
    Calculate class weights using labels from the training dataset.
    """
    labels = []

    for index in range(len(dataset)):
        sample = dataset[index]
        labels.append(int(sample[1]))

    labels = torch.tensor(labels, dtype=torch.long)

    counts = torch.bincount(
        labels,
        minlength=len(class_names)
    ).float()

    weights = (
        counts.sum() /
        (len(class_names) * counts.clamp_min(1))
    ) ** weight_power

    weights = weights / weights.mean()

    print("Training class counts:", counts.tolist())
    print("Calculated class weights:", weights.tolist())

    return weights

def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("Epochs/batch size must be positive and image size at least 32.")
    if args.weight_power < 0:
        raise ValueError("--weight-power must be non-negative.")
    if args.weighted_loss and args.model != "custom_cnn":
        raise ValueError("--weighted-loss is currently supported only for custom_cnn.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("RETINAL OCT DISEASE CLASSIFICATION")
    print("Model:", args.model, "| Device:", device)
    print("Weighted loss:", "ON" if args.weighted_loss else "OFF")
    print("=" * 60)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading OCT dataset...")
    dataset = load_oct_dataset()
    class_names = get_class_names(dataset["train"])
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_loader, val_loader, test_loader = create_loaders(dataset, args, device)

    print("Building model...")
    model = build_model(
        name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    class_weights = None
    if args.weighted_loss:
        class_weights = calculate_class_weights(
            train_loader.dataset, class_names, args.weight_power
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.5
    )

    checkpoint_path = Path(MODEL_DIR) / f"{args.model}_best.pth"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = calculate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "validation_loss": float(val_loss),
        })
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "image_size": args.image_size,
                "weighted_loss": args.weighted_loss,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None else None
                ),
            }, checkpoint_path)
            print("Best checkpoint saved:", checkpoint_path)
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement: {epochs_without_improvement}/{args.patience}")
            if epochs_without_improvement >= args.patience:
                print("Early stopping activated.")
                break

    print("\nLoading best checkpoint and evaluating on test data...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    actual, predicted = predict(model, test_loader, device)

    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="weighted", zero_division=0
    )
    report = classification_report(
        actual, predicted,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    elapsed = time.time() - start_time

    results = {
        "model": args.model,
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "test_samples": len(actual),
        "class_names": class_names,
        "training_history": history,
        "training_time_seconds": round(elapsed, 2),
        "weighted_loss": args.weighted_loss,
        "weight_power": args.weight_power if args.weighted_loss else None,
        "class_weights": (
            class_weights.detach().cpu().tolist()
            if class_weights is not None else None
        ),
        "classification_report": report,
    }

    metrics_path = Path(OUTPUT_DIR) / f"{args.model}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    matrix_path = Path(OUTPUT_DIR) / f"{args.model}_confusion_matrix.png"
    save_confusion_matrix(actual, predicted, class_names, matrix_path, args.model)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(classification_report(
        actual, predicted,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    ))
    print(f"Test accuracy:      {accuracy * 100:.2f}%")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall:    {recall:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}")
    if "DRUSEN" in report:
        print(f"DRUSEN precision:   {report['DRUSEN']['precision']:.4f}")
        print(f"DRUSEN recall:      {report['DRUSEN']['recall']:.4f}")
        print(f"DRUSEN F1-score:    {report['DRUSEN']['f1-score']:.4f}")
    print(f"Elapsed time:       {elapsed / 60:.2f} minutes")
    print("\nSaved model:", checkpoint_path)
    print("Saved metrics:", metrics_path)
    print("Saved confusion matrix:", matrix_path)
    print("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
