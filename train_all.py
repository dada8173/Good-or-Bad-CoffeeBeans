import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

DEFAULT_BEANS = ("ethiopia_washed", "honduras_natural")


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            if self.best_loss is not None:
                print(f"Loss decreased: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            # state_dict() is shallow; deepcopy preserves the actual best epoch.
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def pad_to_square(image, fill=0):
    width, height = image.size
    if width == height:
        return image
    diff = abs(height - width)
    pad1, pad2 = diff // 2, diff - diff // 2
    padding = (0, pad1, 0, pad2) if height < width else (pad1, 0, pad2, 0)
    return TF.pad(image, padding, fill=fill)


def get_model(num_classes: int, img_size: int):
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            feature_dim = 128 * (img_size // 8) * (img_size // 8)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(feature_dim, 256), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(256, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    return CustomCNN()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    matrix = [[0 for _ in classes] for _ in classes]
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        for truth, prediction in zip(labels.tolist(), predictions.tolist()):
            matrix[truth][prediction] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(classes)))
    per_class = {}
    for index, class_name in enumerate(classes):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    macro = {
        metric: sum(values[metric] for values in per_class.values()) / len(classes)
        for metric in ("precision", "recall", "f1")
    }
    weighted = {
        metric: sum(values[metric] * values["support"] for values in per_class.values()) / total
        for metric in ("precision", "recall", "f1")
    }
    return {
        "accuracy": correct / total,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": classes,
        "samples": total,
    }


def train_model(bean_type, dataset_path, model_dir, epochs=40, batch_size=256):
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    transform = transforms.Compose([
        transforms.Lambda(lambda image: pad_to_square(image, fill=0)),
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    if len(dataset.classes) != 2:
        raise ValueError(f"Expected two classes, found {dataset.classes} in {dataset_path}")

    # Preserve the project's existing 80/20 random split as requested.
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(len(dataset.classes), 128).to(device)
    class_counts = [0] * len(dataset.classes)
    for sample_index in train_dataset.indices:
        class_counts[dataset.targets[sample_index]] += 1
    class_weights = [
        train_size / (len(dataset.classes) * count) if count else 0.0
        for count in class_counts
    ]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    stopper = EarlyStopping(patience=10)
    history = []
    log_path = model_dir / f"{bean_type}_training.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        header = (
            f"Starting training for {bean_type} on {device}; samples={len(dataset)}, "
            f"train={train_size}, validation={validation_size}\n"
        )
        print(header, end="")
        log_file.write(header)
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
            record = {
                "epoch": epoch, "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            history.append(record)
            line = (
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} | val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.4f}\n"
            )
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            stopper(validation_loss, model, epoch)
            if stopper.early_stop:
                print("Early stopping triggered.")
                log_file.write("Early stopping triggered.\n")
                break

        model.load_state_dict(stopper.best_model_state)
        weights_name = f"{bean_type}_custom_Noback_best_model.pth"
        weights_path = model_dir / weights_name
        torch.save(stopper.best_model_state, weights_path)
        metrics = evaluate(model, validation_loader, device, dataset.classes)
        metrics.update({
            "bean_type": bean_type, "architecture": "custom", "input_size": 128,
            "classes": dataset.classes,
            "dataset_path": str(dataset_path.as_posix()),
            "split": "80/20 random split of the augmented dataset",
            "split_limitations": (
                "The existing project split was preserved. Augmented variants may cross "
                "the train/validation boundary, and no independent test set is used."
            ),
            "total_samples": len(dataset), "train_samples": train_size,
            "validation_samples": validation_size,
            "training_class_counts": dict(zip(dataset.classes, class_counts)),
            "class_weights": dict(zip(dataset.classes, class_weights)),
            "epochs_completed": len(history), "best_epoch": stopper.best_epoch,
            "best_validation_loss": stopper.best_loss,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
        })
        metrics_path = model_dir / f"{bean_type}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        history_path = model_dir / f"{bean_type}_history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        config_path = model_dir / f"{bean_type}_custom_Noback_best_model.json"
        config_path.write_text(json.dumps({
            "bean_type": bean_type, "architecture": "custom", "img_size": 128,
            "classes": dataset.classes, "model_file": weights_name,
            "metrics_file": metrics_path.name,
            "display_name": f"{bean_type.replace('_', ' ').title()} - Custom CNN (Noback)",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = (
            f"Best epoch={stopper.best_epoch}, val_loss={stopper.best_loss:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_avg']['f1']:.4f}\n"
            f"Saved: {weights_path}\n"
        )
        print(summary, end="")
        log_file.write(summary)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train coffee bean classifiers.")
    parser.add_argument("--beans", nargs="+", default=list(DEFAULT_BEANS))
    parser.add_argument("--data-root", default="coffee_beans_data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    for bean_type in args.beans:
        print(f"\n>>> Training {bean_type}")
        train_model(
            bean_type,
            Path(args.data_root) / bean_type / "corp_augmented_dataNoback",
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
