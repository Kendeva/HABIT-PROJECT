import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# This is a SEPARATE experiment. It does not overwrite maintenance_cnn.pth
# or maintenance_metrics.json used by the current HABIT app.
BASE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "chandrabhuma/building_defect_vqa"
EPOCHS = 10
BATCH_SIZE = 16
SEED = 42

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.12, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class DefectDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.transform = transform
        self.label_to_id = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image = self.transform(row["image"].convert("RGB"))
        label = self.label_to_id[row["answer"]]
        return image, label


def make_model(num_classes, device):
    # Uses ImageNet pretrained weights, then replaces the final classifier.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            predictions = output.argmax(dim=1)
            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    return float(accuracy), float(macro_f1)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)
    full_train = dataset["train"]
    labels = sorted(set(full_train["answer"]))

    # Keep the same split logic as the original training script.
    if "test" in dataset:
        train_indices, val_indices = train_test_split(
            range(len(full_train)),
            test_size=0.15,
            random_state=SEED,
            stratify=full_train["answer"],
        )
        train_data = full_train.select(train_indices)
        val_data = full_train.select(val_indices)
        test_data = dataset["test"]
    else:
        train_val_indices, test_indices = train_test_split(
            range(len(full_train)),
            test_size=0.15,
            random_state=SEED,
            stratify=full_train["answer"],
        )
        train_val_labels = [full_train[i]["answer"] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.1765,
            random_state=SEED,
            stratify=train_val_labels,
        )
        train_data = full_train.select(train_indices)
        val_data = full_train.select(val_indices)
        test_data = full_train.select(test_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        DefectDataset(train_data, labels, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        DefectDataset(val_data, labels, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        DefectDataset(test_data, labels, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = make_model(len(labels), device)

    # Freeze the feature extractor for the first few epochs to stabilize
    # training on a smaller dataset; then fine-tune the final residual stage.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    counts = Counter(train_data["answer"])
    count_values = list(counts.values())
    use_weights = bool(count_values) and max(count_values) / min(count_values) >= 1.5

    if use_weights:
        total = len(train_data)
        weights = [total / (len(labels) * counts[label]) for label in labels]
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0003,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    print("Device:", device)
    print("Classes:", labels)
    print("Training class distribution:", dict(counts))
    print("Class weighting:", "Yes" if use_weights else "No")
    print("The original Custom CNN checkpoint will not be changed.")

    best_f1 = -1.0
    best_accuracy = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_accuracy, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss {total_loss / max(1, len(train_loader)):.4f} | "
            f"Val Accuracy {val_accuracy:.2%} | Val Macro F1 {val_f1:.3f}"
        )

        # Fine-tune the full network after the initial head/layer4 training.
        if epoch == 2:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0001, weight_decay=0.0001
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    test_accuracy, test_f1 = evaluate(model, test_loader, device)

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    # Separate filenames are intentional: app.py continues using the old CNN.
    checkpoint_path = models_dir / "maintenance_resnet18_experiment.pth"
    metrics_path = models_dir / "maintenance_resnet18_experiment_metrics.json"

    torch.save(
        {
            "model_state": best_state,
            "labels": labels,
            "architecture": "resnet18",
            "input_size": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        checkpoint_path,
    )

    metrics = {
        "model": "ResNet-18 pretrained on ImageNet",
        "experiment_only": True,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "validation_macro_f1": best_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1,
        "class_weighting": use_weights,
        "labels": labels,
        "checkpoint": str(checkpoint_path.relative_to(BASE_DIR)),
        "note": "Does not replace the Custom CNN used by the current app.",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(f"Best epoch            : {best_epoch}")
    print(f"Test accuracy         : {test_accuracy:.2%}")
    print(f"Test Macro F1         : {test_f1:.3f}")
    print(f"Experiment checkpoint : {checkpoint_path}")
    print(f"Experiment metrics    : {metrics_path}")
    print("Original model retained: models/maintenance_cnn.pth")


if __name__ == "__main__":
    main()
