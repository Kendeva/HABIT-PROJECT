import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATASET_NAME = "chandrabhuma/building_defect_vqa"
EPOCHS = 12
BATCH_SIZE = 16
SEED = 42

train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.10,
    ),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


class DefectDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.transform = transform
        self.label_to_id = {
            label: index
            for index, label in enumerate(labels)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        image = self.transform(
            row["image"].convert("RGB")
        )

        label = self.label_to_id[
            row["answer"]
        ]

        return image, label


class MaintenanceCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


def evaluate(model, loader, device):
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            prediction = output.argmax(dim=1)

            actual.extend(
                labels.cpu().tolist()
            )
            predicted.extend(
                prediction.cpu().tolist()
            )

    accuracy = accuracy_score(
        actual,
        predicted,
    )
    f1 = f1_score(
        actual,
        predicted,
        average="macro",
    )

    return accuracy, f1


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Loading building-defect dataset...")

    dataset = load_dataset(DATASET_NAME)
    full_train = dataset["train"]

    labels = sorted(
        set(full_train["answer"])
    )

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

        train_val_labels = [
            full_train[index]["answer"]
            for index in train_val_indices
        ]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.1765,
            random_state=SEED,
            stratify=train_val_labels,
        )

        train_data = full_train.select(train_indices)
        val_data = full_train.select(val_indices)
        test_data = full_train.select(test_indices)

    train_loader = DataLoader(
        DefectDataset(
            train_data,
            labels,
            train_transform,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        DefectDataset(
            val_data,
            labels,
            eval_transform,
        ),
        batch_size=BATCH_SIZE,
    )

    test_loader = DataLoader(
        DefectDataset(
            test_data,
            labels,
            eval_transform,
        ),
        batch_size=BATCH_SIZE,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MaintenanceCNN(
        len(labels)
    ).to(device)

    # Use class weighting only when the training distribution is clearly uneven.
    class_counts = Counter(
        train_data["answer"]
    )
    count_values = list(class_counts.values())
    imbalance_ratio = max(count_values) / min(count_values)

    use_class_weights = imbalance_ratio >= 1.5

    if use_class_weights:
        total = len(train_data)
        weights = [
            total / (len(labels) * class_counts[label])
            for label in labels
        ]

        weight_tensor = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        loss_function = nn.CrossEntropyLoss(
            weight=weight_tensor
        )
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("Device:", device)
    print("Classes:", labels)
    print("Training class distribution:", dict(class_counts))
    print("Class weighting:", "Yes" if use_class_weights else "No")

    best_f1 = -1.0
    best_accuracy = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = loss_function(
                output,
                targets,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            device,
        )

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
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Val Accuracy {val_accuracy:.2%} | "
            f"Val F1 {val_f1:.3f}"
        )

    model.load_state_dict(best_state)
    model = model.to(device)

    test_accuracy, test_f1 = evaluate(
        model,
        test_loader,
        device,
    )

    Path("models").mkdir(exist_ok=True)

    torch.save(
        {
            "model_state": best_state,
            "labels": labels,
        },
        "models/maintenance_cnn.pth",
    )

    metrics = {
        "best_epoch": best_epoch,
        "validation_accuracy": float(best_accuracy),
        "validation_macro_f1": float(best_f1),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_f1),
        "class_weighting": use_class_weights,
        "labels": labels,
    }

    Path(
        "models/maintenance_metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Best epoch        : {best_epoch}")
    print(f"Test accuracy     : {test_accuracy:.2%}")
    print(f"Test Macro F1     : {test_f1:.3f}")
    print("Saved: models/maintenance_cnn.pth")


if __name__ == "__main__":
    main()
