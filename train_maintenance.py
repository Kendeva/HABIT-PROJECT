import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

DATASET_NAME = "chandrabhuma/building_defect_vqa"
EPOCHS = 8
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


class DefectDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.label_to_id = {
            label: index
            for index, label in enumerate(labels)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        image = transform(
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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
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
    print("Loading building-defect dataset...")

    dataset = load_dataset(
        DATASET_NAME
    )

    train_data = dataset["train"]

    if "test" in dataset:
        test_data = dataset["test"]
    elif "validation" in dataset:
        test_data = dataset["validation"]
    else:
        split = train_data.train_test_split(
            test_size=0.2,
            seed=42,
        )
        train_data = split["train"]
        test_data = split["test"]

    labels = sorted(
        set(train_data["answer"])
    )

    train_loader = DataLoader(
        DefectDataset(train_data, labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        DefectDataset(test_data, labels),
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

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("Device:", device)
    print("Classes:", labels)

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

        accuracy, f1 = evaluate(
            model,
            test_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Accuracy {accuracy:.2%} | "
            f"F1 {f1:.3f}"
        )

    Path("models").mkdir(
        exist_ok=True
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
        },
        "models/maintenance_cnn.pth",
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(f1),
        "labels": labels,
    }

    Path(
        "models/maintenance_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved: models/maintenance_cnn.pth")


if __name__ == "__main__":
    main()
