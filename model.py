from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from transformers import pipeline

HOME_MODEL = "DejanX13/vit-house-classifier"
MAINTENANCE_MODEL = Path("models/maintenance_cnn.pth")
PROPERTY_MODEL = Path("models/property_model.joblib")

PROPERTY_FEATURES = [
    "province",
    "subsidy_status",
    "land_area_m2",
    "building_area_m2",
    "bedrooms",
    "bathrooms",
    "floors",
    "property_type",
]

HOME_LABELS = {
    "dobre": "Good",
    "good": "Good",
    "srednje": "Fair",
    "medium": "Fair",
    "oronule": "Needs Attention",
    "ruined": "Needs Attention",
    "nepoznato": "Unknown",
    "unknown": "Unknown",
}

MAINTENANCE_LABELS = {
    "algae": "Algae",
    "major_crack": "Major Crack",
    "minor_crack": "Minor Crack",
    "peeling": "Peeling",
    "plain": "No Clear Defect",
    "spalling": "Spalling",
    "stain": "Stain",
}

MAINTENANCE_NOTES = {
    "algae": (
        "Check for moisture around the affected surface "
        "and keep the area dry and ventilated."
    ),
    "major_crack": (
        "Check whether the crack is widening or recurring. "
        "A closer building inspection may be needed."
    ),
    "minor_crack": (
        "Monitor the crack and compare it again later "
        "to see whether its size changes."
    ),
    "peeling": (
        "Check the paint or surface layer and look for "
        "possible moisture behind the affected area."
    ),
    "plain": (
        "No listed defect is clearly indicated in this image. "
        "Recheck if visible symptoms remain."
    ),
    "spalling": (
        "Check the damaged surface for loose material "
        "and whether the affected area is spreading."
    ),
    "stain": (
        "Check whether the stain is related to leakage, "
        "dampness, or a previous water issue."
    ),
}


@lru_cache
def load_home_model():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "image-classification",
        model=HOME_MODEL,
        device=device,
    )


def analyze_home(images, areas):
    predictions = load_home_model()(images)
    results = []

    for image, area, prediction in zip(images, areas, predictions):
        top = prediction[0]
        raw_label = str(top["label"]).lower()

        condition = HOME_LABELS.get(
            raw_label,
            raw_label.replace("_", " ").title(),
        )

        results.append({
            "image": image,
            "area": area,
            "condition": condition,
            "confidence": float(top["score"]),
        })

    scores = {
        "Good": 3,
        "Fair": 2,
        "Needs Attention": 1,
    }

    valid_results = [
        item
        for item in results
        if item["condition"] in scores
    ]

    if valid_results:
        average_score = sum(
            scores[item["condition"]]
            for item in valid_results
        ) / len(valid_results)

        if average_score >= 2.5:
            overall = "Good"
        elif average_score >= 1.5:
            overall = "Fair"
        else:
            overall = "Needs Attention"
    else:
        overall = "Unknown"

    confidence = (
        sum(item["confidence"] for item in results) / len(results)
        if results
        else 0
    )

    return {
        "overall_condition": overall,
        "average_confidence": confidence,
        "areas": results,
    }


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
        return self.classifier(self.features(x))


maintenance_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


def maintenance_model_ready():
    return MAINTENANCE_MODEL.exists()


@lru_cache
def load_maintenance_model():
    if not maintenance_model_ready():
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved = torch.load(
        MAINTENANCE_MODEL,
        map_location=device,
    )

    labels = saved["labels"]
    model = MaintenanceCNN(len(labels)).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()

    return model, labels, device


def analyze_maintenance(image):
    model, labels, device = load_maintenance_model()

    tensor = maintenance_transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(tensor),
            dim=1,
        )[0]

    index = int(probabilities.argmax())
    raw_label = labels[index]

    return {
        "raw_label": raw_label,
        "label": MAINTENANCE_LABELS.get(
            raw_label,
            raw_label.replace("_", " ").title(),
        ),
        "confidence": float(probabilities[index]),
        "note": MAINTENANCE_NOTES.get(
            raw_label,
            "Check the affected surface again if the issue remains visible.",
        ),
    }


def property_model_ready():
    return PROPERTY_MODEL.exists()


def estimate_property_price(info):
    saved = joblib.load(PROPERTY_MODEL)
    model = saved["model"]
    mae = float(saved["mae"])

    features = saved.get(
        "features",
        PROPERTY_FEATURES,
    )

    row = pd.DataFrame(
        [{
            "province": info["province"],
            "subsidy_status": info["subsidy_status"],
            "land_area_m2": info["land_area_m2"],
            "building_area_m2": info["building_area_m2"],
            "bedrooms": info["bedrooms"],
            "bathrooms": info["bathrooms"],
            "floors": info["floors"],
            "property_type": info["property_type"],
        }],
        columns=features,
    )

    prediction_log = float(
        model.predict(row)[0]
    )

    price = float(
        np.expm1(prediction_log)
    )

    range_low = float(
        saved.get("range_low", -0.20)
    )

    range_high = float(
        saved.get("range_high", 0.20)
    )

    minimum = float(
        np.expm1(
            prediction_log + range_low
        )
    )

    maximum = float(
        np.expm1(
            prediction_log + range_high
        )
    )

    return {
        "price": max(0, price),
        "min": max(0, minimum),
        "max": max(price, maximum),
        "mae": mae,
        "coverage": float(
            saved.get(
                "range_coverage",
                0,
            )
        ),
    }
