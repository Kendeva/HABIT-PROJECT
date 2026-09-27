import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "issue_type",
    "priority",
    "queue_length",
    "technicians_available",
    "request_hour",
    "day_of_week",
]


def train_maintenance_model(data):
    preprocessor = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                ["issue_type", "priority"],
            ),
            (
                "number",
                "passthrough",
                [
                    "queue_length",
                    "technicians_available",
                    "request_hour",
                    "day_of_week",
                ],
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=180,
        random_state=42,
        min_samples_leaf=3,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(data[FEATURES], data["waiting_time_minutes"])

    return pipeline


def estimate_wait(model, values):
    input_data = pd.DataFrame([values], columns=FEATURES)
    estimated_minutes = max(5.0, float(model.predict(input_data)[0]))

    low = max(5.0, estimated_minutes * 0.85)
    high = estimated_minutes * 1.15

    if estimated_minutes < 45:
        label = "Short Wait"
    elif estimated_minutes < 120:
        label = "Moderate Wait"
    else:
        label = "Longer Wait"

    return {
        "minutes": estimated_minutes,
        "low": low,
        "high": high,
        "label": label,
    }
