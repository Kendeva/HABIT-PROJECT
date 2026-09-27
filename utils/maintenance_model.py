from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "issue_type","priority","queue_length",
    "technicians_available","request_hour","day_of_week"
]

def train_maintenance_model(df: pd.DataFrame) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["issue_type","priority"]),
        ("num", "passthrough", [
            "queue_length","technicians_available","request_hour","day_of_week"
        ]),
    ])
    model = RandomForestRegressor(
        n_estimators=240,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(df[FEATURES], df["waiting_time_minutes"])
    return pipe

def estimate_wait(model: Pipeline, values: dict) -> dict:
    row = pd.DataFrame([values], columns=FEATURES)
    minutes = max(5.0, float(model.predict(row)[0]))
    low = max(5.0, minutes * 0.85)
    high = minutes * 1.15
    if minutes < 45:
        label = "Short Wait"
    elif minutes < 120:
        label = "Moderate Wait"
    else:
        label = "Longer Wait"
    return {"minutes": minutes, "low": low, "high": high, "label": label}
