from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["area","property_type","bedrooms","bathrooms","size_m2","furnished"]

def train_rental_model(df: pd.DataFrame) -> Pipeline:
    categorical = ["area","property_type","furnished"]
    numeric = ["bedrooms","bathrooms","size_m2"]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])
    model = RandomForestRegressor(
        n_estimators=260,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(df[FEATURES], df["monthly_rent"])
    return pipe

def _round_price(value: float, step: int = 100_000) -> float:
    """Round rental recommendations to a user-friendly Rupiah increment."""
    return round(value / step) * step


def estimate_rent(model: Pipeline, values: dict) -> dict:
    """Estimate a practical rental range and compare the listed price with it."""
    row = pd.DataFrame([values], columns=FEATURES)
    predicted = float(model.predict(row)[0])

    # A recommendation is shown as a range instead of a single exact price.
    # This avoids implying that the model can determine one perfectly correct rent.
    lower = _round_price(predicted * 0.90)
    upper = _round_price(predicted * 1.10)
    if lower == upper:
        upper = lower + 100_000

    listed = float(values.get("listed_rent", predicted))

    if listed > upper:
        status = "Higher Than Recommended"
        reference = upper
        difference_pct = ((listed - upper) / upper * 100) if upper else 0.0
    elif listed < lower:
        status = "Lower Than Recommended"
        reference = lower
        difference_pct = ((lower - listed) / lower * 100) if lower else 0.0
    else:
        status = "Within Recommended Range"
        reference = listed
        difference_pct = 0.0

    return {
        "lower": lower,
        "upper": upper,
        "difference_pct": difference_pct,
        "reference": reference,
        "status": status,
    }
