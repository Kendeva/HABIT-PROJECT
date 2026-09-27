from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["residents","home_type","month","ac_hours_per_day","monthly_usage_kwh"]

def train_energy_models(df: pd.DataFrame):
    # Regression model gives an understandable expected-usage baseline.
    X_reg = df[["residents","home_type","month","ac_hours_per_day"]]
    y = df["monthly_usage_kwh"]
    prep_reg = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["home_type"]),
        ("num", "passthrough", ["residents","month","ac_hours_per_day"]),
    ])
    reg = Pipeline([
        ("prep", prep_reg),
        ("model", RandomForestRegressor(
            n_estimators=240, random_state=42, min_samples_leaf=3, n_jobs=-1
        )),
    ])
    reg.fit(X_reg, y)

    # Isolation Forest is fit on encoded reference observations.
    prep_iso = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["home_type"]),
        ("num", StandardScaler(), ["residents","month","ac_hours_per_day","monthly_usage_kwh"]),
    ])
    X_iso = prep_iso.fit_transform(df[FEATURES])
    iso = IsolationForest(
        n_estimators=220,
        contamination=0.04,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_iso)
    return reg, prep_iso, iso

def analyze_energy(reg, prep_iso, iso, values: dict) -> dict:
    reg_row = pd.DataFrame([{
        "residents": values["residents"],
        "home_type": values["home_type"],
        "month": values["month"],
        "ac_hours_per_day": values["ac_hours_per_day"],
    }])
    expected = float(reg.predict(reg_row)[0])

    iso_row = pd.DataFrame([values], columns=FEATURES)
    X = prep_iso.transform(iso_row)
    pred = int(iso.predict(X)[0])
    score = float(iso.decision_function(X)[0])
    actual = float(values["monthly_usage_kwh"])
    diff = ((actual - expected) / expected * 100) if expected else 0.0

    if pred == -1:
        status = "Unusual Usage"
    else:
        status = "Typical Usage"

    return {
        "expected": expected,
        "lower": expected * 0.88,
        "upper": expected * 1.12,
        "difference_pct": diff,
        "anomaly_score": score,
        "status": status,
    }
