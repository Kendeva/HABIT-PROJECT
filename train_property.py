import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_NAME = "web3hungry/indonesia-affordable-housing"
MAX_RECORDS = 5000

FEATURES = [
    "province",
    "subsidy_status",
    "land_area_m2",
    "building_area_m2",
    "bedrooms",
    "bathrooms",
    "floors",
    "property_type",
]

TARGET = "price_idr"


def build_model():
    numeric_columns = [
        "land_area_m2",
        "building_area_m2",
        "bedrooms",
        "bathrooms",
        "floors",
    ]

    categorical_columns = [
        "province",
        "subsidy_status",
        "property_type",
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ])

    regressor = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.2,
        random_state=42,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", regressor),
    ])


def main():
    print("Loading Indonesian housing dataset...")

    dataset = load_dataset(DATASET_NAME)["train"]
    dataframe = dataset.to_pandas()
    dataframe = dataframe[FEATURES + [TARGET]].copy()

    dataframe[TARGET] = pd.to_numeric(
        dataframe[TARGET],
        errors="coerce",
    )

    numeric_features = [
        "land_area_m2",
        "building_area_m2",
        "bedrooms",
        "bathrooms",
        "floors",
    ]

    for column in numeric_features:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(subset=[TARGET])
    dataframe = dataframe[dataframe[TARGET] > 0]

    lower_limit = dataframe[TARGET].quantile(0.01)
    upper_limit = dataframe[TARGET].quantile(0.99)

    dataframe = dataframe[
        (dataframe[TARGET] >= lower_limit)
        & (dataframe[TARGET] <= upper_limit)
    ]

    if len(dataframe) > MAX_RECORDS:
        dataframe = dataframe.sample(
            n=MAX_RECORDS,
            random_state=42,
        )

    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    X_calibration, X_test, y_calibration, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
    )

    model = build_model()

    print(f"Records used: {len(dataframe):,}")
    print("Training property model...")

    # Log target reduces the effect of very expensive houses.
    model.fit(
        X_train,
        np.log1p(y_train),
    )

    calibration_log = model.predict(X_calibration)
    calibration_actual_log = np.log1p(y_calibration.to_numpy())

    residuals = (
        calibration_actual_log
        - calibration_log
    )

    # Calibrated 80% interval from real validation residuals.
    range_low = float(
        np.quantile(residuals, 0.10)
    )
    range_high = float(
        np.quantile(residuals, 0.90)
    )

    range_low = min(range_low, 0.0)
    range_high = max(range_high, 0.0)

    test_log = model.predict(X_test)
    test_predictions = np.expm1(test_log)

    mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    rmse = mean_squared_error(
        y_test,
        test_predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        test_predictions,
    )

    lower_predictions = np.expm1(
        test_log + range_low
    )
    upper_predictions = np.expm1(
        test_log + range_high
    )

    actual = y_test.to_numpy()

    coverage = np.mean(
        (actual >= lower_predictions)
        & (actual <= upper_predictions)
    )

    average_range_width = np.mean(
        upper_predictions - lower_predictions
    )

    Path("models").mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "mae": float(mae),
            "features": FEATURES,
            "range_low": range_low,
            "range_high": range_high,
            "range_coverage": float(coverage),
        },
        "models/property_model.joblib",
    )

    metrics = {
        "records_used": int(len(dataframe)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "estimated_range_coverage": float(coverage),
        "average_range_width": float(average_range_width),
    }

    Path(
        "models/property_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Training complete.")
    print(f"MAE      : Rp{mae:,.0f}")
    print(f"RMSE     : Rp{rmse:,.0f}")
    print(f"R²       : {r2:.4f}")
    print(f"Coverage : {coverage:.2%}")
    print(
        "Saved: models/property_model.joblib"
    )


if __name__ == "__main__":
    main()
