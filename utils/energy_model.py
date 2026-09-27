import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "residents",
    "home_type",
    "month",
    "ac_hours_per_day",
    "monthly_usage_kwh",
]


def train_energy_models(data):
    regression_features = [
        "residents",
        "home_type",
        "month",
        "ac_hours_per_day",
    ]

    regression_preprocessor = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                ["home_type"],
            ),
            (
                "number",
                "passthrough",
                ["residents", "month", "ac_hours_per_day"],
            ),
        ]
    )

    regression_model = Pipeline(
        [
            ("preprocessor", regression_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    random_state=42,
                    min_samples_leaf=3,
                ),
            ),
        ]
    )
    regression_model.fit(
        data[regression_features],
        data["monthly_usage_kwh"],
    )

    # Isolation Forest uses the full household profile including actual usage.
    anomaly_preprocessor = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["home_type"],
            ),
            (
                "number",
                StandardScaler(),
                [
                    "residents",
                    "month",
                    "ac_hours_per_day",
                    "monthly_usage_kwh",
                ],
            ),
        ]
    )

    prepared_data = anomaly_preprocessor.fit_transform(data[FEATURES])

    anomaly_model = IsolationForest(
        n_estimators=180,
        contamination=0.04,
        random_state=42,
    )
    anomaly_model.fit(prepared_data)

    return regression_model, anomaly_preprocessor, anomaly_model


def analyze_energy(regression_model, anomaly_preprocessor, anomaly_model, values):
    regression_input = pd.DataFrame(
        [
            {
                "residents": values["residents"],
                "home_type": values["home_type"],
                "month": values["month"],
                "ac_hours_per_day": values["ac_hours_per_day"],
            }
        ]
    )

    expected_usage = float(regression_model.predict(regression_input)[0])

    anomaly_input = pd.DataFrame([values], columns=FEATURES)
    prepared_input = anomaly_preprocessor.transform(anomaly_input)
    prediction = int(anomaly_model.predict(prepared_input)[0])
    anomaly_score = float(anomaly_model.decision_function(prepared_input)[0])

    actual_usage = float(values["monthly_usage_kwh"])
    difference = (
        (actual_usage - expected_usage) / expected_usage * 100
        if expected_usage
        else 0
    )

    status = "Unusual Usage" if prediction == -1 else "Typical Usage"

    return {
        "expected": expected_usage,
        "lower": expected_usage * 0.88,
        "upper": expected_usage * 1.12,
        "difference_pct": difference,
        "anomaly_score": anomaly_score,
        "status": status,
    }
