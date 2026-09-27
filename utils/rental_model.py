import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "area",
    "property_type",
    "bedrooms",
    "bathrooms",
    "size_m2",
    "furnished",
]


def train_rental_model(data):
    categorical_features = ["area", "property_type", "furnished"]
    numeric_features = ["bedrooms", "bathrooms", "size_m2"]

    preprocessor = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("number", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(data[FEATURES], data["monthly_rent"])

    return pipeline


def round_price(value, step=100_000):
    return round(value / step) * step


def estimate_rent(model, values):
    input_data = pd.DataFrame([values], columns=FEATURES)
    predicted_rent = float(model.predict(input_data)[0])

    # Show a range instead of one exact recommendation.
    lower = round_price(predicted_rent * 0.90)
    upper = round_price(predicted_rent * 1.10)

    if lower == upper:
        upper = lower + 100_000

    listed_rent = float(values.get("listed_rent", predicted_rent))

    if listed_rent > upper:
        status = "Above Estimated Range"
        difference = (listed_rent - upper) / upper * 100 if upper else 0
    elif listed_rent < lower:
        status = "Below Estimated Range"
        difference = (lower - listed_rent) / lower * 100 if lower else 0
    else:
        status = "Within Estimated Range"
        difference = 0

    return {
        "lower": lower,
        "upper": upper,
        "difference_pct": difference,
        "status": status,
    }
