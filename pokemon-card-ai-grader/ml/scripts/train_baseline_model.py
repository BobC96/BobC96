from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path("../data/processed/features.csv")
MODEL_OUTPUT = Path("../models/baseline_grade_predictor.pkl")


def main():
    df = pd.read_csv(FEATURES_PATH)

    feature_columns = [
        "blur_score",
        "brightness",
        "contrast",
        "avg_edge_whitening",
    ]

    X = df[feature_columns]
    y = df["psa_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    print(f"Mean Absolute Error: {mae:.2f}")

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)

    print(f"Model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
