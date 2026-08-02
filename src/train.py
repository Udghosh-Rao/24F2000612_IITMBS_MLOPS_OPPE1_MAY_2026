import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder


def main(path, iteration):
    print(f"Loading iteration {iteration}...")

    df = pd.read_parquet(path)

    df = df.dropna(
        subset=[
            "target",
            "rolling_avg_10",
            "volume_sum_10",
            "stock_name",
        ]
    ).copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    df["stock_encoded"] = encoder.fit_transform(
        df[["stock_name"]]
    )

    times = df["timestamp"].sort_values().unique()
    cutoff = times[int(len(times) * 0.8)]

    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]

    features = [
        "rolling_avg_10",
        "volume_sum_10",
        "stock_encoded",
    ]

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Cutoff:", cutoff)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("stock-movement-predictor")

    configs = [
        {
            "learning_rate": 0.05,
            "max_iter": 40,
            "max_leaf_nodes": 15,
        },
        {
            "learning_rate": 0.10,
            "max_iter": 60,
            "max_leaf_nodes": 15,
        },
        {
            "learning_rate": 0.10,
            "max_iter": 80,
            "max_leaf_nodes": 31,
        },
    ]

    best_f1 = -1
    best_run = None
    best_model = None

    for number, params in enumerate(configs, 1):

        print(f"\nTuning run {number}/3")

        with mlflow.start_run(
            run_name=f"iteration-{iteration}-run-{number}"
        ) as run:

            model = HistGradientBoostingClassifier(
                random_state=42,
                **params,
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)

            mlflow.log_param("iteration", iteration)
            mlflow.log_param("split", "chronological")
            mlflow.log_params(params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            print("Accuracy:", round(accuracy, 4))
            print("F1:", round(f1, 4))
            print("Run ID:", run.info.run_id)

            if f1 > best_f1:
                best_f1 = f1
                best_run = run.info.run_id
                best_model = model

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "encoder": encoder,
            "features": features,
        },
        f"models/best_iteration_{iteration}.joblib",
    )

    print("\n==============================")
    print("BEST ITERATION", iteration)
    print("==============================")
    print("Run:", best_run)
    print("F1:", best_f1)

    if iteration == 2:
        registered = mlflow.register_model(
            model_uri=f"runs:/{best_run}/model",
            name="StockMovementPredictor",
        )

        print(
            "REGISTERED MODEL VERSION:",
            registered.version,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--iteration", required=True, type=int)

    args = parser.parse_args()

    main(args.input, args.iteration)
