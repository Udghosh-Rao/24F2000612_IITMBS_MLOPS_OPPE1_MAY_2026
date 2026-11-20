"""Train, tune and track the stock movement classifier with MLflow.

Pipeline:
  chronological split  -> train (72%) / validation (8%) / test (20%)
  per config           -> fit on train, tune decision threshold on validation,
                          report metrics on the untouched test window
  best run (by val F1) -> saved as a joblib bundle and optionally registered
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    BASE_FEATURES,
    EXPERIMENT_NAME,
    HYPERPARAMS,
    MODEL_FEATURES,
    MODEL_NAME,
    RANDOM_STATE,
    TRACKING_URI,
)


def chronological_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.1):
    """Split by timestamp so no future data leaks into training."""
    times = np.sort(df["timestamp"].unique())
    test_cut = times[int(len(times) * (1 - test_frac))]
    train_val = df[df["timestamp"] < test_cut]
    test = df[df["timestamp"] >= test_cut]

    tv_times = np.sort(train_val["timestamp"].unique())
    val_cut = tv_times[int(len(tv_times) * (1 - val_frac))]
    return (
        train_val[train_val["timestamp"] < val_cut],
        train_val[train_val["timestamp"] >= val_cut],
        test,
    )


def best_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 on validation data."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
    }


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["target", "stock_name", *BASE_FEATURES]).copy()
    df["target"] = df["target"].astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def main(path, iteration, register=None):
    import mlflow
    import mlflow.sklearn

    df = load_dataset(path)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["stock_encoded"] = encoder.fit_transform(df[["stock_name"]]).ravel()

    features = [f for f in MODEL_FEATURES if f in df.columns]
    train, val, test = chronological_split(df)
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")
    print(f"Features: {features}")

    # Naive baseline: always predict "up". Any useful model must beat this F1.
    baseline_f1 = f1_score(test["target"], np.ones(len(test)), zero_division=0)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best = {"val_f1": -1.0}
    for number, params in enumerate(HYPERPARAMS, 1):
        with mlflow.start_run(run_name=f"iteration-{iteration}-run-{number}") as run:
            model = HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", **params
            )
            model.fit(train[features], train["target"])

            val_proba = model.predict_proba(val[features])[:, 1]
            threshold = best_threshold(val["target"], val_proba)
            val_f1 = f1_score(val["target"], val_proba >= threshold, zero_division=0)

            test_proba = model.predict_proba(test[features])[:, 1]
            metrics = evaluate(test["target"], test_proba, threshold)

            mlflow.log_params({**params, "iteration": iteration, "split": "chronological",
                               "class_weight": "balanced", "n_features": len(features)})
            mlflow.log_metrics({**metrics, "val_f1": val_f1, "threshold": threshold,
                                "baseline_f1_always_up": baseline_f1})
            mlflow.sklearn.log_model(model, name="model",
                                     input_example=test[features].head(5))

            print(f"Run {number}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if val_f1 > best["val_f1"]:
                best = {"val_f1": val_f1, "run_id": run.info.run_id, "model": model,
                        "threshold": threshold, "metrics": metrics}

    Path("models").mkdir(exist_ok=True)
    out = Path(f"models/best_iteration_{iteration}.joblib")
    joblib.dump({"model": best["model"], "encoder": encoder, "features": features,
                 "threshold": best["threshold"]}, out)

    print(f"\nBest run {best['run_id']} saved to {out}")
    print(f"Test metrics: {best['metrics']} | baseline F1: {baseline_f1:.4f}")

    if register if register is not None else iteration == 2:
        version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME).version
        print(f"Registered {MODEL_NAME} version {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--register", action=argparse.BooleanOptionalAction, default=None,
                        help="Register best model (default: only for iteration 2)")
    args = parser.parse_args()
    main(args.input, args.iteration, args.register)
