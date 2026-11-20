"""CI evaluation: verify the registered model and score it on DVC-pinned test data.

Writes reports/metrics.md and reports/confusion_matrix.png for the CML PR comment.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from src.config import MODEL_NAME, MODEL_PATH, TEST_FILE, TRACKING_URI  # noqa: E402
from src.train import evaluate  # noqa: E402


def registry_version():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for {MODEL_NAME}")
    latest = max(versions, key=lambda v: int(v.version))
    print(f"Registry: {MODEL_NAME} v{latest.version} run={latest.run_id} status={latest.status}")
    return latest


def main():
    latest = registry_version()

    bundle = joblib.load(MODEL_PATH)
    model, encoder, features = bundle["model"], bundle["encoder"], bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))
    print(f"Model {type(model).__name__} | features={features} | threshold={threshold}")

    df = pd.read_parquet(TEST_FILE)
    df = df.dropna(subset=["target", "stock_name", "rolling_avg_10", "volume_sum_10"]).copy()
    df["target"] = df["target"].astype(int)
    df["stock_encoded"] = encoder.transform(df[["stock_name"]]).ravel()

    proba = model.predict_proba(df[features])[:, 1]
    metrics = evaluate(df["target"], proba, threshold)
    pred = (proba >= threshold).astype(int)
    print(metrics)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    (out / "metrics.md").write_text(
        f"| Metric | Value |\n|---|---|\n{rows}\n\n"
        f"- Test observations: {len(df):,}\n"
        f"- Registered model: `{MODEL_NAME}` v{latest.version} (run `{latest.run_id}`)\n"
        f"- Decision threshold: {threshold:.2f}\n"
    )

    ConfusionMatrixDisplay.from_predictions(df["target"], pred)
    plt.title("Stock Movement Predictor - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    print("CI evaluation successful")


if __name__ == "__main__":
    main()
