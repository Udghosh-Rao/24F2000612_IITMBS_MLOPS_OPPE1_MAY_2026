from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "StockMovementPredictor"
MODEL_FILE = "models/best_iteration_2.joblib"
TEST_FILE = "ci_data/test.parquet"


# ------------------------------------------------------------
# Verify model exists in MLflow Model Registry
# ------------------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

print("Fetching best model metadata from MLflow Model Registry")

client = MlflowClient()

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError(
        f"No registered versions found for {MODEL_NAME}"
    )

latest = max(
    versions,
    key=lambda version: int(version.version),
)

print("Registered model:", MODEL_NAME)
print("Registry version:", latest.version)
print("Registry run ID:", latest.run_id)
print("Registry status:", latest.status)


# ------------------------------------------------------------
# Load reproducible DVC-pinned model bundle
# ------------------------------------------------------------

model_path = Path(MODEL_FILE)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model artifact missing: {model_path}"
    )

print("Loading DVC-pinned model bundle:", model_path)

bundle = joblib.load(model_path)

model = bundle["model"]
encoder = bundle["encoder"]
features = bundle["features"]

print("Model type:", type(model).__name__)
print("Features:", features)


# ------------------------------------------------------------
# Load DVC-pinned CI test data
# ------------------------------------------------------------

df = pd.read_parquet(TEST_FILE)

df = df.dropna(
    subset=[
        "target",
        "rolling_avg_10",
        "volume_sum_10",
        "stock_name",
    ]
).copy()

df["target"] = df["target"].astype(int)


# ------------------------------------------------------------
# Use SAME encoder used during training
# ------------------------------------------------------------

df["stock_encoded"] = encoder.transform(
    df[["stock_name"]]
).ravel()


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

X = df[features]
y = df["target"]

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
f1 = f1_score(y, pred, zero_division=0)

print("Accuracy:", accuracy)
print("F1:", f1)


# ------------------------------------------------------------
# Generate CML artifacts
# ------------------------------------------------------------

Path("reports").mkdir(exist_ok=True)

with open("reports/metrics.md", "w") as f:
    f.write("# Stock Movement Predictor CI Report\n\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- F1 Score: {f1:.4f}\n")
    f.write(f"- Test observations: {len(df):,}\n")
    f.write(f"- MLflow model: {MODEL_NAME}\n")
    f.write(f"- Registry version: {latest.version}\n")
    f.write(f"- MLflow run ID: {latest.run_id}\n")
    f.write("- Model artifact: DVC-pinned iteration 2 model\n")
    f.write("- Test data: DVC-pinned snapshot\n")

ConfusionMatrixDisplay.from_predictions(
    y,
    pred,
)

plt.title(
    "Stock Movement Predictor - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=150,
)

plt.close()

print("CI evaluation successful")
