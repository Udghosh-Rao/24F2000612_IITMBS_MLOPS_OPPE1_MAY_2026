from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay


mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("Fetching best model from MLflow Model Registry")

model = mlflow.sklearn.load_model(
    "models:/StockMovementPredictor/1"
)

df = pd.read_parquet("ci_data/test.parquet")

# Encoding used during iteration-2 training.
stocks = [
    "AARTIIND",
    "ABCAPITAL",
    "ABFRL",
    "ADANIENT",
    "ADANIGAS",
]

mapping = {stock: float(i) for i, stock in enumerate(stocks)}

df["stock_encoded"] = df["stock_name"].map(mapping).fillna(-1)

features = [
    "rolling_avg_10",
    "volume_sum_10",
    "stock_encoded",
]

X = df[features]
y = df["target"].astype(int)

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
f1 = f1_score(y, pred, zero_division=0)

Path("reports").mkdir(exist_ok=True)

with open("reports/metrics.md", "w") as f:
    f.write("# Stock Movement Predictor CI Report\n\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- F1 Score: {f1:.4f}\n")
    f.write(f"- Test observations: {len(df):,}\n")
    f.write("- Model: StockMovementPredictor v1\n")
    f.write("- Test data: DVC-pinned snapshot\n")

ConfusionMatrixDisplay.from_predictions(y, pred)
plt.title("Stock Movement Predictor - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

print("Accuracy:", accuracy)
print("F1:", f1)
print("CI evaluation successful")
