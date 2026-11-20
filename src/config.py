"""Central configuration shared by data prep, training, evaluation and serving."""

import os

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "stock-movement-predictor"
MODEL_NAME = "StockMovementPredictor"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_iteration_2.joblib")
TEST_FILE = "ci_data/test.parquet"

WINDOW = 10
HORIZON_MINUTES = 5
RANDOM_STATE = 42

# Original assignment features (kept for Feast / backwards compatibility).
BASE_FEATURES = ["rolling_avg_10", "volume_sum_10"]

# Scale-free features: comparable across stocks with very different price levels.
EXTRA_FEATURES = [
    "ret_1",
    "ret_5",
    "price_vs_avg_10",
    "volatility_10",
    "volume_ratio_10",
    "minute_of_day",
]

MODEL_FEATURES = BASE_FEATURES + EXTRA_FEATURES + ["stock_encoded"]

HYPERPARAMS = [
    {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15},
    {"learning_rate": 0.10, "max_iter": 150, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 1.0},
]
