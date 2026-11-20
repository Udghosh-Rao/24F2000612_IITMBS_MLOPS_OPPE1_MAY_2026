"""Streamlit demo: upload minute candles -> features -> backtested model -> prediction.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import BASE_FEATURES, EXTRA_FEATURES, RANDOM_STATE
from src.features import add_features, add_target, clean_candles
from src.monitoring import drift_report
from src.sample_data import make_candles
from src.train import best_threshold, chronological_split, evaluate

FEATURES = BASE_FEATURES + EXTRA_FEATURES

st.set_page_config(page_title="Stock Movement Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Movement Predictor")
st.caption(
    "Predicts whether a stock's close will be higher **5 minutes** later. "
    "Upload minute candles (`timestamp, close, volume`) or use sample data."
)

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Minute candles CSV", type="csv")
    use_sample = st.toggle("Use sample data", value=upload is None)
    st.header("Model")
    lr = st.select_slider("Learning rate", [0.02, 0.05, 0.1, 0.2], value=0.05)
    iters = st.slider("Boosting iterations", 50, 400, 150, step=50)
    leaves = st.select_slider("Max leaf nodes", [7, 15, 31, 63], value=31)


@st.cache_data
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    return add_target(add_features(clean_candles(raw)))


if upload is not None and not use_sample:
    raw = pd.read_csv(upload)
else:
    raw = make_candles(5000, seed=7)

try:
    df = prepare(raw)
except ValueError as err:
    st.error(f"Invalid file: {err}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Candles", f"{len(df):,}")
c2.metric("Date range", f"{df.timestamp.min():%d %b %Y} → {df.timestamp.max():%d %b %Y}")
c3.metric("Up-move rate", f"{df['target'].mean():.1%}")

st.subheader("Price & 10-bar rolling average")
st.line_chart(df.set_index("timestamp")[["close", "rolling_avg_10"]].tail(1500))

labeled = df.dropna(subset=["target"]).copy()
labeled["target"] = labeled["target"].astype(int)
if len(labeled) < 500:
    st.warning("Need at least ~500 labeled candles to train a model.")
    st.stop()

train, val, test = chronological_split(labeled)
model = HistGradientBoostingClassifier(
    learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves,
    class_weight="balanced", random_state=RANDOM_STATE,
).fit(train[FEATURES], train["target"])
threshold = best_threshold(val["target"], model.predict_proba(val[FEATURES])[:, 1])
metrics = evaluate(test["target"], model.predict_proba(test[FEATURES])[:, 1], threshold)
baseline = evaluate(test["target"], np.ones(len(test)), 0.5)

st.subheader("Chronological backtest (last 20% held out)")
cols = st.columns(5)
for col, (k, v) in zip(cols, metrics.items(), strict=True):
    col.metric(k.upper().replace("_", "-"), f"{v:.3f}",
               delta=f"{v - baseline[k]:+.3f} vs always-up" if k in ("f1", "accuracy") else None)
st.caption(f"Train {len(train):,} · Validation {len(val):,} · Test {len(test):,} · "
           f"Threshold tuned on validation: {threshold:.2f}")

st.subheader("Latest prediction")
latest = df.iloc[[-1]]
p_up = float(model.predict_proba(latest[FEATURES])[0, 1])
st.metric(f"As of {latest.timestamp.iloc[0]:%Y-%m-%d %H:%M} UTC",
          "⬆️ UP" if p_up >= threshold else "⬇️ NOT UP", f"P(up) = {p_up:.1%}")

st.subheader("Data drift: training window vs. test window (PSI)")
drift = drift_report(train, test, FEATURES)
st.dataframe(
    drift.style.format({"psi": "{:.4f}"}).map(
        lambda s: {"alert": "color: #d62728", "warn": "color: #ff7f0e"}.get(s, ""), subset=["status"]
    ),
    width="stretch", hide_index=True,
)
st.caption("PSI < 0.1 stable · 0.1–0.25 moderate drift · > 0.25 significant drift (retrain)")

with st.expander("Engineered features (last 20 rows)"):
    st.dataframe(df[["timestamp", "close", "volume", *FEATURES]].tail(20), width="stretch")
