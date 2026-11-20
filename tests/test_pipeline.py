"""Unit tests for feature engineering, labelling and training helpers (no external data)."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_features, add_target, clean_candles
from src.sample_data import make_candles
from src.train import best_threshold, chronological_split, evaluate


@pytest.fixture
def candles():
    return make_candles(600, seed=1)


def test_clean_candles_sorts_and_dedupes(candles):
    shuffled = pd.concat([candles.sample(frac=1, random_state=0), candles.head(3)])
    out = clean_candles(shuffled)
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].is_unique
    assert len(out) == len(candles)


def test_clean_candles_rejects_missing_columns(candles):
    with pytest.raises(ValueError, match="volume"):
        clean_candles(candles.drop(columns="volume"))


def test_rolling_features_match_manual(candles):
    df = add_features(clean_candles(candles))
    assert np.isclose(df["rolling_avg_10"].iloc[9], candles["close"].iloc[:10].mean())
    assert np.isclose(df["volume_sum_10"].iloc[9], candles["volume"].iloc[:10].sum())
    assert np.isclose(df["ret_5"].iloc[20], candles["close"].iloc[20] / candles["close"].iloc[15] - 1)


def test_features_are_causal(candles):
    """Changing future prices must not change features of earlier rows."""
    base = add_features(clean_candles(candles))
    tampered = candles.copy()
    tampered.loc[300:, "close"] *= 10
    changed = add_features(clean_candles(tampered))
    cols = ["rolling_avg_10", "ret_1", "ret_5", "price_vs_avg_10", "volatility_10"]
    pd.testing.assert_frame_equal(base.loc[:299, cols], changed.loc[:299, cols])


def test_target_uses_exact_t_plus_5():
    ts = pd.date_range("2024-01-01 04:00", periods=8, freq="min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": [10, 11, 12, 13, 14, 9, 20, 5], "volume": 1})
    df = df.drop(index=6).reset_index(drop=True)  # gap: no bar at 04:06
    out = add_target(add_features(df))
    assert out.loc[0, "target"] == 0   # 10 -> 9 at 04:05
    assert pd.isna(out.loc[1, "target"])  # 04:06 missing -> unlabeled
    assert out.loc[2, "target"] == 0   # 12 -> 5 at 04:07
    assert out["target"].iloc[-3:].isna().all()


def test_chronological_split_has_no_overlap(candles):
    df = add_target(add_features(clean_candles(candles))).dropna(subset=["target"])
    train, val, test = chronological_split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4])
    t = best_threshold(y, proba)
    assert 0.2 < t <= 0.3
    m = evaluate(y, proba, t)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0


def test_build_dataset_end_to_end(tmp_path):
    from src.prepare_data import build_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    make_candles(400, seed=4).to_csv(raw / "AAA__EQ__NSE__NSE__MINUTE.csv", index=False)
    make_candles(400, seed=5, price=50).to_csv(raw / "BBB__EQ__NSE__NSE__MINUTE.csv", index=False)

    out = tmp_path / "engineered.parquet"
    build_dataset([raw], out)
    df = pd.read_parquet(out)
    assert set(df["stock_name"]) == {"AAA", "BBB"}
    assert df["target"].notna().sum() > 700
