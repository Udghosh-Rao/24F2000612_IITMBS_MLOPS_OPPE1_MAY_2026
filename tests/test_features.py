import numpy as np
import pandas as pd


RAW = "data/v0/AARTIIND__EQ__NSE__NSE__MINUTE.csv"
ENGINEERED = "feature_repo/data/iteration1.parquet"


def load():
    raw = pd.read_csv(RAW)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = (
        raw.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    eng = pd.read_parquet(ENGINEERED)
    eng = (
        eng[eng["stock_name"] == "AARTIIND"]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return raw, eng


def test_rolling_avg_10_against_raw():
    raw, eng = load()

    expected = raw["close"].iloc[:10].mean()
    actual = eng["rolling_avg_10"].iloc[9]

    assert np.isclose(actual, expected)


def test_volume_sum_10_against_raw():
    raw, eng = load()

    expected = raw["volume"].iloc[:10].sum()
    actual = eng["volume_sum_10"].iloc[9]

    assert np.isclose(actual, expected)
