"""Feature engineering shared by training and serving (single source of truth).

All features are causal: each row only uses its own and earlier observations,
so there is no look-ahead leakage.
"""

import pandas as pd

from src.config import HORIZON_MINUTES, WINDOW


def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, parse timestamps, sort chronologically and drop duplicate bars."""
    missing = {"timestamp", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and scale-free features to a single stock's sorted candles."""
    df = df.copy()
    close, volume = df["close"], df["volume"]

    df["rolling_avg_10"] = close.rolling(WINDOW, min_periods=1).mean()
    df["volume_sum_10"] = volume.rolling(WINDOW, min_periods=1).sum()

    ret_1 = close.pct_change(1)
    df["ret_1"] = ret_1
    df["ret_5"] = close.pct_change(5)
    df["price_vs_avg_10"] = close / df["rolling_avg_10"] - 1
    df["volatility_10"] = ret_1.rolling(WINDOW, min_periods=2).std()
    df["volume_ratio_10"] = volume / volume.rolling(WINDOW, min_periods=1).mean()
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = 1 if close exactly HORIZON_MINUTES later is higher, else 0.

    Rows without an observed bar at t+horizon get a missing target.
    """
    future = df[["timestamp", "close"]].copy()
    future["timestamp"] -= pd.Timedelta(minutes=HORIZON_MINUTES)
    future = future.rename(columns={"close": "close_t_plus_5"})

    df = df.merge(future, on="timestamp", how="left")
    df["target"] = (df["close_t_plus_5"] > df["close"]).astype("Int64")
    df.loc[df["close_t_plus_5"].isna(), "target"] = pd.NA
    return df
