"""Generate realistic-looking synthetic minute candles for demos and tests."""

import argparse

import numpy as np
import pandas as pd


def make_candles(n=2000, start="2024-01-01 03:45", seed=0, price=100.0) -> pd.DataFrame:
    """Random-walk minute bars during NSE trading hours (UTC 03:45-10:00)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start=pd.Timestamp(start).normalize(), periods=n // 375 + 2, freq="B")
    stamps = [
        d + pd.Timedelta(hours=3, minutes=45 + m) for d in days for m in range(375)
    ][:n]
    returns = rng.normal(0, 0.001, n) + 0.3 * np.r_[0, rng.normal(0, 0.001, n - 1)]
    close = price * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=8, sigma=0.6, size=n).round()
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(stamps, utc=True), "close": close.round(2), "volume": volume}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--output", default="sample_data/DEMO__EQ__NSE__NSE__MINUTE.csv")
    args = p.parse_args()
    make_candles(args.rows).to_csv(args.output, index=False)
    print("Wrote", args.output)
