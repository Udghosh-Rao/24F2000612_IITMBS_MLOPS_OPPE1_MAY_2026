from pathlib import Path
import argparse
import pandas as pd


def process_stock(path: Path) -> pd.DataFrame:
    stock_name = path.name.split("__")[0]

    df = pd.read_csv(path)

    required = {"timestamp", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Important: source files must not be assumed chronological.
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    df["stock_name"] = stock_name

    # Last 10 available observations, including current observation.
    df["rolling_avg_10"] = (
        df["close"]
        .rolling(window=10, min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling(window=10, min_periods=1)
        .sum()
    )

    # Ground truth: actual close exactly five minutes later.
    future = df[["timestamp", "close"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(minutes=5)
    future = future.rename(columns={"close": "close_t_plus_5"})

    df = df.merge(future, on="timestamp", how="left")

    # Rows without an observed price exactly t+5 cannot have valid ground truth.
    df["target"] = (
        df["close_t_plus_5"] > df["close"]
    ).astype("Int64")

    df.loc[df["close_t_plus_5"].isna(), "target"] = pd.NA

    return df


def build_dataset(input_dirs, output_path):
    frames = []

    for input_dir in input_dirs:
        paths = sorted(Path(input_dir).glob("*.csv"))

        if not paths:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        for path in paths:
            print(f"Processing {path}")
            frames.append(process_stock(path))

    result = pd.concat(frames, ignore_index=True)

    result = result.sort_values(
        ["stock_name", "timestamp"]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(output_path, index=False)

    print("\nDataset created")
    print(f"Output: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Stocks: {result['stock_name'].nunique()}")
    print(f"Valid targets: {result['target'].notna().sum():,}")
    print(result.groupby("stock_name").size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    build_dataset(args.input_dirs, args.output)
