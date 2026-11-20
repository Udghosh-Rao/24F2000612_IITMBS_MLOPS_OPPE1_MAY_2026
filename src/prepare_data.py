"""Build the engineered dataset from raw per-stock minute CSVs."""

import argparse
from pathlib import Path

import pandas as pd

from src.features import add_features, add_target, clean_candles


def process_stock(path: Path) -> pd.DataFrame:
    stock_name = path.name.split("__")[0]
    try:
        df = clean_candles(pd.read_csv(path))
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err

    df["stock_name"] = stock_name
    return add_target(add_features(df))


def build_dataset(input_dirs, output_path):
    frames = []
    for input_dir in input_dirs:
        paths = sorted(Path(input_dir).glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")
        for path in paths:
            print(f"Processing {path}")
            frames.append(process_stock(path))

    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["stock_name", "timestamp"])
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    print(f"\nDataset created: {output_path}")
    print(f"Rows: {len(result):,} | Stocks: {result['stock_name'].nunique()}")
    print(f"Valid targets: {result['target'].notna().sum():,}")
    print(f"Positive rate: {result['target'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_dataset(args.input_dirs, args.output)
