import argparse
import gc
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from src.config import BASE_FEATURES, EXTRA_FEATURES


def main(input_path, output_path, chunk_size=10000):
    df = pd.read_parquet(input_path)

    entity_df = (
        df.loc[
            df["target"].notna(),
            ["stock_name", "timestamp", "target"]
        ]
        .sort_values(["timestamp", "stock_name"])
        .reset_index(drop=True)
    )

    entity_df["target"] = entity_df["target"].astype(int)

    print("Rows for Feast:", len(entity_df))
    print("Stocks:", sorted(entity_df.stock_name.unique()))

    store = FeatureStore(repo_path="feature_repo")
    available = set(df.columns)
    feature_refs = [
        f"stock_features:{name}"
        for name in BASE_FEATURES + EXTRA_FEATURES
        if name in available
    ]
    print("Features:", feature_refs)

    output_path = Path(output_path)
    parts_dir = output_path.parent / (output_path.stem + "_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_files = []

    for start in range(0, len(entity_df), chunk_size):
        end = min(start + chunk_size, len(entity_df))
        print(f"Retrieving {start:,} -> {end:,}", flush=True)

        chunk = entity_df.iloc[start:end].copy()

        result = store.get_historical_features(
            entity_df=chunk,
            features=feature_refs,
        ).to_df()

        part = parts_dir / f"part_{start:09d}.parquet"
        result.to_parquet(part, index=False)
        part_files.append(part)

        del chunk, result
        gc.collect()

    print("Combining chunks...")

    result = pd.concat(
        [pd.read_parquet(p) for p in part_files],
        ignore_index=True
    )

    result = result.sort_values(
        ["timestamp", "stock_name"]
    ).reset_index(drop=True)

    result.to_parquet(output_path, index=False)

    print("SUCCESS:", output_path)
    print("Rows:", len(result))
    print("Stocks:", sorted(result.stock_name.unique()))
    print(result.isna().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    main(args.input, args.output, args.chunk_size)
