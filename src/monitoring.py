"""Data drift monitoring with the Population Stability Index (PSI).

PSI < 0.1: stable | 0.1-0.25: moderate drift | > 0.25: significant drift (retrain).

CLI:  python -m src.monitoring --reference ref.parquet --current cur.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_FEATURES, EXTRA_FEATURES

PSI_WARN, PSI_ALERT = 0.1, 0.25


def psi(reference, current, bins: int = 10) -> float:
    """PSI between two numeric samples using reference quantile bins."""
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    ref_pct = np.histogram(ref, edges)[0] / len(ref) + eps
    cur_pct = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "alert" if value > PSI_ALERT else "warn" if value > PSI_WARN else "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> pd.DataFrame:
    features = features or [f for f in BASE_FEATURES + EXTRA_FEATURES if f in reference and f in current]
    rows = [{"feature": f, "psi": psi(reference[f], current[f])} for f in features]
    report = pd.DataFrame(rows)
    report["status"] = report["psi"].map(status)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--output", default="reports/drift.md")
    args = p.parse_args()

    report = drift_report(pd.read_parquet(args.reference), pd.read_parquet(args.current))
    Path(args.output).parent.mkdir(exist_ok=True)
    rows = "\n".join(f"| {r.feature} | {r.psi:.4f} | {r.status} |" for r in report.itertuples())
    Path(args.output).write_text(f"| Feature | PSI | Status |\n|---|---|---|\n{rows}\n")
    print(report.to_string(index=False))
    if (report["status"] == "alert").any():
        raise SystemExit("Significant drift detected - consider retraining")
