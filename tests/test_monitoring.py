import numpy as np
import pandas as pd

from src.monitoring import drift_report, psi


def test_psi_identical_distribution_is_near_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-3


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(2, 1, size=5000)) > 0.25


def test_drift_report_flags_shifted_feature():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"ret_1": rng.normal(size=2000), "volume_ratio_10": rng.normal(size=2000)})
    cur = ref.copy()
    cur["volume_ratio_10"] += 3
    report = drift_report(ref, cur).set_index("feature")
    assert report.loc["volume_ratio_10", "status"] == "alert"
    assert report.loc["ret_1", "status"] == "ok"
