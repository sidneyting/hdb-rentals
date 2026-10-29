"""Time-ordered validation.

The test set starts where training ends -- train runs 2021-01 to 2025-03,
test runs 2025-03 to 2026-07 -- so every test prediction is an extrapolation
forward in time. These splitters score that same forward horizon, which makes
the validation number comparable to the one the leaderboard will report.

They are not here to prevent leakage. Letting a model see labelled rows from
the month it is scored on was measured at roughly 1.6 RMSE against a 464
baseline -- a third of a percent, and not worth engineering around. Rent rose
steeply from 2021 to mid-2023 and has been flat since, so knowing the month
carries little information at the recent end where both validation and test
live.

The exposure these splits genuinely cannot measure is extrapolation. A tree
model cannot predict above the largest MONTH_INDEX it trained on; it returns
the value of its last leaf for month 51 and for month 100 alike. Every test
row is past that boundary, so a plain time feature freezes all 50,000
predictions at the 2025-03 level for sixteen months. Validating on held-out
history will not reveal this, because history stops where the extrapolation
begins.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd


def holdout_split(
    df: pd.DataFrame, *, n_valid_months: int = 4, month_col: str = "MONTH_INDEX"
) -> tuple[np.ndarray, np.ndarray]:
    """Train on everything before the last ``n_valid_months``, validate on them.

    Four months is the default because it is the horizon the test set opens
    with; widen it to stress-test further-out predictions.
    """
    months = df[month_col].to_numpy()
    cutoff = np.sort(np.unique(months))[-n_valid_months]
    return np.flatnonzero(months < cutoff), np.flatnonzero(months >= cutoff)


def expanding_window_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    valid_months: int = 3,
    month_col: str = "MONTH_INDEX",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``n_splits`` (train_idx, valid_idx) pairs that walk forward in time.

    Each fold trains on all months up to a cutoff and validates on the next
    ``valid_months``. The training window grows with each fold, mirroring how
    the final model is fit on all history before predicting the future.
    """
    months = df[month_col].to_numpy()
    unique = np.sort(np.unique(months))
    last = len(unique)
    for k in range(n_splits, 0, -1):
        v_end = last - (k - 1) * valid_months
        v_start = v_end - valid_months
        if v_start <= 0:
            raise ValueError(
                f"not enough months ({last}) for {n_splits} folds of {valid_months}"
            )
        v_months = set(unique[v_start:v_end])
        cutoff = unique[v_start]
        yield (
            np.flatnonzero(months < cutoff),
            np.flatnonzero(np.isin(months, list(v_months))),
        )


def horizon_matched_split(
    df: pd.DataFrame, *, horizon_months: int = 16, month_col: str = "MONTH_INDEX"
) -> tuple[np.ndarray, np.ndarray]:
    """One fold whose forecast distance matches the real test set.

    The expanding-window folds ask the model to predict 1-3 months ahead; the
    test set asks for 1-16. That mismatch does not change how configurations
    rank -- measured, full features beat fewer features at both distances --
    so the short folds remain the right tool for iteration.

    What the short folds hide is bias. Trained through month 33, the model
    under-predicts by $0 at three months out and by $75 at sixteen, growing
    monotonically, because it cannot extrapolate past its last training month.
    RMSE actually *falls* over the same range (557 -> 478) because rent is less
    variable in the later months, so watching RMSE alone suggests the long
    horizon is fine when it is not.

    Run this occasionally as a reality check rather than on every iteration,
    and read ``horizon_report`` alongside it.
    """
    months = df[month_col].to_numpy()
    cutoff = months.max() - horizon_months + 1
    if cutoff <= months.min():
        raise ValueError(f"horizon of {horizon_months} months leaves no training data")
    return np.flatnonzero(months < cutoff), np.flatnonzero(months >= cutoff)


def horizon_report(
    df: pd.DataFrame,
    y_true,
    y_pred,
    valid_idx: np.ndarray,
    *,
    buckets: Sequence[tuple[int, int]] = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 16)),
    month_col: str = "MONTH_INDEX",
) -> pd.DataFrame:
    """Break error down by how far ahead each row was predicted.

    Returns RMSE, RMSE normalised by the target's own spread, and mean signed
    bias per bucket. The bias column is the point of this function: a model
    that cannot extrapolate drifts one direction as the horizon grows, and a
    single aggregate RMSE will not show it.
    """
    months = df[month_col].to_numpy()
    last_trained = months[np.setdiff1d(np.arange(len(df)), valid_idx)].max()
    ahead = months[valid_idx] - last_trained
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    rows = []
    for lo, hi in buckets:
        sel = (ahead >= lo) & (ahead <= hi)
        if not sel.any():
            continue
        rows.append(
            {
                "horizon": f"+{lo}..+{hi}",
                "n": int(sel.sum()),
                "target_sd": round(float(t[sel].std()), 1),
                "rmse": round(rmse(t[sel], p[sel]), 1),
                "rmse_over_sd": round(float(rmse(t[sel], p[sel]) / t[sel].std()), 3),
                "bias": round(float((p[sel] - t[sel]).mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
