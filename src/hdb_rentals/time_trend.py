"""The time-trend comparison: can a macro series stand in for ``MONTH_INDEX``?

Each arm changes only the time feature; everything else is the ``lgbm-raw``
baseline in baselines.py (arm A). The decision and its reasoning are in the
``notes`` of each logged entry and in the macro.py docstring -- in short, keep
``MONTH_INDEX``: COE maps one level to two rent regimes, and a placebo share
price matches COE's small gain.

Like baselines.py, this was rebuilt after the fact from the feature lists in
``experiments.jsonl``, because the original code was never committed. It
reproduces every logged arm exactly.

    python -m hdb_rentals.time_trend          # score and print
    python -m hdb_rentals.time_trend --log    # also log runs with no clean entry yet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .baselines import RAW_FEATURES, expanding_run, horizon_run, model_matrix, report_and_log
from .data import load_train
from .macro import attach_macro

NO_TIME = [f for f in RAW_FEATURES if f != "MONTH_INDEX"]

# A share price with no plausible link to rent. If it helps as much as a macro
# series does, the comparison is not measuring macro content.
PLACEBO_NAME = "Penguin International Limited"


def _placebo(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the placebo's monthly mean adjusted close as ``PLACEBO``."""
    st = pd.read_csv(config.AUX_CSV["stocks"], parse_dates=["DATE"])
    st = st[st["NAME"] == PLACEBO_NAME]
    monthly = st.groupby(st["DATE"].dt.strftime("%Y-%m"))["ADJUSTED_CLOSE"].mean()
    out = df.assign(PLACEBO=df["RENT_APPROVAL_DATE"].map(monthly))
    assert out["PLACEBO"].notna().all(), "placebo series has a gap"
    return out


def run_arms(df: pd.DataFrame | None = None) -> list[dict]:
    """Score arms B-E on both schemes and the placebo on the 16-month fold.
    Returns keyword arguments for ``log_experiment`` without logging."""
    df = load_train() if df is None else df
    gated, gated_cols = attach_macro(df, gate=True)
    ungated, ungated_cols = attach_macro(df, gate=False)

    arms = [
        ("lgbm-no-time", df, NO_TIME,
         "Arm B: no time feature. The control -- without it 'macro helps' cannot be "
         "told apart from 'removing the cliff helps'."),
        ("lgbm-macro-only", gated, NO_TIME + gated_cols,
         "Arm C: gated COE replaces MONTH_INDEX. Worse than B: COE rose and fell in "
         "2023-24 while rent stayed flat, so one COE level maps to two rent regimes."),
        ("lgbm-macro-plus-time", gated, RAW_FEATURES + gated_cols,
         "Arm D: gated COE alongside MONTH_INDEX. Small RMSE gain, but bias at +13..+16 "
         "moves the wrong way, and the placebo matches it."),
        ("lgbm-macro-ungated", ungated, NO_TIME + ungated_cols,
         "Arm E: all macro series, range gate off, replacing MONTH_INDEX. Beats gated C, "
         "so staying inside the training range was not the binding test."),
    ]

    runs = []
    for name, frame, features, notes in arms:
        X, y = model_matrix(frame, features), frame[config.TARGET].to_numpy()
        runs.append(expanding_run(frame, X, y, name=name, features=features, notes=notes))
        runs.append(horizon_run(frame, X, y, name=name, features=features, notes=notes))

    frame = _placebo(df)
    features = RAW_FEATURES + ["PLACEBO"]
    X, y = model_matrix(frame, features), frame[config.TARGET].to_numpy()
    runs.append(horizon_run(frame, X, y, name="placebo-penguin", features=features, notes=(
        "Falsification check, not a candidate. A boat builder's share price alongside "
        "MONTH_INDEX matches arm D's gain, so that gain is not macro content."
    )))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="append runs that have no clean entry in the log yet")
    report_and_log(run_arms(), log=ap.parse_args().log, code=Path(__file__))


if __name__ == "__main__":
    main()
