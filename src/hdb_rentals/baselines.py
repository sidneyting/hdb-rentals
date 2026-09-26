"""The reference runs every other experiment is compared against.

Until this module existed, the code that fit these models lived outside the
repo, so log entries recorded a commit that could not reproduce them. This is
that code, rebuilt from the settings in ``experiments.jsonl``; it reproduces
every logged baseline number exactly (holdout 465.84, the five expanding-window
folds, horizon-matched 506.18 and its bias table).

It doubles as the environment check. LightGBM is deterministic at these
settings, so ``lgbm-raw`` on the holdout must return 465.84 -- anything else
means this environment differs from the one that produced the reference
numbers, and every comparison against them is void.

    python -m hdb_rentals.baselines          # score and print
    python -m hdb_rentals.baselines --log    # also log runs with no clean entry yet
"""

from __future__ import annotations

import argparse
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas.api.types import CategoricalDtype
from sklearn.dummy import DummyRegressor

from . import config
from .data import load_train
from .tracking import log_experiment, read_log
from .validation import (
    expanding_window_splits,
    holdout_split,
    horizon_matched_split,
    horizon_report,
    rmse,
)

RAW_FEATURES = ["FLAT_AGE", "FLAT_MODEL", "FLAT_TYPE", "FLOOR_AREA_SQM", "MONTH_INDEX", "TOWN"]

# Passed to LightGBM as pandas categories. The logged numbers depend on this:
# it is the one setting the log did not record, and the exact reproduction is
# the evidence that this is what the original runs did.
CATEGORICAL = ["FLAT_MODEL", "FLAT_TYPE", "TOWN"]

LGBM_PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63}

# Validation labels as they appear in the log. Scores only compare within one.
HOLDOUT = "holdout-last-4-months"
EXPANDING = "expanding-window-5x3"
HORIZON = "horizon-matched-16m"


@lru_cache(maxsize=1)
def category_levels() -> dict[str, CategoricalDtype]:
    """The level set for each categorical column, fixed once from training data.

    LightGBM never sees the string "maisonette"; it sees the integer code
    pandas assigned it, and ``astype("category")`` assigns codes by sorted
    order of the levels *present in the frame it is called on*. Test is missing
    ``improved maisonette``, which train has, so a test matrix built from its
    own levels shifts 14 of the 20 FLAT_MODEL codes by one: code 7 means
    ``maisonette`` to the model and ``model a`` to the test frame. Nothing
    raises, nothing is null, and most maisonette and model-a rows predict as
    the wrong flat model.

    Taking the levels from train and reusing them everywhere keeps one mapping.
    It also leaves the logged numbers untouched, because the training frame's
    own levels are what produced them.
    """
    train = load_train()
    return {c: CategoricalDtype(sorted(train[c].dropna().unique())) for c in CATEGORICAL}


def model_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select ``features`` and cast the text columns to categories.

    Built once over the whole frame and then sliced, so every fold shares the
    same category codes -- and every *frame* shares them too, train and test
    alike, because the levels come from ``category_levels`` rather than from
    whatever happens to be in ``df``.

    A level absent from that fixed set would be cast to null and predicted as
    missing, so it raises instead. The trade-off is deliberate: a loud failure
    the first time an unseen category appears, rather than a quiet one at
    submission time.
    """
    X = df[features].copy()
    for col, dtype in category_levels().items():
        if col not in X.columns:
            continue
        unseen = sorted(set(X[col].dropna().unique()) - set(dtype.categories))
        if unseen:
            raise ValueError(
                f"{col} has levels the training data never had: {unseen}. "
                f"Decide whether to add them to the fixed set or drop the rows."
            )
        X[col] = X[col].astype(dtype)
    return X


def fit_predict(X, y, trn_idx, val_idx, params=LGBM_PARAMS) -> np.ndarray:
    model = LGBMRegressor(**params, verbose=-1)
    model.fit(X.iloc[trn_idx], y[trn_idx])
    return model.predict(X.iloc[val_idx])


def score_expanding(df, X, y, params=LGBM_PARAMS) -> list[float]:
    """RMSE per expanding-window fold. The headline score is their mean."""
    return [rmse(y[v], fit_predict(X, y, t, v, params)) for t, v in expanding_window_splits(df)]


def score_horizon(df, X, y, params=LGBM_PARAMS) -> tuple[float, pd.DataFrame]:
    """RMSE on the 16-month fold, plus the months-ahead breakdown behind it."""
    trn, val = horizon_matched_split(df)
    pred = fit_predict(X, y, trn, val, params)
    return rmse(y[val], pred), horizon_report(df, y[val], pred, val)


def expanding_run(df, X, y, *, name, features, notes, params=LGBM_PARAMS) -> dict:
    folds = score_expanding(df, X, y, params)
    return {
        "name": name, "score": float(np.mean(folds)), "fold_scores": folds,
        "validation": EXPANDING, "features": features, "model": "LGBMRegressor",
        "params": params, "notes": notes,
    }


def horizon_run(df, X, y, *, name, features, notes, params=LGBM_PARAMS) -> dict:
    """The 16-month run, with the full months-ahead table stored in the entry.

    The +13..+16 bias also goes into ``notes`` in the form the report script
    parses, so older and newer entries read the same way.
    """
    score, report = score_horizon(df, X, y, params)
    bias = report.set_index("horizon").loc["+13..+16", "bias"]
    return {
        "name": name, "score": score, "validation": HORIZON, "features": features,
        "model": "LGBMRegressor", "params": params,
        "notes": f"{notes} Bias +13..+16: {bias}.",
        "extra": {"horizon_report": report.to_dict(orient="records")},
    }


def run_baselines(df: pd.DataFrame | None = None) -> list[dict]:
    """Score the four reference runs. Returns keyword arguments for
    ``log_experiment``, one dict per run, without logging anything."""
    df = load_train() if df is None else df
    y = df[config.TARGET].to_numpy()
    X = model_matrix(df, RAW_FEATURES)
    raw = {"name": "lgbm-raw", "features": RAW_FEATURES}

    trn, val = holdout_split(df)
    median = DummyRegressor(strategy="median").fit(X.iloc[trn], y[trn]).predict(X.iloc[val])
    return [
        {
            "name": "baseline-median", "score": rmse(y[val], median), "validation": HOLDOUT,
            "features": [], "model": "DummyRegressor(median)", "params": {},
            "notes": "Floor to beat. Predicts the training median for every row.",
        },
        {
            **raw, "score": rmse(y[val], fit_predict(X, y, trn, val)), "validation": HOLDOUT,
            "model": "LGBMRegressor", "params": LGBM_PARAMS,
            "notes": "Raw columns only, no auxiliary data. Reference for quick iteration.",
        },
        expanding_run(df, X, y, **raw, notes=(
            "Raw columns only. Score is the mean of the five fold RMSEs; "
            "compare other runs fold by fold, not mean to mean."
        )),
        horizon_run(df, X, y, **raw, notes="Raw columns only, 16 months ahead."),
    ]


def _commit_has(commit: str, path: Path) -> bool:
    rel = path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{rel}"],
        cwd=config.PROJECT_ROOT, capture_output=True,
    ).returncode == 0


def has_clean_entry(run: dict, code: Path) -> bool:
    """Whether the log already holds an entry for this run and scheme that could
    be re-run from its commit: clean, made at a commit containing ``code`` (the
    module that produced it), and carrying every extra field the run would add.

    Clean alone is not enough. Entries logged before the training code was
    committed are clean but point at a tree that cannot reproduce them.
    """
    needed = set(run.get("extra") or {})
    return any(
        r["name"] == run["name"]
        and r.get("validation") == run["validation"]
        and not r.get("dirty")
        and needed <= r.keys()
        and _commit_has(r["commit"], code)
        for r in read_log()
    )


def report_and_log(runs: list[dict], *, log: bool, code: Path = Path(__file__)) -> None:
    """Print each run, and with ``log`` append those not yet logged reproducibly.
    ``code`` is the module that produced the runs."""
    for run in runs:
        folds = run.get("fold_scores")
        folds = f"  folds {[round(s, 1) for s in folds]}" if folds else ""
        print(f"{run['name']:<22} {run['validation']:<22} {run['score']:8.2f}{folds}")
        if log and not has_clean_entry(run, code):
            entry = log_experiment(**run)
            print(f"  logged at {entry['commit']}{' (dirty)' if entry['dirty'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="append runs that have no clean entry in the log yet")
    report_and_log(run_baselines(), log=ap.parse_args().log)


if __name__ == "__main__":
    main()
