"""Rent as a market level times a relative premium, with the level forecast.

``lgbm-raw`` freezes every test prediction at the last training month, because
a tree cannot extrapolate ``MONTH_INDEX``. The EDA on main (section 7) splits
rent into two parts that behave differently:

    rent ~= market level(month) x relative premium(flat, location)

The premium is stable over time, so a tree can learn it with no time feature.
Only the level moves, and it can be forecast on its own as a 51-point monthly
series. Each arm here fits the level on training months only, trains the tree
on rent divided by that level, and multiplies back by a *forecast* level for
the months scored. Arms differ only in how the level is carried forward:

    hold     last training level, repeated -- the control. Same decomposition,
             no trend, so it separates "the split helps" from "the trend helps".
    damped   Holt's trend with damping: the recent slope, fading out.
    linear   a straight line through all training months. Expected to
             overshoot, because it averages the 2021-23 climb into the slope.
    oracle   the level fitted with the scored months' own rents. Not a
             candidate -- it sees the answers -- but it bounds what a perfect
             level forecast would buy.

The level is a hedonic index: OLS of log rent on month dummies plus flat and
location attributes, so it tracks the market rather than which flats happened
to rent that month. It is refit inside every fold.

Acceptance, fixed before running: on the 16-month fold, bias at +13..+16 moves
toward zero from -80.1 without RMSE rising above 506.18; on the expanding
window, compare fold by fold against lgbm-raw. One forecast origin cannot rank
trend methods, so a pass here earns a second look, not a default.

    python -m hdb_rentals.market_level          # score and print
    python -m hdb_rentals.market_level --log    # also log runs with no clean entry yet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config
from .baselines import (
    EXPANDING,
    HORIZON,
    LGBM_PARAMS,
    RAW_FEATURES,
    model_matrix,
    report_and_log,
)
from .data import load_train
from .validation import expanding_window_splits, horizon_matched_split, horizon_report, rmse

PREMIUM_FEATURES = [f for f in RAW_FEATURES if f != "MONTH_INDEX"]

# Attributes the hedonic regression holds fixed while reading off the month effect.
_HEDONIC_CATEGORICAL = ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
_HEDONIC_NUMERIC = ["FLOOR_AREA_SQM", "FLAT_AGE"]

METHODS = ("hold", "damped", "linear", "oracle")


def hedonic_level(df: pd.DataFrame, idx: np.ndarray) -> pd.Series:
    """Log market level per ``MONTH_INDEX``, from the rows in ``idx`` only.

    The month coefficients of log rent on month dummies and flat attributes,
    relative to the first month. Log so that a month effect is a percentage
    move, which is what "the market rose 5%" means for every flat alike.
    """
    rows = df.iloc[idx]
    months = rows["MONTH_INDEX"]
    X = pd.concat(
        [
            pd.get_dummies(months, prefix="m", drop_first=True, dtype=float),
            pd.get_dummies(rows[_HEDONIC_CATEGORICAL], drop_first=True, dtype=float),
            rows[_HEDONIC_NUMERIC].astype(float),
        ],
        axis=1,
    )
    X.insert(0, "const", 1.0)
    coef, *_ = np.linalg.lstsq(X.to_numpy(), np.log(rows[config.TARGET].to_numpy()), rcond=None)
    coef = pd.Series(coef, index=X.columns)

    ordered = np.sort(months.unique())
    effect = [0.0] + [coef[f"m_{m}"] for m in ordered[1:]]
    return pd.Series(effect, index=ordered, name="log_level")


def forecast_level(level: pd.Series, months: np.ndarray, method: str) -> pd.Series:
    """Carry a training-months level forward to ``months`` past its end."""
    ahead = np.sort(np.unique(months[months > level.index.max()]))
    h = np.arange(1, len(ahead) + 1)
    if method == "hold":
        future = np.full(len(ahead), level.iloc[-1])
    elif method in ("damped", "blend"):
        fit = ExponentialSmoothing(
            level.to_numpy(), trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
        future = fit.forecast(len(ahead))
        if method == "blend":
            # Average of hold and damped in log space, as notebook 10 (E6) uses.
            future = (np.full(len(ahead), level.iloc[-1]) + future) / 2
    elif method == "linear":
        slope, intercept = np.polyfit(level.index, level.to_numpy(), 1)
        future = level.iloc[-1] + slope * h
    else:
        raise ValueError(f"no forecast rule for {method!r}")
    return pd.concat([level, pd.Series(future, index=ahead)])


def fit_predict(df, X, y, trn, val, method, params=LGBM_PARAMS) -> np.ndarray:
    """Train the tree on rent / level, predict, and scale by the forecast level.

    The tree's target is a ratio, not a log: predicting in logs and
    exponentiating back returns something closer to a median than a mean,
    which biases an RMSE score low.
    """
    months = df["MONTH_INDEX"].to_numpy()
    if method == "oracle":
        level = hedonic_level(df, np.concatenate([trn, val]))
    else:
        level = forecast_level(hedonic_level(df, trn), months[val], method)
    scale = np.exp(level.reindex(months).to_numpy())

    model = LGBMRegressor(**params, verbose=-1)
    model.fit(X.iloc[trn], y[trn] / scale[trn])
    return model.predict(X.iloc[val]) * scale[val]


_NOTES = {
    "hold": "Control: level held at its last training value. Same decomposition as the "
            "trend arms with no trend, so any gain here is the split, not the forecast.",
    "damped": "Candidate: Holt's damped trend carries the level forward.",
    "linear": "Straight line through every training month. Expected to overshoot, since "
              "the slope averages in the 2021-23 climb.",
    "oracle": "Not a candidate: the level is fitted with the scored months' own rents. "
              "Bounds what a perfect level forecast could buy.",
}


def run_arms(df: pd.DataFrame | None = None) -> list[dict]:
    """Score every level method on both schemes. Returns keyword arguments for
    ``log_experiment`` without logging."""
    df = load_train() if df is None else df
    X = model_matrix(df, PREMIUM_FEATURES)
    y = df[config.TARGET].to_numpy()
    shared = {
        "features": PREMIUM_FEATURES,
        "model": "hedonic level x LGBMRegressor",
    }

    runs = []
    for method in METHODS:
        name = f"level-{method}"
        params = {**LGBM_PARAMS, "level": method}
        notes = f"Tree on rent / hedonic market level. {_NOTES[method]}"

        folds = [
            rmse(y[v], fit_predict(df, X, y, t, v, method)) for t, v in expanding_window_splits(df)
        ]
        runs.append({
            **shared, "name": name, "params": params, "notes": notes,
            "score": float(np.mean(folds)), "fold_scores": folds, "validation": EXPANDING,
        })

        trn, val = horizon_matched_split(df)
        pred = fit_predict(df, X, y, trn, val, method)
        report = horizon_report(df, y[val], pred, val)
        bias = report.set_index("horizon").loc["+13..+16", "bias"]
        runs.append({
            **shared, "name": name, "params": params,
            "notes": f"{notes} Bias +13..+16: {bias}.",
            "score": rmse(y[val], pred), "validation": HORIZON,
            "extra": {"horizon_report": report.to_dict(orient="records")},
        })
    return runs


# Second forecast origin, inside the climb: train 2021-01..2022-06, score the
# 16 months 2022-07..2023-10. The first origin (train to 2023-11) only ever
# forecast a flat market, which is the easy case for "hold". This is the case
# the damped trend is meant to insure against.
CLIMB_LAST_MONTH = 33  # 2023-10
CLIMB = "horizon-16m-origin-2022-06"


def run_climb_origin(df: pd.DataFrame | None = None) -> list[dict]:
    """``lgbm-raw`` and every level method on the 16-month fold ending 2023-10."""
    df = load_train() if df is None else df
    df = df[df["MONTH_INDEX"] <= CLIMB_LAST_MONTH].reset_index(drop=True)
    y = df[config.TARGET].to_numpy()
    trn, val = horizon_matched_split(df)

    def entry(name, features, model, params, pred, notes):
        report = horizon_report(df, y[val], pred, val)
        bias = report.set_index("horizon").loc["+13..+16", "bias"]
        return {
            "name": name, "features": features, "model": model, "params": params,
            "score": rmse(y[val], pred), "validation": CLIMB,
            "notes": f"{notes} Bias +13..+16: {bias}.",
            "extra": {"horizon_report": report.to_dict(orient="records")},
        }

    X_raw = model_matrix(df, RAW_FEATURES)
    model = LGBMRegressor(**LGBM_PARAMS, verbose=-1).fit(X_raw.iloc[trn], y[trn])
    runs = [entry(
        "lgbm-raw", RAW_FEATURES, "LGBMRegressor", LGBM_PARAMS, model.predict(X_raw.iloc[val]),
        "Reference at the climb origin: MONTH_INDEX frozen at 2022-06.",
    )]

    X = model_matrix(df, PREMIUM_FEATURES)
    for method in METHODS:
        runs.append(entry(
            f"level-{method}", PREMIUM_FEATURES, "hedonic level x LGBMRegressor",
            {**LGBM_PARAMS, "level": method}, fit_predict(df, X, y, trn, val, method),
            f"Tree on rent / hedonic market level, climb origin. {_NOTES[method]}",
        ))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="append runs that have no clean entry in the log yet")
    ap.add_argument("--climb-only", action="store_true",
                    help="score only the second forecast origin, inside the climb")
    args = ap.parse_args()
    runs = run_climb_origin() if args.climb_only else run_arms() + run_climb_origin()
    report_and_log(runs, log=args.log, code=Path(__file__))


if __name__ == "__main__":
    main()
