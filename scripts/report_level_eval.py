"""Every number and the figure in the report's market-level section (4b_level_eval.tex).

    uv run python scripts/report_level_eval.py

Rebuilt from the report's text, because the code behind that section is not in this
repository. Uses Part 2.3's E0 columns and tuned settings (compare_gnn_lightgbm.py).
Column order follows E0, since LightGBM's column sampling depends on it: 487.49 for the
month index needs this order. Writes outputs/report_level/:

    table1.csv           Table I, on the holdout
    macro.csv            COE, share and placebo columns added to the month-index model
    macro_checks.csv     the same over five seeds, and tuned against the older untuned settings
    no_location.csv      the same level change without location features
    rolling.csv          RMSE by forecast origin, each scoring the next 16 months
    rolling_premiums.csv the same on the submitted model's columns, with block premiums rebuilt per origin
    level_backtest.png   Fig. 1, from rolling.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import compare_gnn_lightgbm as cmp  # noqa: E402
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from hdb_rentals.baselines import LGBM_PARAMS  # noqa: E402
from hdb_rentals.macro import load_macro  # noqa: E402
from hdb_rentals.market_level import forecast_level, hedonic_level  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

OUT = ROOT / "outputs" / "report_level"
CAT = ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
FIT_END, HOLD_START, HOLD_END = 35, 36, 50  # 2023-12, 2024-01, 2025-03
ORIGINS = range(11, 35)  # 2021-12 to 2023-11
LEVELS = ["hold", "damped", "average", "straight line", "oracle"]
METHOD = {"hold": "hold", "damped": "damped", "average": "blend", "straight line": "linear"}


def load():
    hdb = cmp.part23_block_table()
    tr = cmp.clean_rent(pd.read_csv(cmp.DATA / "train.csv")).sort_values("MONTH_INDEX", kind="stable").reset_index(drop=True)
    tr = tr.merge(hdb[cmp.KEY + cmp.DIST], on=cmp.KEY, how="left", validate="m:1")
    tr["FLAT_AGE"] = tr["RENT_APPROVAL_DATE"].dt.year - tr["LEASE_COMMENCE_DATE"]
    hedonic = tr[["MONTH_INDEX", cmp.TARGET, *CAT, "FLOOR_AREA_SQM", "FLAT_AGE"]].copy()
    for c in CAT:
        tr[c] = tr[c].astype(pd.CategoricalDtype(sorted(tr[c].unique())))
    macro = load_macro(gate=False)
    tr = tr.join(macro.reindex(tr["MONTH_INDEX"]).reset_index(drop=True))
    stocks = pd.read_csv(cmp.DATA / "auxiliary" / "sg-stock-prices.csv")
    penguin = stocks[stocks["NAME"].str.contains("Penguin")]
    month = pd.to_datetime(penguin["DATE"]).dt.to_period("M")
    placebo = penguin.groupby(month)["ADJUSTED_CLOSE"].mean()
    placebo.index = [(p.year - 2021) * 12 + p.month - 1 for p in placebo.index]
    tr["PLACEBO"] = placebo.reindex(tr["MONTH_INDEX"]).to_numpy()
    return tr, hedonic


TR, HEDONIC = load()
MONTHS = TR["MONTH_INDEX"].to_numpy()
Y = TR[cmp.TARGET].to_numpy(float)
E0_LEVEL = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", *cmp.DIST, *CAT]
E0 = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "MONTH_INDEX", *cmp.DIST, *CAT]
NO_LOCATION = ["FLOOR_AREA_SQM", "FLAT_AGE", *CAT]


def fit_predict(cols, fit, val, params=cmp.LGBM_TUNED, scale=None, seed=0):
    """LightGBM on rent, or on rent / scale and multiplied back when scale is given."""
    s = np.ones(len(Y)) if scale is None else scale
    est = lgb.LGBMRegressor(**params, verbose=-1, n_jobs=-1, random_state=seed)
    return est.fit(TR[cols].iloc[fit], Y[fit] / s[fit]).predict(TR[cols].iloc[val]) * s[val]


def level_scale(level, fit, val):
    """Market level per row of fit and val, from fit months only unless oracle."""
    if level == "oracle":
        log_level = hedonic_level(HEDONIC, np.concatenate([fit, val]))
    else:
        ahead = np.arange(MONTHS[fit].max() + 1, MONTHS[val].max() + 1)
        log_level = forecast_level(hedonic_level(HEDONIC, fit), ahead, METHOD[level])
    scale = np.full(len(Y), np.nan)
    rows = np.concatenate([fit, val])
    scale[rows] = np.exp(log_level.reindex(MONTHS[rows]).to_numpy())
    return scale


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def row(name, pred, val):
    ahead, err = MONTHS[val] - FIT_END, pred - Y[val]
    near, far = ahead <= 3, ahead >= 13
    return {"arm": name, "rmse": rmse(pred, Y[val]), "bias": err.mean(),
            "bias_1_3": err[near].mean(), "bias_13_15": err[far].mean(),
            "rmse_1_3": rmse(pred[near], Y[val][near]), "rmse_13_15": rmse(pred[far], Y[val][far])}


def holdout_tables():
    fit = np.where(MONTHS <= FIT_END)[0]
    val = np.where((MONTHS >= HOLD_START) & (MONTHS <= HOLD_END))[0]

    table = [row("month index", fit_predict(E0, fit, val), val)]
    for level in LEVELS:
        table.append(row(level, fit_predict(E0_LEVEL, fit, val, scale=level_scale(level, fit, val)), val))
    pd.DataFrame(table).round(2).to_csv(OUT / "table1.csv", index=False)

    arms = {"COE_B, COE_E": ["COE_B", "COE_E"], "shares": ["STOCK_BANKS", "STOCK_PROPERTY", "STOCK_BROAD"],
            "placebo": ["PLACEBO"]}
    macro = [row(name, fit_predict(E0 + cols, fit, val), val) for name, cols in arms.items()]
    pd.DataFrame(macro).round(2).to_csv(OUT / "macro.csv", index=False)

    coe, shares = arms["COE_B, COE_E"], arms["shares"]
    checks = []
    for seed in range(5):
        for name, cols in [("month index", E0), ("+ COE", E0 + coe), ("+ shares", E0 + shares), ("+ placebo", E0 + arms["placebo"])]:
            checks.append({"check": "tuned, seed", "seed": seed, "arm": name, "rmse": rmse(fit_predict(cols, fit, val, seed=seed), Y[val])})
    for name, cols in [("month index", E0), ("+ COE", E0 + coe)]:
        checks.append({"check": "untuned settings", "seed": 0, "arm": name, "rmse": rmse(fit_predict(cols, fit, val, LGBM_PARAMS), Y[val])})
    pd.DataFrame(checks).round(2).to_csv(OUT / "macro_checks.csv", index=False)

    average = level_scale("average", fit, val)
    rows = []
    for name, params in [("tuned", cmp.LGBM_TUNED), ("default", LGBM_PARAMS)]:
        rows.append(row(f"{name}, month index", fit_predict([*NO_LOCATION, "MONTH_INDEX"], fit, val, params), val))
        rows.append(row(f"{name}, average", fit_predict(NO_LOCATION, fit, val, params, average), val))
    pd.DataFrame(rows).round(2).to_csv(OUT / "no_location.csv", index=False)


def rolling():
    rows = []
    for origin in ORIGINS:
        fit = np.where(MONTHS <= origin)[0]
        val = np.where((MONTHS > origin) & (MONTHS <= origin + 16))[0]
        r = {"origin": origin, "month index": rmse(fit_predict(E0, fit, val), Y[val])}
        for level in LEVELS:
            r[level] = rmse(fit_predict(E0_LEVEL, fit, val, scale=level_scale(level, fit, val)), Y[val])
        rows.append(r)
        pd.DataFrame(rows).round(2).to_csv(OUT / "rolling.csv", index=False)
        print(r, flush=True)


def block_premiums(fit, val):
    """Part 2.3 E1 for one origin: `oof` premiums for fit rows, all fit rows for scored rows."""
    block = (TR["TOWN"].astype(str) + "|" + TR["BLOCK"].astype(str) + "|" + TR["STREET"].astype(str)).astype("category").cat.codes.to_numpy(dtype=np.int64)
    flat = TR["FLAT_TYPE"].cat.codes.to_numpy(dtype=np.int64)
    nb, nft = block.max() + 1, flat.max() + 1
    group = block * nft + flat
    rel = Y / TR.groupby(["MONTH_INDEX", "FLAT_TYPE"], observed=True)[cmp.TARGET].transform("median").to_numpy()

    def encode(src, dst):
        sb, cb = np.bincount(block[src], rel[src], nb), np.bincount(block[src], minlength=nb)
        sg, cg = np.bincount(group[src], rel[src], nb * nft), np.bincount(group[src], minlength=nb * nft)
        pb = (20 + sb[block[dst]]) / (20 + cb[block[dst]])
        return np.column_stack([pb, (5 * pb + sg[group[dst]]) / (5 + cg[group[dst]]), cg[group[dst]]])

    out = np.full((len(Y), 3), np.nan)
    for train_idx, own_idx in KFold(5, shuffle=True, random_state=0).split(fit):
        out[fit[own_idx]] = encode(fit[train_idx], fit[own_idx])
    out[val] = encode(fit, val)
    return out


def rolling_premiums():
    """Rolling origins on the submitted model's columns. Level arms weight rows by the level squared, as Part 2.3 E6."""
    prem = ["BLOCK_PREM", "BLOCK_FT_PREM", "BLOCK_FT_N"]
    num, time_num = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"], ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "MONTH_INDEX"]
    rows = []
    for origin in ORIGINS:
        fit = np.where(MONTHS <= origin)[0]
        val = np.where((MONTHS > origin) & (MONTHS <= origin + 16))[0]
        frame = TR.copy()
        frame[prem] = block_premiums(fit, val)

        def run(cols, scale=None):
            s = np.ones(len(Y)) if scale is None else scale
            est = lgb.LGBMRegressor(**cmp.LGBM_TUNED, verbose=-1, n_jobs=-1, random_state=0)
            weight = None if scale is None else s[fit] ** 2
            est.fit(frame[cols].iloc[fit], Y[fit] / s[fit], sample_weight=weight)
            return est.predict(frame[cols].iloc[val]) * s[val]

        r = {"origin": origin, "month index": rmse(run([*time_num, *cmp.DIST, *prem, *CAT]), Y[val])}
        for level in LEVELS:
            r[level] = rmse(run([*num, *cmp.DIST, *prem, *CAT], level_scale(level, fit, val)), Y[val])
        rows.append(r)
        pd.DataFrame(rows).round(2).to_csv(OUT / "rolling_premiums.csv", index=False)
        print(r, flush=True)


def figure():
    r = pd.read_csv(OUT / "rolling.csv")
    ink, ink_2, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    styles = [("straight line", "Straight line", axis, 1.6), ("hold", "Hold", "#eb6834", 2),
              ("damped", "Damped trend", "#1baf7a", 2), ("average", "Average of hold and damped", "#2a78d6", 3.2)]
    months = pd.period_range("2021-01", periods=60, freq="M").to_timestamp()
    x = months[r["origin"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    # The seven origins from 2022-07 to 2023-01 whose scored windows span the flattening, as in the text.
    ax.axvspan(months[18] - pd.Timedelta(days=15), months[24] + pd.Timedelta(days=15), color="#eaebe6", zorder=0)
    ax.text(months[18], 335, "windows that span\nthe 2023 flattening", color=ink_2, va="top", fontsize=9.5)
    for key, label, color, width in styles:
        ax.plot(x, (r[key] - r["oracle"]).clip(lower=0), color=color, lw=width, marker="o", ms=3, label=label)
    ax.set_ylim(0, 345)
    ax.set_title("Cost of the level forecast, by starting month", loc="left", fontweight="bold", color=ink)
    ax.set_ylabel("RMSE above the oracle ($)", color=ink_2)
    ax.set_xlabel("Starting month (trained on everything up to here, scored on the next 16 months)", color=ink_2)
    ax.set_xticks(months[12::3][:8], [m.strftime("%Y-%m") for m in months[12::3][:8]])
    ax.tick_params(colors=muted, length=0)
    ax.grid(color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "level_backtest.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--rolling-premiums" in sys.argv:
        rolling_premiums()
    else:
        holdout_tables()
        if "--holdout-only" not in sys.argv:
            rolling()
            figure()
    print("DONE")
