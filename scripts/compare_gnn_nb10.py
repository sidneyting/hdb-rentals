"""GNN vs notebook 10 on notebook 10's own splits, rows and (optionally) block features.

    uv run python scripts/compare_gnn_nb10.py            # full run
    uv run python scripts/compare_gnn_nb10.py --smoke    # 3 epochs, a quick execution check

Windows are notebook 10's (cells [4] and [6]):
    tuning   fit 2021-01..2023-03, score 2023-04..2023-12 -> GNN early stopping (epoch count)
    holdout  fit 2021-01..2023-12, score 2024-01..2025-03 -> the reported RMSE
The GNN is refit on the holdout fit rows for the epoch count chosen on the tuning split,
as experiment_weighted_gnns.py does. Notebook 10's E0 LightGBM is re-run here first; it must
reproduce the notebook's 603.6 / 487.5, which checks that both sides score the same rows.

Arms (one seed each, as notebook 10 reports one LightGBM run):
    level     how rent / market level is carried past the last fit month: damped (the GNN
              scripts' default) or blend (notebook 10's E6 choice)
    features  gnn = the GNN's block table as-is; nb10 = plus notebook 10's block columns the
              GNN lacks (hawker, CBD distance, MRT interchange)
    model     GNN_uniform, and MLP_matched (same network with the graph switched off)
Notebook 10's block / street premiums are not added; compare with E0, not E5/S1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

# Import torch (via the GNN module) before LightGBM, as experiment_weighted_gnns.py notes.
import experiment_weighted_gnns as ewg  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hdb_rentals.rent_adjustment import MarketRentAdjustment  # noqa: E402

OUT = ROOT / "outputs" / "gnn_vs_nb10"
DATA, CLEANED = ROOT / "data", ROOT / "cleaned"
TARGET, KEY = "MONTHLY_RENT", ["TOWN", "BLOCK", "STREET"]
TUNE_START, HOLD_START, HOLD_END = pd.Timestamp("2023-04-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-04-01")

# ---------------------------------------------------------------- notebook 10, cell [3]
# Copied, not imported, so these numbers are built exactly as the notebook builds them.


def lower_strip(df):
    out = df.copy()
    text = [c for c in out.columns if pd.api.types.is_string_dtype(out[c])]
    out[text] = out[text].apply(lambda s: s.str.strip().str.lower())
    return out


def clean_rent(df):
    out = lower_strip(df).drop(columns=["FURNISHED", "FEE"])
    out["FLAT_TYPE"] = out["FLAT_TYPE"].str.replace(r"[\s\-]+", "-", regex=True)
    d = pd.to_datetime(out["RENT_APPROVAL_DATE"], format="%Y-%m")
    out["RENT_APPROVAL_DATE"] = d
    out["MONTH_INDEX"] = (d.dt.year - 2021) * 12 + d.dt.month - 1
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def nb10_block_table():
    hdb = pd.read_csv(CLEANED / "sg_hdb_block_cleaned.csv", dtype={"POSTAL_CODE": str})
    mrt = pd.read_csv(CLEANED / "sg_mrt_stations_cleaned.csv")
    schools = pd.read_csv(CLEANED / "sg_schools_cleaned.csv")
    malls = lower_strip(pd.read_csv(DATA / "auxiliary" / "sg-shopping-malls.csv"))
    with open(DATA / "auxiliary" / "HawkerCentresGEOJSON.geojson") as f:
        geo = json.load(f)["features"]
    hawker = lower_strip(pd.DataFrame({
        "NAME": [g["properties"]["NAME"] for g in geo],
        "STATUS": [g["properties"]["STATUS"] for g in geo],
        "LONGITUDE": [g["geometry"]["coordinates"][0] for g in geo],
        "LATITUDE": [g["geometry"]["coordinates"][1] for g in geo],
    }))
    mrt_open = mrt[mrt["STATUS"].eq("open")].drop_duplicates("NAME")
    poi_groups = {
        "primary": schools[schools["MAINLEVEL_CODE"].str.contains("primary", na=False)],
        "secondary": schools[schools["MAINLEVEL_CODE"].str.contains("secondary", na=False)],
        "mrt": mrt_open,
        "mrt_interchange": mrt_open[mrt_open["IS_INTERCHANGE"]],
        "malls": malls,
        "hawker": hawker[hawker["STATUS"].ne("under construction")],
    }
    lat, lon = hdb["LATITUDE"].to_numpy(), hdb["LONGITUDE"].to_numpy()
    for prefix, poi in poi_groups.items():
        d = haversine_km(lat[:, None], lon[:, None], poi["LATITUDE"].to_numpy()[None, :], poi["LONGITUDE"].to_numpy()[None, :])
        hdb[f"{prefix}_nearest_km"] = d.min(axis=1)
        for label, r in {"1km": 1.0, "2km": 2.0}.items():
            hdb[f"{prefix}_within_{label}"] = (d <= r).sum(axis=1)
    cbd = mrt.loc[mrt["NAME"].eq("raffles place mrt station"), ["LATITUDE", "LONGITUDE"]].iloc[0]
    hdb["dist_to_cbd_km"] = haversine_km(lat, lon, cbd["LATITUDE"], cbd["LONGITUDE"])
    return hdb


DIST = [
    "MAX_FLOOR", "primary_within_2km", "mrt_within_2km",
    "primary_nearest_km", "mrt_nearest_km", "secondary_within_1km",
    "mrt_interchange_within_1km", "mrt_interchange_nearest_km",
    "malls_within_1km", "malls_nearest_km", "hawker_within_1km", "hawker_nearest_km", "dist_to_cbd_km",
]
# The DIST columns the GNN's block table has no counterpart for.
NB10_EXTRA = ["hawker_nearest_km", "hawker_within_1km", "dist_to_cbd_km",
              "mrt_interchange_nearest_km", "mrt_interchange_within_1km"]

# ---------------------------------------------------------------- notebook 10, cells [4] and [6]: E0
LGBM_TUNED = {"num_leaves": 63, "min_child_samples": 100, "subsample": 0.85, "subsample_freq": 1,
              "colsample_bytree": 0.5, "reg_lambda": 20.0, "cat_smooth": 30.0, "n_estimators": 357,
              "learning_rate": 0.05}


def nb10_e0(hdb):
    tr = clean_rent(pd.read_csv(DATA / "train.csv")).sort_values("MONTH_INDEX", kind="stable").reset_index(drop=True)
    tr = tr.merge(hdb[KEY + DIST + ["LATITUDE", "LONGITUDE"]], on=KEY, how="left", validate="m:1")
    num, cat = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "MONTH_INDEX"] + DIST, ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
    categories = {c: pd.CategoricalDtype(sorted(tr[c].unique())) for c in cat}

    def matrix(df):
        X = df[num + cat].copy()
        for c in cat:
            X[c] = X[c].astype(categories[c])
        return X

    rows = []
    for split, start, end in [("tuning", TUNE_START, HOLD_START), ("holdout", HOLD_START, HOLD_END)]:
        d = tr["RENT_APPROVAL_DATE"]
        fit_df, val_df = tr[d < start], tr[(d >= start) & (d < end)]
        est = lgb.LGBMRegressor(**LGBM_TUNED, verbose=-1, n_jobs=-1, random_state=0)
        p = est.fit(matrix(fit_df), fit_df[TARGET].to_numpy()).predict(matrix(val_df))
        rows.append(dict(split=split, model="nb10_E0_LightGBM", level="MONTH_INDEX", features="nb10",
                         n_rows=len(val_df), **ewg.metrics(val_df[TARGET], p)))
    return rows

# ---------------------------------------------------------------- GNN side


def prepare(fit, others, level, block_num):
    # experiment_weighted_gnns.prepare with the level method and block columns as arguments.
    adjustment = MarketRentAdjustment(level).fit(fit)
    bp = ewg.preprocessor(block_num, ewg.BLOCK_CAT + ["SUBZONE"])
    bp.fit(ewg.blocks.iloc[np.sort(fit.block_id.unique())])
    node_x = torch.tensor(bp.transform(ewg.blocks).astype("float32"))
    rp = ewg.preprocessor(["FLOOR_AREA_SQM", "FLAT_AGE"], ["FLAT_TYPE", "FLAT_MODEL"])
    rp.fit(fit)
    fit_scale = adjustment.multipliers(fit.MONTH_INDEX)
    adjusted = fit.MONTHLY_RENT.to_numpy() / fit_scale
    mean, std = float(adjusted.mean()), float(adjusted.std(ddof=1))
    normalizer = np.mean(fit_scale ** 2)

    def pack(frame):
        scale = adjustment.multipliers(frame.MONTH_INDEX)
        y = frame.MONTHLY_RENT.to_numpy(float)
        return dict(frame=frame, ids=torch.tensor(frame.block_id.to_numpy(int)),
                    record=torch.tensor(rp.transform(frame).astype("float32")),
                    y=y, scale=scale, scaled_y=torch.tensor(((y / scale - mean) / std).astype("float32")),
                    weight=torch.tensor((scale ** 2 / normalizer).astype("float32")))
    return dict(nodes=node_x, parts=[pack(f) for f in (fit, *others)], mean=mean, std=std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 epochs, patience 2, small row samples")
    ap.add_argument("--levels", default="damped,blend")
    ap.add_argument("--features", default="gnn,nb10")
    ap.add_argument("--models", default="MLP_matched,GNN_uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    max_epochs = 3 if args.smoke else args.max_epochs
    ewg.PATIENCE = 2 if args.smoke else 30
    torch.set_num_threads(args.threads)
    out = OUT / "smoke" if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)

    # Notebook 10's block columns onto the GNN's block table, by town + block + street.
    hdb = nb10_block_table()
    blocks = ewg.blocks
    extra = lower_strip(hdb[KEY + NB10_EXTRA])
    merged = blocks[KEY].merge(extra, on=KEY, how="left", validate="m:1")
    assert len(merged) == len(blocks)
    for c in NB10_EXTRA:
        blocks[c] = merged[c].to_numpy(float)
    rented = blocks.block_id.isin(ewg.rent.block_id)
    print(f"nb10 columns matched for {blocks.loc[rented, NB10_EXTRA[0]].notna().mean():.1%} of rented blocks, "
          f"{blocks[NB10_EXTRA[0]].notna().mean():.1%} of all {len(blocks):,} graph blocks", flush=True)
    feature_sets = {"gnn": list(ewg.BLOCK_NUM), "nb10": list(ewg.BLOCK_NUM) + NB10_EXTRA}

    results = nb10_e0(hdb)
    for r in results:
        print("REFERENCE", r, flush=True)

    rent = ewg.rent
    d = rent.RENT_APPROVAL_DATE
    inner_fit, inner_val = rent[d < TUNE_START], rent[(d >= TUNE_START) & (d < HOLD_START)]
    fit, val = rent[d < HOLD_START], rent[(d >= HOLD_START) & (d < HOLD_END)]
    if args.smoke:
        inner_fit, fit = inner_fit.sample(2500, random_state=0).sort_index(), fit.sample(3500, random_state=0).sort_index()
        inner_val, val = inner_val.sample(800, random_state=0).sort_index(), val.sample(800, random_state=0).sort_index()
    print(f"GNN rows: tuning fit {len(inner_fit):,} score {len(inner_val):,}; "
          f"holdout fit {len(fit):,} score {len(val):,}", flush=True)

    config = dict(vars(args), max_epochs=max_epochs, patience=ewg.PATIENCE, hidden=ewg.HIDDEN,
                  tuning=[str(TUNE_START.date()), str(HOLD_START.date())],
                  holdout=[str(HOLD_START.date()), str(HOLD_END.date())], nb10_extra=NB10_EXTRA,
                  torch_version=torch.__version__)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    predictions = []
    for level in args.levels.split(","):
        for feat in args.features.split(","):
            inner = prepare(inner_fit, [inner_val], level, feature_sets[feat])
            outer = prepare(fit, [val], level, feature_sets[feat])
            for kind in args.models.split(","):
                tags = dict(model=kind, level=level, features=feat, seed=args.seed)
                start = time.monotonic()
                tuned, epochs, history = ewg.train(kind, inner, args.seed, max_epochs, tune=True)
                tune_rmse = min(h["inner_rmse"] for h in history)
                fitted, _, _ = ewg.train(kind, outer, args.seed, epochs)
                pred = ewg.predict(fitted, outer, outer["parts"][1])
                results.append(dict(split="tuning", **tags, epochs=epochs, n_rows=len(inner_val), rmse=tune_rmse))
                results.append(dict(split="holdout", **tags, epochs=epochs, n_rows=len(val),
                                    **ewg.metrics(val.MONTHLY_RENT, pred)))
                frame = val[["row_id", "block_id", "RENT_APPROVAL_DATE", "MONTHLY_RENT"]].copy()
                frame["prediction"] = pred
                predictions.append(frame.assign(**tags))
                print("RESULT", tags, "epochs", epochs, "tuning", round(tune_rmse, 2),
                      "holdout", ewg.metrics(val.MONTHLY_RENT, pred),
                      "minutes", round((time.monotonic() - start) / 60, 1), flush=True)
                pd.DataFrame(results).to_csv(out / "results.csv", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    r = pd.DataFrame(results)
    arms = r[["model", "level", "features"]].drop_duplicates()
    table = r.set_index(["model", "level", "features", "split"])["rmse"].unstack("split")
    table = table.reindex(pd.MultiIndex.from_frame(arms))[["tuning", "holdout"]]
    table["holdout vs E0"] = table["holdout"] - table.loc[("nb10_E0_LightGBM", "MONTH_INDEX", "nb10"), "holdout"]
    print("\nRMSE (SGD), notebook 10 windows\n", table.round(2).to_string(), flush=True)
    table.round(2).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
