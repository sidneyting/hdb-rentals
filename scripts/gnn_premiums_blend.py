"""GNN with notebook 10's block premiums, and GNN + LightGBM blends, on notebook 10's windows.

    uv run python scripts/gnn_premiums_blend.py            # full run
    uv run python scripts/gnn_premiums_blend.py --smoke    # 3 epochs, a quick execution check
    uv run python scripts/gnn_premiums_blend.py --level blend --same-level   # both models on one level

Same windows, rows, level (damped) and block columns (the GNN's own) as compare_gnn_nb10.py.
--level sets the neural model's market level (damped or blend). By default LightGBM keeps
`MONTH_INDEX`, as in notebook 10's E0/E1; --same-level makes it learn rent / the same level instead
(notebook 10's E6 setup: `MONTH_INDEX` dropped, rows weighted by level squared), so the two models
differ only in the premium model. Non-default runs write to a subfolder named after the options.

Premium inputs (notebook 10 E1: block and block + flat type premiums of relative rent, smoothing
20 / 5, plus the block + flat type row count; built from each split's fit rows only):
    none      no premiums (compare_gnn_nb10.py's arm, re-run for its tuning predictions)
    row       each row gets E1's premiums: out-of-fold for fit rows, all fit rows for scored rows,
              exactly the columns notebook 10's LightGBM gets
    row+node  as row, and each block node also carries its block premium and log row count, so
              the graph can pass premiums between neighbouring blocks. Node premiums use all fit
              rows (a fit row's own rent is in its block's node), so this arm may overfit; scored
              rows' rents never enter either way, and the tuning split decides whether it is kept.

Blends: prediction = w * neural + (1 - w) * LightGBM, with w chosen on the tuning split (grid 0.05)
and applied unchanged to the holdout. The partner is E0 for `none` and E1 for the premium arms.
The neural model's tuning predictions come from a refit on the tuning fit rows for the chosen
epoch count, the same model the early stopping picked.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import compare_gnn_nb10 as cmp  # noqa: E402  (imports torch before LightGBM)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ewg = cmp.ewg
OUT = ROOT / "outputs" / "gnn_premiums_blend"
TARGET, KEY = cmp.TARGET, cmp.KEY
PREM = ["BLOCK_PREM", "BLOCK_FT_PREM", "BLOCK_FT_N"]
NODE_PREM = ["NODE_BLOCK_PREM", "NODE_LOG_N"]
M_BLOCK, M_FT = 20, 5
SPLITS = {"tuning": (cmp.TUNE_START, cmp.HOLD_START), "holdout": (cmp.HOLD_START, cmp.HOLD_END)}

# ---------------------------------------------------------------- notebook 10, cells [4] and [12]


def nb10_premium_frames(hdb):
    """Notebook 10's frames per split with E1 `oof` premiums, plus block-level fit premiums."""
    raw = cmp.clean_rent(pd.read_csv(cmp.DATA / "train.csv"))
    raw["row_id"] = np.arange(len(raw))  # train.csv order, as the GNN's rent frame
    tr = raw.sort_values("MONTH_INDEX", kind="stable").reset_index(drop=True)
    te = cmp.clean_rent(pd.read_csv(cmp.DATA / "test.csv"))
    tr = tr.merge(hdb[KEY + cmp.DIST], on=KEY, how="left", validate="m:1")
    blocks = pd.concat([tr, te])[KEY].drop_duplicates().reset_index(drop=True)
    blocks["BID"] = np.arange(len(blocks))
    ft_levels = sorted(tr["FLAT_TYPE"].unique())
    nb, nft = len(blocks), len(ft_levels)
    tr = tr.merge(blocks, on=KEY, how="left", validate="m:1")
    tr["FT"] = pd.Categorical(tr["FLAT_TYPE"], categories=ft_levels).codes
    tr["G"] = tr["BID"] * nft + tr["FT"]
    tr["REL"] = tr[TARGET] / tr.groupby(["MONTH_INDEX", "FLAT_TYPE"])[TARGET].transform("median")

    def encode(src, dst):
        bid, g, rel = src["BID"].to_numpy(), src["G"].to_numpy(), src["REL"].to_numpy()
        w = np.ones(len(src))
        sb, cb = np.bincount(bid, w * rel, nb), np.bincount(bid, w, nb)
        sg, cg = np.bincount(g, w * rel, nb * nft), np.bincount(g, w, nb * nft)
        n = np.bincount(g, minlength=nb * nft)
        db, dg = dst["BID"].to_numpy(), dst["G"].to_numpy()
        pb = (M_BLOCK + sb[db]) / (M_BLOCK + cb[db])
        pg = (M_FT * pb + sg[dg]) / (M_FT + cg[dg])
        return np.column_stack([pb, pg, n[dg]])

    frames = {}
    for split, (start, end) in SPLITS.items():
        d = tr["RENT_APPROVAL_DATE"]
        fit_df = tr[d < start].reset_index(drop=True)
        val_df = tr[(d >= start) & (d < end)].reset_index(drop=True)
        own = np.empty((len(fit_df), 3))
        for f_idx, p_idx in KFold(5, shuffle=True, random_state=0).split(fit_df):
            own[p_idx] = encode(fit_df.iloc[f_idx], fit_df.iloc[p_idx])
        fit_df[PREM], val_df[PREM] = own, encode(fit_df, val_df)
        sb = np.bincount(fit_df["BID"], fit_df["REL"], nb)
        cb = np.bincount(fit_df["BID"], minlength=nb).astype(float)
        node = blocks[KEY].assign(NODE_BLOCK_PREM=(M_BLOCK + sb) / (M_BLOCK + cb), NODE_LOG_N=np.log1p(cb))
        frames[split] = (fit_df, val_df, node)
    return tr, frames


def nb10_lgbm(tr, frames, level=None):
    """E0 and E1 (E0 + `oof` premiums) LightGBM, predictions keyed by row_id.

    level=None keeps `MONTH_INDEX` (notebook 10's E0/E1). A level method instead fits rent / level
    without `MONTH_INDEX`, weighted by level squared (notebook 10's E6)."""
    time_col = [] if level else ["MONTH_INDEX"]
    num, cat = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"] + time_col + cmp.DIST, ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
    categories = {c: pd.CategoricalDtype(sorted(tr[c].unique())) for c in cat}

    def matrix(df, extra):
        X = df[num + list(extra) + cat].copy()
        for c in cat:
            X[c] = X[c].astype(categories[c])
        return X

    preds, rows = {}, []
    for name, extra in [("nb10_E0_LightGBM", []), ("nb10_E1_LightGBM", PREM)]:
        for split, (fit_df, val_df, _) in frames.items():
            est = lgb.LGBMRegressor(**cmp.LGBM_TUNED, verbose=-1, n_jobs=-1, random_state=0)
            if level:
                # Fitted on the GNN's rent frame (same rows, has FLAT_AGE), as cmp.prepare does.
                adj = cmp.MarketRentAdjustment(level).fit(ewg.rent[ewg.rent.row_id.isin(fit_df.row_id)])
                fs, vs = adj.multipliers(fit_df["MONTH_INDEX"]), adj.multipliers(val_df["MONTH_INDEX"])
                est.fit(matrix(fit_df, extra), fit_df[TARGET].to_numpy() / fs, sample_weight=fs ** 2)
                p = est.predict(matrix(val_df, extra)) * vs
            else:
                p = est.fit(matrix(fit_df, extra), fit_df[TARGET].to_numpy()).predict(matrix(val_df, extra))
            preds[(name, split)] = pd.Series(p, index=val_df["row_id"].to_numpy())
            rows.append(dict(split=split, model=name, premiums="row" if extra else "none",
                             n_rows=len(val_df), **ewg.metrics(val_df[TARGET], p)))
    return preds, rows

# ---------------------------------------------------------------- GNN side


def with_premiums(frame, prem_frame):
    out = frame.merge(prem_frame[["row_id"] + PREM], on="row_id", how="left", validate="1:1")
    assert out[PREM].notna().all().all() and out["row_id"].tolist() == frame["row_id"].tolist()
    out["BLOCK_FT_N"] = np.log1p(out["BLOCK_FT_N"])  # counts on a log scale for the network
    return out


def best_blend(y_tune, nn_tune, gb_tune):
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    scores = [ewg.metrics(y_tune, w * nn_tune + (1 - w) * gb_tune)["rmse"] for w in grid]
    return float(grid[int(np.argmin(scores))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--premiums", default="none,row,row+node")
    ap.add_argument("--models", default="MLP_matched,GNN_uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--level", default="damped", choices=["damped", "blend"])
    ap.add_argument("--same-level", action="store_true", help="LightGBM uses the neural model's level")
    args = ap.parse_args()
    max_epochs = 3 if args.smoke else args.max_epochs
    ewg.PATIENCE = 2 if args.smoke else 30
    torch.set_num_threads(args.threads)
    out = OUT / "smoke" if args.smoke else OUT
    if args.level != "damped" or args.same_level or args.seed != 0:
        out = out / f"level-{args.level}{'_same' if args.same_level else ''}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    hdb = cmp.nb10_block_table()
    tr, frames = nb10_premium_frames(hdb)
    gb_preds, results = nb10_lgbm(tr, frames, args.level if args.same_level else None)
    for r in results:
        print("REFERENCE", r, flush=True)

    rent, blocks = ewg.rent, ewg.blocks
    d = rent.RENT_APPROVAL_DATE
    parts = {"tuning": (rent[d < cmp.TUNE_START], rent[(d >= cmp.TUNE_START) & (d < cmp.HOLD_START)]),
             "holdout": (rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)])}
    for split, (fit_df, val_df, _) in frames.items():
        assert set(parts[split][0].row_id) == set(fit_df.row_id) and set(parts[split][1].row_id) == set(val_df.row_id)
    parts = {s: tuple(with_premiums(f, pd.concat(frames[s][:2])) for f in pair) for s, pair in parts.items()}
    if args.smoke:
        parts = {s: (f.sample(3000, random_state=0).sort_index(), v.sample(800, random_state=0).sort_index())
                 for s, (f, v) in parts.items()}
    # Graph blocks never rented in train or test get the no-history premium (1, zero rows).
    node_prem = {s: blocks[KEY].merge(frames[s][2], on=KEY, how="left", validate="m:1")[NODE_PREM]
                 .fillna({"NODE_BLOCK_PREM": 1.0, "NODE_LOG_N": 0.0}).to_numpy(float) for s in SPLITS}
    print(f"graph blocks with no rentals in train or test: "
          f"{blocks[KEY].merge(frames['holdout'][2], on=KEY, how='left')['NODE_LOG_N'].isna().sum()}", flush=True)

    (out / "config.json").write_text(json.dumps(dict(vars(args), max_epochs=max_epochs, patience=ewg.PATIENCE,
                                                     level=args.level, smoothing=[M_BLOCK, M_FT]), indent=2))
    predictions = []
    for prem in args.premiums.split(","):
        record_extra = [] if prem == "none" else PREM
        block_num = list(ewg.BLOCK_NUM) + (NODE_PREM if prem == "row+node" else [])
        bundles = {}
        for s, (f, v) in parts.items():
            if prem == "row+node":
                blocks[NODE_PREM] = node_prem[s]  # read by prepare when it builds node features
            bundles[s] = cmp.prepare(f, [v], args.level, block_num, record_extra)
        partner = "nb10_E0_LightGBM" if prem == "none" else "nb10_E1_LightGBM"
        for kind in args.models.split(","):
            tags = dict(model=kind, premiums=prem, seed=args.seed)
            start = time.monotonic()
            _, epochs, history = ewg.train(kind, bundles["tuning"], args.seed, max_epochs, tune=True)
            nn = {}
            for s in SPLITS:
                fitted, _, _ = ewg.train(kind, bundles[s], args.seed, epochs)
                nn[s] = pd.Series(ewg.predict(fitted, bundles[s], bundles[s]["parts"][1]),
                                  index=parts[s][1]["row_id"].to_numpy())
            y = {s: parts[s][1].set_index("row_id")[TARGET] for s in SPLITS}
            gb = {s: gb_preds[(partner, s)].reindex(y[s].index) for s in SPLITS}
            w = best_blend(y["tuning"], nn["tuning"], gb["tuning"])
            for s in SPLITS:
                results.append(dict(split=s, **tags, epochs=epochs, n_rows=len(y[s]), **ewg.metrics(y[s], nn[s])))
                results.append(dict(split=s, model=f"{kind} + {partner.split('_')[1]} blend", premiums=prem,
                                    seed=args.seed, weight=w, n_rows=len(y[s]),
                                    **ewg.metrics(y[s], w * nn[s] + (1 - w) * gb[s])))
                predictions.append(pd.DataFrame({"row_id": y[s].index, "split": s, TARGET: y[s].to_numpy(),
                                                 "neural": nn[s].to_numpy(), "lightgbm": gb[s].to_numpy(), **tags}))
            hold = [r for r in results[-4:] if r["split"] == "holdout"]
            print("RESULT", tags, "epochs", epochs, "holdout", round(hold[0]["rmse"], 2),
                  f"| blend w={w}", "holdout", round(hold[1]["rmse"], 2),
                  "minutes", round((time.monotonic() - start) / 60, 1), flush=True)
            pd.DataFrame(results).to_csv(out / "results.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    r = pd.DataFrame(results)
    arms = r[["model", "premiums"]].drop_duplicates()
    table = r.set_index(["model", "premiums", "split"])["rmse"].unstack("split")
    table = table.reindex(pd.MultiIndex.from_frame(arms))[["tuning", "holdout"]]
    table["weight"] = r[r.split == "holdout"].set_index(["model", "premiums"])["weight"].reindex(table.index) \
        if "weight" in r else np.nan
    for ref in ["E0", "E1"]:
        table[f"holdout vs {ref}"] = table["holdout"] - table.loc[(f"nb10_{ref}_LightGBM", "none" if ref == "E0" else "row"), "holdout"]
    print("\nRMSE (SGD), notebook 10 windows\n", table.round(2).to_string(), flush=True)
    table.round(2).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
