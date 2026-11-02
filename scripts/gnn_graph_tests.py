"""Tests of whether the graph matters: unseen blocks, shuffled edges, a planted-effect sweep, spatial correlation.

    uv run python scripts/gnn_graph_tests.py unseen --seed 0     # also: shuffled, planted (seeds 0, 1, 2)
    uv run python scripts/gnn_graph_tests.py spatial             # no training
    uv run python scripts/gnn_graph_tests.py summary             # tables from the runs above
    add --smoke for 3 epochs on a small sample

Writes outputs/gnn_graph_tests/.

unseen    A random 20% of blocks lose all their rentals from the fit rows, so they have no history. Early
          stopping uses those blocks' rows in the tuning window; the score is their rows in the holdout window.
          Everything is on the `blend` level. LightGBM E1 is refit on the same rows.
          Arms: row (own premiums), row+nbr (plus neighbours' premiums, gnn_neighbour_premiums.py), each for
          MLP_matched and GNN_uniform; GNN row and MLP row+nbr also on shuffled edges.
shuffled  GNN_uniform with `row` premiums on the `blend` level and shuffled edges, on all rows, against the real
          edges in outputs/gnn_premiums_blend/level-blend_same_seed*/.
          Shuffling relabels the blocks at random, so each keeps its number of neighbours and the graph keeps its
          shape, but neighbours are no longer nearby.
planted   Part 3.2's planted effect at several strengths, damped level, no premiums, its epoch counts. A new
          random u per seed. Block premiums are left out because they would absorb a planted block effect.
spatial   Correlation of two blocks' fit-row premiums by distance, against shuffled premiums, raw and within town.
          Blocks with at least 10 fit rows.

Decision rules, fixed before running. Gaps are paired within a seed; "clears" means mean gap > 2 x sd of the gap
over the three seeds. Three seeds make this a rough screen, not a formal test.
    R1 Neighbours' premiums help unseen blocks: row minus row+nbr clears, for that model.
    R2 Message passing helps unseen blocks: MLP minus GNN clears, for row and for row+nbr.
    R3 The real neighbours matter on unseen blocks: shuffled minus real clears, for GNN row and MLP row+nbr.
    R4 The real neighbours matter on all rows: shuffled minus real clears, for GNN row.
    R5 Smallest detectable effect: smallest planted strength where (MLP - GNN) minus the same gap at strength 0 clears.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import gnn_neighbour_premiums as gnp  # noqa: E402  (imports torch before LightGBM)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

gpb, cmp, ewg = gnp.gpb, gnp.cmp, gnp.ewg
OUT = ROOT / "outputs" / "gnn_graph_tests"
TARGET, KEY, PREM, NODE_PREM = cmp.TARGET, cmp.KEY, gpb.PREM, gpb.NODE_PREM
REAL_GRAPH = ewg.GRAPH
HELD_FRACTION = 0.2
LEVEL = "blend"
BETAS = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0]  # 300 pushed some rents below zero on seed 1
EARTH_KM = 6371.0088
UNSEEN_ARMS = [("MLP_matched", "row", "real"), ("GNN_uniform", "row", "real"), ("GNN_uniform", "row", "shuffled"),
               ("MLP_matched", "row+nbr", "real"), ("GNN_uniform", "row+nbr", "real"), ("MLP_matched", "row+nbr", "shuffled")]


def shuffled_graph(seed):
    """The same graph with blocks relabelled at random: same degrees and shape, neighbours no longer nearby."""
    perm = torch.as_tensor(np.random.default_rng(1000 + seed).permutation(REAL_GRAPH.n_nodes), dtype=torch.long)
    fields = {f: getattr(REAL_GRAPH, f) for f in REAL_GRAPH.__dataclass_fields__}
    fields["edge_index"] = perm[REAL_GRAPH.edge_index]
    return type(REAL_GRAPH)(**fields)


def held_frames(tr, held_bid):
    """gnn_premiums_blend.part23_premium_frames with the held blocks' rentals removed from the fit rows."""
    kept = tr[~tr["BID"].isin(held_bid)].copy()
    blocks = tr[KEY + ["BID"]].drop_duplicates("BID").sort_values("BID")
    nb, nft = int(tr["BID"].max()) + 1, int(tr["FT"].max()) + 1
    kept["REL"] = kept[TARGET] / kept.groupby(["MONTH_INDEX", "FLAT_TYPE"])[TARGET].transform("median")

    def encode(src, dst):
        bid, g, rel = src["BID"].to_numpy(), src["G"].to_numpy(), src["REL"].to_numpy()
        sb, cb = np.bincount(bid, rel, nb), np.bincount(bid, minlength=nb).astype(float)
        sg, cg = np.bincount(g, rel, nb * nft), np.bincount(g, minlength=nb * nft).astype(float)
        db, dg = dst["BID"].to_numpy(), dst["G"].to_numpy()
        pb = (gpb.M_BLOCK + sb[db]) / (gpb.M_BLOCK + cb[db])
        pg = (gpb.M_FT * pb + sg[dg]) / (gpb.M_FT + cg[dg])
        return np.column_stack([pb, pg, cg[dg]])

    frames = {}
    for split, (start, end) in gpb.SPLITS.items():
        fit_df = kept[kept["RENT_APPROVAL_DATE"] < start].reset_index(drop=True)
        d = tr["RENT_APPROVAL_DATE"]
        val_df = tr[(d >= start) & (d < end)].reset_index(drop=True)
        own = np.empty((len(fit_df), 3))
        for f_idx, p_idx in gpb.KFold(5, shuffle=True, random_state=0).split(fit_df):
            own[p_idx] = encode(fit_df.iloc[f_idx], fit_df.iloc[p_idx])
        fit_df[PREM], val_df[PREM] = own, encode(fit_df, val_df)
        sb = np.bincount(fit_df["BID"], fit_df["REL"], nb)
        cb = np.bincount(fit_df["BID"], minlength=nb).astype(float)
        node = blocks.assign(NODE_BLOCK_PREM=(gpb.M_BLOCK + sb[blocks["BID"]]) / (gpb.M_BLOCK + cb[blocks["BID"]]),
                             NODE_LOG_N=np.log1p(cb[blocks["BID"]]))[KEY + NODE_PREM]
        frames[split] = (fit_df, val_df, node)
    return frames


def node_premiums(frame):
    x = ewg.blocks[KEY].merge(frame, on=KEY, how="left", validate="m:1")[NODE_PREM] \
        .fillna({"NODE_BLOCK_PREM": 1.0, "NODE_LOG_N": 0.0}).to_numpy(float)
    x[:, 0] = np.log(x[:, 0])
    return torch.tensor((x - x.mean(0)) / x.std(0), dtype=torch.float32)


def set_up(smoke):
    ewg.PATIENCE = 2 if smoke else 30
    return 3 if smoke else 320


def run_unseen(args, out):
    max_epochs = set_up(args.smoke)
    tr, _ = gpb.part23_premium_frames(cmp.part23_block_table())
    rng = np.random.default_rng(100 + args.seed)
    all_bid = tr["BID"].unique()
    held_bid = rng.choice(all_bid, int(HELD_FRACTION * len(all_bid)), replace=False)
    frames = held_frames(tr, held_bid)
    assert not any(f["BID"].isin(held_bid).any() for f, _, _ in frames.values())
    gb_preds, _ = gpb.part23_lgbm(tr, frames, LEVEL)
    held_rows = set(tr.loc[tr["BID"].isin(held_bid), "row_id"])
    rent, d = ewg.rent, ewg.rent.RENT_APPROVAL_DATE
    parts = {"tuning": (rent[d < cmp.TUNE_START], rent[(d >= cmp.TUNE_START) & (d < cmp.HOLD_START)]),
             "holdout": (rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)])}
    bundles = {}
    for s, (f, v) in parts.items():
        f = gpb.with_premiums(f[f.row_id.isin(frames[s][0].row_id)], frames[s][0])
        v = gpb.with_premiums(v, frames[s][1])
        if s == "tuning":
            v = v[v.row_id.isin(held_rows)]
        if args.smoke:
            f, v = f.sample(3000, random_state=0).sort_index(), v.sample(800, random_state=0).sort_index()
        bundles[s] = cmp.prepare(f, [v], LEVEL, list(ewg.BLOCK_NUM), PREM)
        bundles[s]["node_premiums"] = node_premiums(frames[s][2])
    val = bundles["holdout"]["parts"][1]["frame"]
    y, is_held = val[TARGET].to_numpy(float), val["row_id"].isin(held_rows).to_numpy()
    rows = []

    def record(name, premiums, graph, pred, epochs=np.nan):
        for slice_name, mask in [("unseen blocks", is_held), ("seen blocks", ~is_held)]:
            rows.append(dict(seed=args.seed, model=name, premiums=premiums, graph=graph, slice=slice_name,
                             epochs=epochs, n_rows=int(mask.sum()), **ewg.metrics(y[mask], pred[mask])))
        pd.DataFrame(rows).to_csv(out / f"unseen_seed{args.seed}.csv", index=False)

    record("E1 LightGBM", "row", "none", gb_preds[("part23_E1_LightGBM", "holdout")].reindex(val["row_id"]).to_numpy())
    for kind, premiums, graph in UNSEEN_ARMS:
        start = time.monotonic()
        ewg.GRAPH = REAL_GRAPH if graph == "real" else shuffled_graph(args.seed)
        trainer = ewg.train if premiums == "row" else gnp.train
        _, epochs, _ = trainer(kind, bundles["tuning"], args.seed, max_epochs, tune=True)
        fitted, _, _ = trainer(kind, bundles["holdout"], args.seed, epochs)
        record(kind, premiums, graph, ewg.predict(fitted, bundles["holdout"], bundles["holdout"]["parts"][1]), epochs)
        print("RESULT", kind, premiums, graph, "seed", args.seed, "epochs", epochs, rows[-2]["rmse"],
              "minutes", round((time.monotonic() - start) / 60, 1), flush=True)
    ewg.GRAPH = REAL_GRAPH


def run_shuffled(args, out):
    max_epochs = set_up(args.smoke)
    _, frames = gpb.part23_premium_frames(cmp.part23_block_table())
    rent, d = ewg.rent, ewg.rent.RENT_APPROVAL_DATE
    parts = {"tuning": (rent[d < cmp.TUNE_START], rent[(d >= cmp.TUNE_START) & (d < cmp.HOLD_START)]),
             "holdout": (rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)])}
    bundles = {}
    for s, pair in parts.items():
        f, v = (gpb.with_premiums(x, pd.concat(frames[s][:2])) for x in pair)
        if args.smoke:
            f, v = f.sample(3000, random_state=0).sort_index(), v.sample(800, random_state=0).sort_index()
        bundles[s] = cmp.prepare(f, [v], LEVEL, list(ewg.BLOCK_NUM), PREM)
    ewg.GRAPH = shuffled_graph(args.seed)
    _, epochs, _ = ewg.train("GNN_uniform", bundles["tuning"], args.seed, max_epochs, tune=True)
    fitted, _, _ = ewg.train("GNN_uniform", bundles["holdout"], args.seed, epochs)
    ewg.GRAPH = REAL_GRAPH
    part = bundles["holdout"]["parts"][1]
    row = dict(seed=args.seed, model="GNN_uniform", premiums="row", graph="shuffled", epochs=epochs,
               **ewg.metrics(part["y"], ewg.predict(fitted, bundles["holdout"], part)))
    pd.DataFrame([row]).to_csv(out / f"shuffled_seed{args.seed}.csv", index=False)
    print("RESULT", row, flush=True)


def run_planted(args, out):
    set_up(args.smoke)
    rent, d = ewg.rent, ewg.rent.RENT_APPROVAL_DATE
    fit, hold = rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)]
    if args.smoke:
        fit, hold = fit.sample(3000, random_state=0).sort_index(), hold.sample(800, random_state=0).sort_index()
    r = pd.read_csv(ROOT / "outputs/gnn_vs_lightgbm/results.csv")
    r = r[(r.level == "damped") & (r.features == "gnn") & (r.split == "holdout")].set_index("model")["epochs"]
    # Same draw as Part 3.2 for seed 0.
    u = np.random.default_rng(args.seed + 1).standard_normal(REAL_GRAPH.n_nodes)
    row_ids, col_ids = REAL_GRAPH.edge_index.numpy()
    degree = np.bincount(row_ids, minlength=REAL_GRAPH.n_nodes)
    total = np.bincount(row_ids, u[col_ids], minlength=REAL_GRAPH.n_nodes)
    neighbour_mean_u = np.divide(total, degree, out=np.zeros_like(u), where=degree > 0)
    ewg.blocks["PLANTED_U"] = u
    block_num = list(ewg.BLOCK_NUM) + ["PLANTED_U"]
    adjustment = cmp.MarketRentAdjustment("damped").fit(fit)

    def plant(frame, beta):
        out_frame = frame.copy()
        out_frame["MONTHLY_RENT"] = frame.MONTHLY_RENT + beta * neighbour_mean_u[frame.block_id.to_numpy()] * adjustment.multipliers(frame.MONTH_INDEX)
        assert (out_frame.MONTHLY_RENT > 0).all()
        return out_frame

    rows = []
    for beta in BETAS:
        start = time.monotonic()
        fit_p, hold_p = plant(fit, beta), plant(hold, beta)
        bundle = cmp.prepare(fit_p, [hold_p], "damped", block_num)
        effect_sd = float((hold_p.MONTHLY_RENT - hold.MONTHLY_RENT).std())
        for kind in ["MLP_matched", "GNN_uniform"]:
            epochs = 3 if args.smoke else int(r[kind])
            fitted, _, _ = ewg.train(kind, bundle, args.seed, epochs)
            part = bundle["parts"][1]
            rows.append(dict(seed=args.seed, beta=beta, effect_sd=effect_sd, model=kind, epochs=epochs,
                             **ewg.metrics(part["y"], ewg.predict(fitted, bundle, part))))
        pd.DataFrame(rows).to_csv(out / f"planted_seed{args.seed}.csv", index=False)
        print("RESULT beta", beta, "seed", args.seed, [round(x["rmse"], 2) for x in rows[-2:]],
              "minutes", round((time.monotonic() - start) / 60, 1), flush=True)


def run_spatial(args, out):
    _, frames = gpb.part23_premium_frames(cmp.part23_block_table())
    fit = frames["holdout"][0]
    g = fit.groupby(KEY).agg(prem=("REL", "mean"), n=("REL", "size")).reset_index()
    g = g[g.n >= 10].merge(ewg.blocks[KEY + ["LATITUDE", "LONGITUDE"]], on=KEY, how="inner", validate="1:1")
    g["within_town"] = g.prem - g.groupby("TOWN").prem.transform("mean")
    bands = [0, .1, .2, .3, .5, 1, 2, 5, 10]
    tree = BallTree(np.radians(g[["LATITUDE", "LONGITUDE"]].to_numpy()), metric="haversine")
    ids, dist = tree.query_radius(np.radians(g[["LATITUDE", "LONGITUDE"]].to_numpy()), r=bands[-1] / EARTH_KM, return_distance=True)
    a = np.concatenate([np.full(len(i), k) for k, i in enumerate(ids)])
    b, km = np.concatenate(ids), np.concatenate(dist) * EARTH_KM
    keep = a < b
    a, b, km = a[keep], b[keep], km[keep]
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (km > lo) & (km <= hi)
        row = dict(band_km=f"{lo}-{hi}", n_pairs=int(m.sum()))
        for col in ["prem", "within_town"]:
            x = g[col].to_numpy()
            row[f"corr {col}"] = float(np.corrcoef(x[a[m]], x[b[m]])[0, 1])
            null = [np.corrcoef(p[a[m]], p[b[m]])[0, 1] for p in (rng.permutation(x) for _ in range(50))]
            row[f"shuffled sd {col}"] = float(np.std(null))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "spatial.csv", index=False)
    print(f"{len(g)} blocks with at least 10 fit rows\n", table.round(3).to_string(), flush=True)


def clears(gaps):
    gaps = np.asarray(gaps, float)
    return dict(gap=float(gaps.mean()), sd=float(gaps.std(ddof=1)), threshold=float(2 * gaps.std(ddof=1)),
                clears=bool(gaps.mean() > 2 * gaps.std(ddof=1)))


def summarise(out):
    u = pd.concat([pd.read_csv(p) for p in sorted(out.glob("unseen_seed*.csv"))])
    table = u.groupby(["slice", "model", "premiums", "graph"], sort=False).agg(
        rmse=("rmse", "mean"), sd=("rmse", "std"), bias=("bias", "mean"), n_rows=("n_rows", "mean"), epochs=("epochs", "mean"))
    table.round(2).to_csv(out / "unseen_summary.csv")
    held = u[u.slice == "unseen blocks"].set_index(["model", "premiums", "graph", "seed"])["rmse"].unstack("seed")
    arm = lambda m, p, g="real": held.loc[(m, p, g)].to_numpy()
    rules = [("R1 neighbours' premiums help, MLP", arm("MLP_matched", "row") - arm("MLP_matched", "row+nbr")),
             ("R1 neighbours' premiums help, GNN", arm("GNN_uniform", "row") - arm("GNN_uniform", "row+nbr")),
             ("R2 message passing helps, row", arm("MLP_matched", "row") - arm("GNN_uniform", "row")),
             ("R2 message passing helps, row+nbr", arm("MLP_matched", "row+nbr") - arm("GNN_uniform", "row+nbr")),
             ("R3 real neighbours matter, GNN row", arm("GNN_uniform", "row", "shuffled") - arm("GNN_uniform", "row")),
             ("R3 real neighbours matter, MLP row+nbr", arm("MLP_matched", "row+nbr", "shuffled") - arm("MLP_matched", "row+nbr"))]
    s = pd.concat([pd.read_csv(p) for p in sorted(out.glob("shuffled_seed*.csv"))]).set_index("seed").sort_index()
    real = pd.concat([pd.read_csv(ROOT / f"outputs/gnn_premiums_blend/level-{LEVEL}_same_seed{k}/results.csv").assign(seed=k) for k in s.index])
    real = real[(real.split == "holdout") & (real.model == "GNN_uniform")].set_index("seed").sort_index()
    rules.append(("R4 real neighbours matter on all rows, GNN row", s["rmse"].to_numpy() - real["rmse"].to_numpy()))
    rule_table = pd.DataFrame([dict(rule=name, **clears(g)) for name, g in rules])
    rule_table.round(2).to_csv(out / "rules.csv", index=False)
    pd.DataFrame({"graph": ["real", "shuffled"], "rmse": [real.rmse.mean(), s.rmse.mean()], "sd": [real.rmse.std(), s.rmse.std()],
                  "bias": [real.bias.mean(), s.bias.mean()]}).round(2).to_csv(out / "shuffled_summary.csv", index=False)
    p = pd.concat([pd.read_csv(f) for f in sorted(out.glob("planted_seed*.csv"))])
    p = p[p.beta.isin(BETAS)]
    wide = p.pivot(index=["seed", "beta"], columns="model", values="rmse")
    wide["gap"] = wide["MLP_matched"] - wide["GNN_uniform"]
    gap = wide["gap"].unstack("beta")
    rows = []
    for beta in gap.columns:
        beyond = clears((gap[beta] - gap[0.0]).to_numpy()) if beta else dict(gap=0.0, sd=0.0, threshold=0.0, clears=False)
        sub = wide.xs(beta, level="beta")
        rows.append(dict(beta=beta, effect_sd=p[p.beta == beta].effect_sd.mean(), MLP=sub["MLP_matched"].mean(), GNN=sub["GNN_uniform"].mean(),
                         gap=gap[beta].mean(), gap_vs_zero=beyond["gap"], threshold=beyond["threshold"], clears=beyond["clears"]))
    sweep = pd.DataFrame(rows)
    sweep.round(2).to_csv(out / "planted_summary.csv", index=False)
    print(table.round(2).to_string(), "\n\n", rule_table.round(2).to_string(), "\n\n", sweep.round(2).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["unseen", "shuffled", "planted", "spatial", "summary"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--level", default="blend", choices=["blend", "hold"], help="level for unseen and shuffled; other than blend writes to a subfolder")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global LEVEL
    LEVEL = args.level
    base = OUT if LEVEL == "blend" else OUT / f"level-{LEVEL}"
    out = base / "smoke" if args.smoke else base
    out.mkdir(parents=True, exist_ok=True)
    {"unseen": run_unseen, "shuffled": run_shuffled, "planted": run_planted, "spatial": run_spatial,
     "summary": lambda a, o: summarise(o)}[args.experiment](args, out)


if __name__ == "__main__":
    main()
