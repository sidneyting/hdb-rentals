"""Explain the model the report scores: GNN_uniform with `row` premiums on the `blend` level.

    uv run python scripts/explain_table2_gnn.py [--level hold]

Same frames, inputs, level and epoch counts as `gnn_premiums_blend.py --premiums row --level blend
--same-level` (seeds 0, 1, 2), with Part 3.2's explainer and checks. The own-block premiums are row
inputs here, so IG attributes them with the flat's record, and neighbours carry only block features.
The planted-effect check is not repeated: a block's premium would absorb a planted neighbour effect
(Part 2.4), so it stays a check of the method on the premium-free GNN (Part 3.2).

Writes outputs/13_interpretability_table2/: models/, accuracy.csv, importance.csv, reliance.csv,
substitution.csv, reach.csv, checks.csv, fidelity.csv, stability.csv, neighbour_checks.csv,
by_history.csv and two figures.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import gnn_premiums_blend as pb  # noqa: E402  (imports torch before LightGBM)
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hdb_rentals import gnn  # noqa: E402
from hdb_rentals.explain import SubgraphExplainer  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

cmp, ewg = pb.cmp, pb.ewg
LEVEL = "hold" if "--level" in sys.argv and sys.argv[sys.argv.index("--level") + 1] == "hold" else "blend"
OUT = ROOT / "outputs" / "13_interpretability_table2"
if LEVEL != "blend":
    OUT = OUT / f"level-{LEVEL}"
SEEDS, N_ROWS, N_PER_BUCKET, IG_STEPS = [0, 1, 2], 400, 120, 32
RESULTS = ROOT / "outputs" / "gnn_premiums_blend" / f"level-{LEVEL}_same_seed{{}}" / "results.csv"
BLOCK_CAT = ewg.BLOCK_CAT + ["SUBZONE"]
RECORD_NUM, RECORD_CAT = ["FLOOR_AREA_SQM", "FLAT_AGE", *pb.PREM], ["FLAT_TYPE", "FLAT_MODEL"]
FLAT_FAMS = ["Floor area", "Flat type", "Flat model", "Flat age"]
FAMS = [*FLAT_FAMS, "Block premiums", "Town and area", "Location and building", "Neighbouring blocks"]
BUCKETS = ["0", "1-2", "3-5", "6-10", ">10"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titlesize": 11, "font.size": 9.5, "axes.axisbelow": True,
})


def source_column(name, cat):
    kind, rest = name.split("__", 1)
    if kind == "num":
        return rest.removeprefix("missingindicator_")
    return next(c for c in cat if rest.startswith(c + "_"))


def family(col):
    named = {"FLOOR_AREA_SQM": "Floor area", "FLAT_AGE": "Flat age", "LEASE_COMMENCE_DATE": "Flat age",
             "FLAT_TYPE": "Flat type", "FLAT_MODEL": "Flat model", "TOWN": "Town and area",
             "PLANNING_AREA": "Town and area", "SUBZONE": "Town and area"}
    if col in pb.PREM:
        return "Block premiums"
    return named.get(col, "Location and building")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ewg.PATIENCE = 30

    # The same data flow as gnn_premiums_blend.main, for the holdout split.
    hdb = cmp.part23_block_table()
    tr, frames = pb.part23_premium_frames(hdb)
    rent, blocks, graph = ewg.rent, ewg.blocks, ewg.GRAPH
    d = rent.RENT_APPROVAL_DATE
    fit_rent, hold_rent = rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)]
    fit_lgb, hold_lgb, _ = frames["holdout"]
    all_prem = pd.concat([fit_lgb, hold_lgb])
    fit, hold = pb.with_premiums(fit_rent, all_prem), pb.with_premiums(hold_rent, all_prem)
    bundle = cmp.prepare(fit, [hold], LEVEL, list(ewg.BLOCK_NUM), pb.PREM)
    part, nodes = bundle["parts"][1], bundle["nodes"]
    dollars = bundle["std"] * part["scale"]

    # Column names, by re-fitting the preprocessors; the asserts check they match prepare's matrices.
    bp = ewg.preprocessor(list(ewg.BLOCK_NUM), BLOCK_CAT).fit(blocks.iloc[np.sort(fit.block_id.unique())])
    rp = ewg.preprocessor(RECORD_NUM, RECORD_CAT).fit(fit)
    assert np.allclose(bp.transform(blocks), nodes.numpy(), atol=1e-5)
    assert np.allclose(rp.transform(hold), part["record"].numpy(), atol=1e-5)
    node_fam = np.array([family(source_column(n, BLOCK_CAT)) for n in bp.get_feature_names_out()])
    rec_names = [source_column(n, RECORD_CAT) for n in rp.get_feature_names_out()]
    rec_fam = np.array([family(c) for c in rec_names])
    premium_cols = np.flatnonzero(rec_fam == "Block premiums")  # reset together: they are one signal
    count_col = rec_names.index("BLOCK_FT_N")
    level_cols = [c for c in premium_cols if c != count_col]

    # Models, on the epoch counts of the Table II runs.
    recorded = {s: pd.read_csv(str(RESULTS).format(s)).query("split == 'holdout' and premiums == 'row'").set_index("model")
                for s in SEEDS}

    def get_model(kind, seed):
        path = OUT / "models" / f"{kind}_seed{seed}.pt"
        if path.exists():
            return gnn.load_model(path, graph)[0]
        model, _, _ = ewg.train(kind, bundle, seed, int(recorded[seed].epochs[kind]), save_to=path)
        return model.eval()

    models = {f"GNN seed {s}": get_model("GNN_uniform", s) for s in SEEDS}
    models["MLP"] = get_model("MLP_matched", 0)
    accuracy = [{"model": name, "holdout rmse": rmse(gnn.predict(m, bundle, part), part["y"]),
                 "table ii run": recorded[0 if name == "MLP" else int(name[-1])].rmse["MLP_matched" if name == "MLP" else "GNN_uniform"]}
                for name, m in models.items()]
    gnn.seed_everything(99)
    models["GNN untrained"] = gnn.WeightedRentalGNN("GNN_uniform", nodes.shape[1], part["record"].shape[1], graph).eval()

    # LightGBM with premiums on the same level, as in Table II, for TreeSHAP.
    num, cat = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"] + cmp.DIST, ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
    categories = {c: pd.CategoricalDtype(sorted(tr[c].unique())) for c in cat}

    def matrix(df):
        X = df[num + pb.PREM + cat].copy()
        for c in cat:
            X[c] = X[c].astype(categories[c])
        return X

    adj = cmp.MarketRentAdjustment(LEVEL).fit(fit_rent)
    fs, vs = adj.multipliers(fit_lgb["MONTH_INDEX"]), adj.multipliers(hold_lgb["MONTH_INDEX"])
    est = lgb.LGBMRegressor(**cmp.LGBM_TUNED, verbose=-1, n_jobs=-1, random_state=0)
    est.fit(matrix(fit_lgb), fit_lgb[cmp.TARGET].to_numpy() / fs, sample_weight=fs ** 2)
    pred = est.predict(matrix(hold_lgb)) * vs
    accuracy.append({"model": "LightGBM", "holdout rmse": rmse(pred, hold_lgb[cmp.TARGET].to_numpy()),
                     "table ii run": recorded[0].rmse["part23_E1_LightGBM"]})
    pd.DataFrame(accuracy).round(2).to_csv(OUT / "accuracy.csv", index=False)
    print(pd.DataFrame(accuracy).round(2).to_string(), flush=True)

    # Rows: a random sample for overall results, and a sample per history bucket.
    explainer = SubgraphExplainer(graph.edge_index, graph.n_nodes)
    node_base, rec_base = nodes[np.sort(fit.block_id.unique())].mean(0), bundle["parts"][0]["record"].mean(0)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(hold), N_ROWS, replace=False))
    history = pd.cut(np.expm1(hold["BLOCK_FT_N"].to_numpy()).round(), [-1, 0, 2, 5, 10, np.inf], labels=BUCKETS)
    bucket_rows = {b: np.sort(rng.choice(np.flatnonzero(history == b), N_PER_BUCKET, replace=False)) for b in BUCKETS}

    @torch.no_grad()
    def full_outputs(model, edge_mask=None):
        return model(nodes, part, edge_mask).numpy()

    def explain(model, which):
        f_full = full_outputs(model)
        out = []
        for i in which:
            e = explainer.explain(model, nodes, part["record"][i], int(part["ids"][i]), row=int(i),
                                  node_base=node_base, record_base=rec_base, steps=IG_STEPS)
            out.append(dict(row=int(i), block=e.block, S=e.nodes, lb=e.local, rec=e.record, own=e.own, nbr=e.neighbours,
                            nbr_blocks=e.neighbour_blocks, f_full=float(f_full[i]), f_sub=e.output, base=e.baseline))
        return out

    def history_parts(e):
        d_i, rec, own = dollars[e["row"]], e["rec"], pd.Series(e["own"]).groupby(node_fam).sum()
        return {"own premiums": abs(rec[level_cols].sum()) * d_i, "own row count": abs(rec[count_col]) * d_i,
                "own block features": abs(own[["Town and area", "Location and building"]].sum()) * d_i,
                "neighbours": abs(e["nbr"].sum()) * d_i,
                "flat": abs(rec[[i for i in range(len(rec)) if i not in premium_cols]].sum()) * d_i}

    def families(e):
        d_i = dollars[e["row"]]
        s = pd.concat([pd.Series(e["rec"] * d_i).groupby(rec_fam).sum(), pd.Series(e["own"] * d_i).groupby(node_fam).sum()])
        s = s.groupby(level=0).sum()
        s["Neighbouring blocks"] = float(e["nbr"].sum() * d_i)
        return s.reindex(FAMS).fillna(0)

    explanations = {name: explain(m, rows) for name, m in models.items()}
    tables = {name: pd.DataFrame([families(e) for e in ex]) for name, ex in explanations.items()}
    print("explained overall rows", flush=True)

    # Which features matter, against TreeSHAP on the same rows.
    row_ids = hold["row_id"].to_numpy()[rows]
    positions = pd.Series(np.arange(len(hold_lgb)), index=hold_lgb["row_id"].to_numpy()).loc[row_ids].to_numpy()
    X = matrix(hold_lgb).iloc[positions]
    contrib = est.predict(X, pred_contrib=True)[:, :-1] * vs[positions, None]
    shap = pd.DataFrame(contrib, columns=X.columns).T.groupby(lambda c: family(c)).sum().T
    importance = pd.DataFrame({"GNN (IG)": tables["GNN seed 0"].abs().mean(), "MLP (IG)": tables["MLP"].abs().mean(),
                               "LightGBM (TreeSHAP)": shap.abs().mean().reindex(FAMS)})
    importance.round(1).to_csv(OUT / "importance.csv")
    both = importance.drop("Neighbouring blocks")
    print(importance.round(1).to_string(), "\nGNN vs LightGBM Spearman", spearmanr(both["GNN (IG)"], both["LightGBM (TreeSHAP)"]).statistic, flush=True)

    # Does the GNN use its neighbours?
    reliance = []
    for name in ["GNN seed 0", "GNN seed 1", "GNN seed 2", "MLP"]:
        m, ex = models[name], explanations[name]
        nbr = np.array([np.abs(e["nbr"]).sum() * dollars[e["row"]] for e in ex])
        tot = np.array([(np.abs(e["rec"]).sum() + np.abs(e["own"]).sum() + np.abs(e["nbr"]).sum()) * dollars[e["row"]] for e in ex])
        moved = np.abs(full_outputs(m) - full_outputs(m, torch.zeros(graph.n_edges))) * dollars
        t = tables[name].abs().mean()
        reliance.append({"model": name, "neighbour share": nbr.sum() / tot.sum(),
                         "neighbour share of net effects": t["Neighbouring blocks"] / t.sum(),
                         "mean moved with no edges": moved.mean()})
    pd.DataFrame(reliance).round(3).to_csv(OUT / "reliance.csv", index=False)

    def location_effect(t, parts):
        return t[parts].sum(axis=1).abs().mean()

    g, m = tables["GNN seed 0"], tables["MLP"]
    block_fams = ["Block premiums", "Town and area", "Location and building"]
    pd.Series({"GNN: own block, premiums included": location_effect(g, block_fams),
               "GNN: neighbours": g["Neighbouring blocks"].abs().mean(),
               "GNN: own block + neighbours": location_effect(g, [*block_fams, "Neighbouring blocks"]),
               "MLP: own block, premiums included": location_effect(m, block_fams)}).round(1).to_csv(OUT / "substitution.csv")

    records = []
    for e in explanations["GNN seed 0"]:
        hop = explainer.hop(e["block"], e["nbr_blocks"])
        records.extend({"hop": int(h), "dollars": abs(a) * dollars[e["row"]]} for h, a in zip(hop, e["nbr"].sum(1)))
    pd.DataFrame(records).groupby("hop").dollars.agg(["size", "mean"]).round(2).to_csv(OUT / "reach.csv")

    # Checks: exactness, completeness, seed stability, randomisation, fidelity.
    checks = []
    for name, ex in explanations.items():
        comp = [abs(e["rec"].sum() + e["own"].sum() + e["nbr"].sum() - (e["f_sub"] - e["base"])) * dollars[e["row"]] for e in ex]
        checks.append({"model": name, "max subgraph vs full": max(abs(e["f_sub"] - e["f_full"]) for e in ex),
                       "median completeness error $": np.median(comp), "max completeness error $": np.max(comp)})
    pd.DataFrame(checks).to_csv(OUT / "checks.csv", index=False)

    def row_spearman(a, b):
        return float(np.nanmean([spearmanr(x, y).statistic for x, y in zip(tables[a].abs().to_numpy(), tables[b].abs().to_numpy())]))

    pairs = [("GNN seed 0", "GNN seed 1"), ("GNN seed 0", "GNN seed 2"), ("GNN seed 1", "GNN seed 2"),
             ("GNN seed 0", "GNN untrained"), ("GNN seed 0", "MLP")]
    pd.DataFrame([{"pair": f"{a} vs {b}", "mean per-rental spearman": row_spearman(a, b)} for a, b in pairs]).round(3).to_csv(OUT / "stability.csv", index=False)

    @torch.no_grad()
    def fidelity(name, ks=(1, 3, 5, 10, 20), draws=5):
        r_ng = np.random.default_rng(0)
        model, res = models[name], {k: ([], [], []) for k in ks}
        for e in explanations[name]:
            i, S, lb = e["row"], e["S"], e["lb"]
            sub = explainer.submodel(model, S)
            x, r = nodes[S].clone(), part["record"][i].clone()
            attr = np.r_[e["rec"], np.insert(e["nbr"], lb, e["own"], axis=0).ravel()]
            diff = np.r_[(r - rec_base).numpy(), (x - node_base).numpy().ravel()]
            pool = np.flatnonzero(np.abs(diff) > 1e-9)
            n_local = len(r) + (lb + 1) * x.shape[1]
            own_pool = pool[(pool < len(r)) | ((pool >= len(r) + lb * x.shape[1]) & (pool < n_local))]
            top = pool[np.argsort(-np.abs(attr[pool]))]
            base_flat = np.r_[rec_base.numpy(), node_base.expand_as(x).numpy().ravel()]
            full_flat = np.r_[r.numpy(), x.numpy().ravel()]

            def score(idx):
                if np.isin(idx, premium_cols).any():
                    idx = np.union1d(idx, premium_cols)
                v = full_flat.copy()
                v[idx] = base_flat[idx]
                rr = torch.tensor(v[:len(r)], dtype=torch.float32)
                xx = torch.tensor(v[len(r):].reshape(x.shape), dtype=torch.float32)
                return float(sub(xx, dict(ids=torch.tensor([lb]), record=rr[None]))[0])

            for k in ks:
                if k > len(pool):
                    continue
                res[k][0].append(abs(score(top[:k]) - e["f_sub"]) * dollars[i])
                res[k][1].append(np.mean([abs(score(r_ng.choice(pool, k, replace=False)) - e["f_sub"]) * dollars[i] for _ in range(draws)]))
                if k <= len(own_pool):
                    res[k][2].append(np.mean([abs(score(r_ng.choice(own_pool, k, replace=False)) - e["f_sub"]) * dollars[i] for _ in range(draws)]))
        return pd.DataFrame({k: {"top": np.mean(t), "random whole subgraph": np.mean(rd), "random flat and own block": np.mean(ro)}
                             for k, (t, rd, ro) in res.items()}).T

    fidelity("GNN seed 0").round(1).to_csv(OUT / "fidelity.csv")

    # Do the attributions rank individual direct neighbours? Reset one neighbour's features to the
    # baseline, the most-attributed against a random one, and compare the ranking across seeds.
    @torch.no_grad()
    def neighbour_checks(name, seed=0):
        r_ng, top, rand = np.random.default_rng(seed), [], []
        model = models[name]
        for e in explanations[name]:
            direct = np.flatnonzero(explainer.hop(e["block"], e["nbr_blocks"]) == 1)
            if len(direct) < 2:
                continue
            i, S, lb = e["row"], e["S"], e["lb"]
            sub = explainer.submodel(model, S)

            def moved(k):
                x = nodes[S].clone()
                x[int(np.searchsorted(S, e["nbr_blocks"][k]))] = node_base
                return abs(float(sub(x, dict(ids=torch.tensor([lb]), record=part["record"][i][None]))[0]) - e["f_sub"]) * dollars[i]

            totals = np.abs(e["nbr"][direct].sum(1))
            top.append(moved(direct[int(np.argmax(totals))]))
            rand.append(moved(r_ng.choice(direct)))
        return {"top neighbour moved": float(np.mean(top)), "random neighbour moved": float(np.mean(rand)), "rows": len(top)}

    def neighbour_rank(a, b):
        vals = []
        for ea, eb in zip(explanations[a], explanations[b]):
            direct = np.flatnonzero(explainer.hop(ea["block"], ea["nbr_blocks"]) == 1)
            if len(direct) >= 3:
                vals.append(spearmanr(np.abs(ea["nbr"][direct].sum(1)), np.abs(eb["nbr"][direct].sum(1))).statistic)
        return float(np.nanmean(vals))

    nc = neighbour_checks("GNN seed 0")
    for a, b in pairs:
        if "MLP" not in b:
            nc[f"neighbour rank {a} vs {b}"] = neighbour_rank(a, b)
    pd.Series(nc).round(3).to_csv(OUT / "neighbour_checks.csv")
    print("checks done", flush=True)

    # By history: the rows where neighbours should matter most.
    history_rows = []
    for name in ["GNN seed 0", "MLP"]:
        for bucket, which in bucket_rows.items():
            t = pd.DataFrame([history_parts(e) for e in explain(models[name], which)]).mean()
            history_rows.append({"model": name, "history": bucket, **t.to_dict()})
    by_history = pd.DataFrame(history_rows)
    by_history.round(1).to_csv(OUT / "by_history.csv", index=False)
    print(by_history.round(1).to_string(), flush=True)

    # Figures.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(FAMS))
    for j, (col, color) in enumerate(zip(["GNN (IG)", "LightGBM (TreeSHAP)"], [BLUE, ORANGE])):
        ax.barh(y + (j - 0.5) * 0.38, importance[col].fillna(0), height=0.34, color=color, label=col)
    ax.set_yticks(y, FAMS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |contribution|, SGD per month")
    ax.set_title("What moves the prediction away from an average flat")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "importance_table2.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    parts = [("own premiums", "Own block's premiums", BLUE), ("own block features", "Own block's features", AQUA), ("neighbours", "Neighbouring blocks", ORANGE)]
    for ax, name, title in zip(axes, ["GNN seed 0", "MLP"], ["GNN", "MLP (graph off)"]):
        t = by_history[by_history.model == name].set_index("history").reindex(BUCKETS)
        x = np.arange(len(BUCKETS))
        for j, (col, label, color) in enumerate(parts):
            ax.bar(x + (j - 1) * 0.27, t[col], width=0.24, color=color, label=label)
        ax.set_xticks(x, BUCKETS)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.set_xlabel("earlier training rentals of the same block + flat type")
    axes[0].set_ylabel("mean |contribution|, SGD per month")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "history_table2.png", dpi=200)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
