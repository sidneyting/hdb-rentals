"""Neighbours' premiums through the graph without leakage, and errors by how much history a block has.

    uv run python scripts/gnn_neighbour_premiums.py            # full run
    uv run python scripts/gnn_neighbour_premiums.py --smoke    # 3 epochs, a quick execution check

Same windows, rows, level (damped), block columns and row premiums (`row`, Part 2.3 E1) as
gnn_premiums_blend.py. New here:

Neighbour premium channel (`row+nbr`). gnn_premiums_blend.py's `row+node` put each block's premium,
built from all fit rows, into its own node, so a fit row could read its own rent. Here node
premiums enter only through a separate one-hop channel added after the last message-passing layer:

    h_i = mean over neighbours j of f(premium_j, log rows_j)        (graph has no self-loops)
    z_i <- z_i + gate(z_i, h_i) * W h_i

A block never receives its own premium, and with one hop it cannot come back through a neighbour
(two layers of ordinary message passing would return it). The channel is added to both GNN_uniform
and MLP_matched; the MLP version isolates the neighbour premiums from ordinary message passing.

Breakdown. Holdout RMSE of every model by how many fit rows its block + flat type has (0, 1-2, 3-5,
6-10, >10), the slice where borrowing from neighbours should matter. `row` arms are read from
outputs/gnn_premiums_blend/predictions.csv, so run gnn_premiums_blend.py first.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import gnn_premiums_blend as gpb  # noqa: E402  (imports torch before LightGBM)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

cmp, ewg = gpb.cmp, gpb.ewg
OUT = ROOT / "outputs" / "gnn_neighbour_premiums"
TARGET, KEY = cmp.TARGET, cmp.KEY
BUCKETS = ([-1, 0, 2, 5, 10, np.inf], ["0", "1-2", "3-5", "6-10", ">10"])


class NeighbourPremiumGNN(ewg.WeightedRentalGNN):
    """WeightedRentalGNN plus a one-hop channel carrying neighbours' (never own) premiums."""

    def __init__(self, kind, node_dim, record_dim, premiums):
        super().__init__(kind, node_dim, record_dim, ewg.GRAPH, hidden=ewg.HIDDEN)
        hidden = ewg.HIDDEN
        self.register_buffer("premiums", premiums)
        # Uniform mean over neighbours; edge_index holds both directions and no self-loops.
        self.register_buffer("neighbour_adjacency", self.normalize(torch.ones(self.edge_index.shape[1])))
        self.premium_in = nn.Sequential(nn.Linear(premiums.shape[1], hidden), nn.ReLU())
        self.premium_message = nn.Linear(hidden, hidden, bias=False)
        self.premium_gate = nn.Linear(2 * hidden, hidden)
        nn.init.constant_(self.premium_gate.bias, -2.)

    def forward(self, nodes, part, edge_mask=None):
        # WeightedRentalGNN.forward (uniform / MLP kinds), then the premium channel, then the head.
        # edge_mask multiplies each directed edge in message passing and in the premium channel.
        z = self.drop(torch.relu(self.projection(nodes)))
        adjacency = self.adjacency(edge_mask)
        for root, message, gate in zip(self.roots, self.messages, self.gates):
            aggregated = z if adjacency is None else torch.sparse.mm(adjacency, z)
            incoming = message(aggregated)
            amount = torch.sigmoid(gate(torch.cat([z, incoming], dim=1)))
            z = self.drop(torch.relu(root(z) + amount * incoming))
        neighbours = self.neighbour_adjacency if edge_mask is None else self.normalize(edge_mask)
        h = self.premium_message(torch.sparse.mm(neighbours, self.premium_in(self.premiums)))
        amount = torch.sigmoid(self.premium_gate(torch.cat([z, h], dim=1)))
        self.last_premium_gate = amount.detach()
        z = z + amount * h
        return self.head(torch.cat([z[part["ids"]], part["record"]], dim=1)).squeeze(-1)


def train(kind, bundle, seed, epochs, tune=False):
    # ewg.train, building NeighbourPremiumGNN with the bundle's node premiums.
    ewg.seed_everything(seed)
    model = NeighbourPremiumGNN(kind, bundle["nodes"].shape[1], bundle["parts"][0]["record"].shape[1],
                                bundle["node_premiums"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    fit = bundle["parts"][0]
    best, best_epoch, stale, history = float("inf"), epochs, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (fit["weight"] * (model(bundle["nodes"], fit) - fit["scaled_y"]).square()).mean()
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        record = dict(epoch=epoch, train_objective=float(loss.detach()))
        if tune:
            score = ewg.metrics(bundle["parts"][1]["y"], ewg.predict(model, bundle, bundle["parts"][1]))["rmse"]
            record["inner_rmse"] = score
            if score < best - .01:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(record)
        if epoch == 1 or epoch % 40 == 0:
            print(kind, seed, "tune" if tune else "refit", epoch, "loss", round(record["train_objective"], 4), flush=True)
        if tune and stale >= ewg.PATIENCE:
            break
    return model, best_epoch, history


def by_history(frame, pred_cols):
    bucket = pd.cut(frame["history"], BUCKETS[0], labels=BUCKETS[1])
    rows = []
    for name in pred_cols:
        err = frame[name] - frame[TARGET]
        g = err.groupby(bucket, observed=False)
        rows.append(pd.DataFrame({"model": name, "history": g.size().index.astype(str), "n_rows": g.size().to_numpy(),
                                  "rmse": np.sqrt(g.apply(lambda e: np.mean(e ** 2))).to_numpy(),
                                  "bias": g.mean().to_numpy()}))
        rows.append(pd.DataFrame({"model": [name], "history": ["all"], "n_rows": [len(err)],
                                  "rmse": [float(np.sqrt(np.mean(err ** 2)))], "bias": [float(err.mean())]}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
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

    hdb = cmp.part23_block_table()
    tr, frames = gpb.part23_premium_frames(hdb)
    gb_preds, results = gpb.part23_lgbm(tr, frames)
    for r in results:
        print("REFERENCE", r, flush=True)

    rent, blocks = ewg.rent, ewg.blocks
    d = rent.RENT_APPROVAL_DATE
    parts = {"tuning": (rent[d < cmp.TUNE_START], rent[(d >= cmp.TUNE_START) & (d < cmp.HOLD_START)]),
             "holdout": (rent[d < cmp.HOLD_START], rent[(d >= cmp.HOLD_START) & (d < cmp.HOLD_END)])}
    history = {s: pd.concat(frames[s][:2]).set_index("row_id")["BLOCK_FT_N"] for s in gpb.SPLITS}
    parts = {s: tuple(gpb.with_premiums(f, pd.concat(frames[s][:2])) for f in pair) for s, pair in parts.items()}
    if args.smoke:
        parts = {s: (f.sample(3000, random_state=0).sort_index(), v.sample(800, random_state=0).sort_index())
                 for s, (f, v) in parts.items()}

    # Node premiums per split from that split's fit rows (log premium, log rows), standardised.
    node_premiums = {}
    for s in gpb.SPLITS:
        x = blocks[KEY].merge(frames[s][2], on=KEY, how="left", validate="m:1")[gpb.NODE_PREM] \
            .fillna({"NODE_BLOCK_PREM": 1.0, "NODE_LOG_N": 0.0}).to_numpy(float)
        x[:, 0] = np.log(x[:, 0])
        node_premiums[s] = torch.tensor((x - x.mean(0)) / x.std(0), dtype=torch.float32)

    (out / "config.json").write_text(json.dumps(dict(vars(args), max_epochs=max_epochs, patience=ewg.PATIENCE,
                                                     level="damped", premiums="row+nbr"), indent=2))
    bundles = {}
    for s, (f, v) in parts.items():
        bundles[s] = cmp.prepare(f, [v], "damped", list(ewg.BLOCK_NUM), gpb.PREM)
        bundles[s]["node_premiums"] = node_premiums[s]

    y = {s: parts[s][1].set_index("row_id")[TARGET] for s in gpb.SPLITS}
    gb = {s: gb_preds[("part23_E1_LightGBM", s)].reindex(y[s].index) for s in gpb.SPLITS}
    hold = pd.DataFrame({TARGET: y["holdout"], "history": history["holdout"].reindex(y["holdout"].index),
                         "E1 LightGBM": gb["holdout"]})
    gates = []
    for kind in args.models.split(","):
        tags = dict(model=kind, premiums="row+nbr", seed=args.seed)
        start = time.monotonic()
        _, epochs, _ = train(kind, bundles["tuning"], args.seed, max_epochs, tune=True)
        nn_pred = {}
        for s in gpb.SPLITS:
            fitted, _, _ = train(kind, bundles[s], args.seed, epochs)
            nn_pred[s] = pd.Series(ewg.predict(fitted, bundles[s], bundles[s]["parts"][1]), index=y[s].index)
        g = fitted.last_premium_gate.mean(1).numpy()  # holdout model, per block, eval mode
        gates.append(blocks[KEY].assign(model=kind, premium_gate=g, fit_rows=np.expm1(
            blocks[KEY].merge(frames["holdout"][2], on=KEY, how="left")["NODE_LOG_N"].fillna(0).to_numpy())))
        w = gpb.best_blend(y["tuning"], nn_pred["tuning"], gb["tuning"])
        for s in gpb.SPLITS:
            results.append(dict(split=s, **tags, epochs=epochs, n_rows=len(y[s]), **ewg.metrics(y[s], nn_pred[s])))
            results.append(dict(split=s, model=f"{kind} + E1 blend", premiums="row+nbr", seed=args.seed, weight=w,
                                n_rows=len(y[s]), **ewg.metrics(y[s], w * nn_pred[s] + (1 - w) * gb[s])))
        hold[f"{kind} row+nbr"] = nn_pred["holdout"]
        print("RESULT", tags, "epochs", epochs, "holdout", round(results[-2]["rmse"], 2),
              f"| blend w={w}", "holdout", round(results[-1]["rmse"], 2),
              "| mean premium gate", round(float(g.mean()), 3),
              "minutes", round((time.monotonic() - start) / 60, 1), flush=True)
        pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    pd.concat(gates, ignore_index=True).to_csv(out / "premium_gates.csv", index=False)

    # `row` arms from gnn_premiums_blend.py, same seed and windows.
    prev = ROOT / "outputs" / "gnn_premiums_blend" / ("smoke" if args.smoke else "") / "predictions.csv"
    if prev.exists():
        p = pd.read_csv(prev)
        p = p[(p.split == "holdout") & (p.premiums == "row")]
        for kind, g in p.groupby("model"):
            hold[f"{kind} row"] = g.set_index("row_id")["neural"].reindex(hold.index)
    hold.reset_index().to_csv(out / "holdout_predictions.csv", index=False)
    cols = [c for c in hold.columns if c not in (TARGET, "history") and hold[c].notna().all()]
    breakdown = by_history(hold, cols)
    breakdown.round(2).to_csv(out / "by_history.csv", index=False)

    r = pd.DataFrame(results)
    arms = r[["model", "premiums"]].drop_duplicates()
    summary = r.set_index(["model", "premiums", "split"])["rmse"].unstack("split") \
        .reindex(pd.MultiIndex.from_frame(arms))[["tuning", "holdout"]]
    summary["weight"] = r[r.split == "holdout"].set_index(["model", "premiums"]).get("weight")
    summary.round(2).to_csv(out / "summary.csv")
    print("\nRMSE (SGD), Part 2.3 windows\n", summary.round(2).to_string(), flush=True)
    table = breakdown.pivot(index="model", columns="history", values="rmse").reindex(cols)[BUCKETS[1] + ["all"]]
    sizes = breakdown.drop_duplicates("history").set_index("history")["n_rows"].reindex(BUCKETS[1] + ["all"])
    print("\nHoldout RMSE by fit rows in the same block + flat type\n", table.round(1).to_string(),
          "\nrows", sizes.to_dict(), flush=True)


if __name__ == "__main__":
    main()
