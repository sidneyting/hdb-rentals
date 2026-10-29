"""Gated message-passing GNN over HDB blocks, shared by the graph scripts and notebooks.

A rental's prediction is the head applied to its block's embedding and its own record
features (floor area, flat age, flat type, flat model). The embedding comes from two
gated layers that mix in neighbouring blocks over a fixed geographic edge list.

For explanations, ``forward`` takes an optional ``edge_mask``: one value per directed
edge, multiplied into the model's own edge weights before they are normalised per
receiving block. A mask of ones reproduces the unmasked prediction; zeros remove edges.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.neighbors import BallTree
from torch import nn

EARTH_KM = 6371.0088
KINDS = ["MLP_matched", "GNN_uniform", "GNN_distance", "GNN_similarity", "GNN_learned_edges", "GNN_multiscale"]
EDGE_FEATURE_NAMES = [
    "distance_km", "completion_year_gap/20", "max_floor_gap/10",
    "same_subzone", "same_town", "nearest_mrt_distance_gap_km",
]


@dataclass
class Graph:
    """Block graph tensors. ``edge_index`` holds both directions of every edge."""

    n_nodes: int
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    far_adjacency: torch.Tensor
    far_present: torch.Tensor

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]


def build_graph(blocks: pd.DataFrame, edges: pd.DataFrame, *, far_band_km=(0.25, 1.5),
                far_neighbors=8, far_decay_km=0.75) -> Graph:
    """Symmetric geographic edges from ``edges`` (source < target) plus the multiscale far band.

    ``blocks`` must be indexed by block_id 0..N-1 in order.
    """
    n = len(blocks)
    u = edges.source.to_numpy(int)
    v = edges.target.to_numpy(int)
    row = np.r_[u, v]
    col = np.r_[v, u]
    distance = np.tile(edges.distance_km.to_numpy(float), 2)
    year = blocks.YEAR_COMPLETED.fillna(blocks.YEAR_COMPLETED.median()).to_numpy()
    floor = blocks.MAX_FLOOR.fillna(blocks.MAX_FLOOR.median()).to_numpy()
    edge_x = np.column_stack([
        distance,
        abs(year[row] - year[col]) / 20.0,
        abs(floor[row] - floor[col]) / 10.0,
        (blocks.SUBZONE.iloc[row].to_numpy() == blocks.SUBZONE.iloc[col].to_numpy()).astype(float),
        (blocks.TOWN.iloc[row].to_numpy() == blocks.TOWN.iloc[col].to_numpy()).astype(float),
        abs(blocks.mrt_nearest_km.to_numpy()[row] - blocks.mrt_nearest_km.to_numpy()[col]),
    ]).astype("float32")

    # A separate neighbourhood band tests whether the original graph is too local.
    coords = np.radians(blocks[["LATITUDE", "LONGITUDE"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    near, far = far_band_km
    far_ids, far_dist = tree.query_radius(coords, r=far / EARTH_KM, return_distance=True, sort_results=True)
    far_rows, far_cols, far_weights = [], [], []
    for i, (ids, dists) in enumerate(zip(far_ids, far_dist)):
        km = dists * EARTH_KM
        valid = (km >= near) & (km <= far)
        ids, km = ids[valid][:far_neighbors], km[valid][:far_neighbors]
        far_rows.extend([i] * len(ids))
        far_cols.extend(ids)
        far_weights.extend(np.exp(-km / far_decay_km))
    far_sparse = coo_matrix((far_weights, (far_rows, far_cols)), shape=(n, n)).tocsr()
    far_sparse = far_sparse.maximum(far_sparse.T).tocoo()
    far_index = torch.tensor(np.stack([far_sparse.row, far_sparse.col]), dtype=torch.long)
    fw = torch.tensor(far_sparse.data, dtype=torch.float32)
    fd = torch.zeros(n).scatter_add_(0, far_index[0], fw)
    far_adjacency = torch.sparse_coo_tensor(
        far_index, fw / fd[far_index[0]].clamp_min(1e-10), (n, n), check_invariants=True
    ).coalesce()

    return Graph(
        n_nodes=n,
        edge_index=torch.tensor(np.stack([row, col]), dtype=torch.long),
        edge_features=torch.tensor(edge_x),
        far_adjacency=far_adjacency,
        far_present=fd.gt(0).float().unsqueeze(1),
    )


class WeightedRentalGNN(nn.Module):
    """Two gated message-passing layers over blocks, then a head on [block embedding, record].

    ``MLP_matched`` has the same layers with the neighbour aggregation replaced by the
    block's own embedding, so it is the graph-free control.
    """

    def __init__(self, kind: str, node_dim: int, record_dim: int, graph: Graph, hidden: int = 64):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.n_nodes = graph.n_nodes
        self.init_args = dict(kind=kind, node_dim=node_dim, record_dim=record_dim, hidden=hidden)
        # Parameter creation order is fixed: changing it changes every seeded run.
        self.projection = nn.Linear(node_dim, hidden)
        self.roots = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.messages = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.gates = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(2)])
        for gate in self.gates:
            nn.init.constant_(gate.bias, -2.0)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(hidden + record_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
        if kind == "GNN_learned_edges":
            self.edge_network = nn.Sequential(nn.Linear(graph.edge_features.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.edge_network[-1].weight)
            nn.init.zeros_(self.edge_network[-1].bias)
            self.log_tau = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.3)))))
        if kind == "GNN_multiscale":
            self.scale_logits = nn.Parameter(torch.zeros(2))
        # Graph buffers are not saved with the weights; load_model rebuilds them from a Graph.
        self.register_buffer("far_adjacency", graph.far_adjacency, persistent=False)
        self.register_buffer("far_present", graph.far_present, persistent=False)
        self.register_buffer("edge_index", graph.edge_index, persistent=False)
        self.register_buffer("edge_features", graph.edge_features, persistent=False)
        fixed = torch.ones(graph.n_edges)
        if kind == "GNN_distance":
            fixed = torch.exp(-self.edge_features[:, 0] / 0.3)
        if kind == "GNN_similarity":
            fixed = torch.exp(
                -self.edge_features[:, 0] / 0.3
                - self.edge_features[:, 1] * (20 / 15)
                - self.edge_features[:, 2] * (10 / 8)
            )
        self.register_buffer("fixed_weights", fixed, persistent=False)
        if kind not in ["MLP_matched", "GNN_learned_edges"]:
            self.register_buffer("fixed_adjacency", self.normalize(fixed), persistent=False)

    def normalize(self, w: torch.Tensor) -> torch.Tensor:
        """Row-normalise edge weights per receiving block into a sparse adjacency."""
        degree = torch.zeros(self.n_nodes, dtype=w.dtype).scatter_add(0, self.edge_index[0], w)
        w = w / degree[self.edge_index[0]].clamp_min(1e-10)
        return torch.sparse_coo_tensor(self.edge_index, w, (self.n_nodes, self.n_nodes), check_invariants=True).coalesce()

    def edge_weights(self) -> torch.Tensor | None:
        """Raw (unnormalised) weight per directed edge, or None for the graph-free MLP."""
        if self.kind == "MLP_matched":
            return None
        if self.kind == "GNN_learned_edges":
            tau = torch.nn.functional.softplus(self.log_tau).clamp(0.03, 3.0)
            return torch.exp(-self.edge_features[:, 0] / tau) * torch.sigmoid(self.edge_network(self.edge_features).squeeze(-1))
        return self.fixed_weights

    def adjacency(self, edge_mask: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.kind == "MLP_matched":
            return None
        if edge_mask is None and self.kind != "GNN_learned_edges":
            return self.fixed_adjacency
        w = self.edge_weights()
        if edge_mask is not None:
            if edge_mask.shape != w.shape:
                raise ValueError(f"edge_mask must have shape {tuple(w.shape)}, got {tuple(edge_mask.shape)}")
            w = w * edge_mask
        return self.normalize(w)

    def forward(self, nodes: torch.Tensor, part: dict, edge_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Scaled premium per record in ``part`` (needs ``part['ids']`` and ``part['record']``).

        ``edge_mask`` (n_edges,) multiplies the raw edge weights; ignored by MLP_matched.
        """
        z = self.drop(torch.relu(self.projection(nodes)))
        adjacency = self.adjacency(edge_mask)
        for layer, (root, message, gate) in enumerate(zip(self.roots, self.messages, self.gates)):
            # Identity adjacency in MLP keeps the same active layers and gates.
            aggregated = z if adjacency is None else torch.sparse.mm(adjacency, z)
            if self.kind == "GNN_multiscale":
                amount = torch.sigmoid(self.scale_logits[layer]) * self.far_present
                aggregated = (1 - amount) * aggregated + amount * torch.sparse.mm(self.far_adjacency, z)
            incoming = message(aggregated)
            amount = torch.sigmoid(gate(torch.cat([z, incoming], dim=1)))
            z = self.drop(torch.relu(root(z) + amount * incoming))
        return self.head(torch.cat([z[part["ids"]], part["record"]], dim=1)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: WeightedRentalGNN, bundle: dict, part: dict, edge_mask: torch.Tensor | None = None) -> np.ndarray:
    """Rent in dollars: the model's scaled premium, un-standardised, times the market level."""
    model.eval()
    adjusted = model(bundle["nodes"], part, edge_mask).numpy() * bundle["std"] + bundle["mean"]
    return np.maximum(adjusted, 0) * part["scale"]


def train(kind: str, bundle: dict, graph: Graph, seed: int, epochs: int, *, tune: bool = False,
          patience: int = 30, hidden: int = 64, log_every: int = 40):
    """Fit on ``bundle['parts'][0]``. With ``tune``, early-stop on RMSE of ``parts[1]``.

    Returns (model, best_epoch, history). Without ``tune``, best_epoch is ``epochs``.
    """
    seed_everything(seed)
    model = WeightedRentalGNN(kind, bundle["nodes"].shape[1], bundle["parts"][0]["record"].shape[1], graph, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    fit = bundle["parts"][0]
    best = float("inf")
    best_epoch = epochs
    stale = 0
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        err = model(bundle["nodes"], fit) - fit["scaled_y"]
        loss = (fit["weight"] * err.square()).mean()
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        record = dict(epoch=epoch, train_objective=float(loss.detach()))
        if tune:
            val = bundle["parts"][1]
            score = float(np.sqrt(np.mean((predict(model, bundle, val) - val["y"]) ** 2)))
            record["inner_rmse"] = score
            if score < best - 0.01:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(record)
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(kind, seed, "tune" if tune else "refit", epoch, "loss", round(record["train_objective"], 4),
                  "seconds", round(time.monotonic() - start, 1), flush=True)
        if tune and stale >= patience:
            break
    return model, best_epoch, history


def save_model(model: WeightedRentalGNN, path: str | Path, **metadata) -> Path:
    """Save learned weights, constructor arguments and graph size (not the graph itself)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(), init_args=model.init_args,
        n_nodes=model.n_nodes, n_edges=model.edge_index.shape[1], metadata=metadata,
    ), path)
    return path


def load_model(path: str | Path, graph: Graph) -> tuple[WeightedRentalGNN, dict]:
    """Rebuild a saved model on ``graph`` (must match the one it was trained on). Returns (model, metadata)."""
    saved = torch.load(path, weights_only=True)
    if (saved["n_nodes"], saved["n_edges"]) != (graph.n_nodes, graph.n_edges):
        raise ValueError(f"graph mismatch: saved {saved['n_nodes']} nodes / {saved['n_edges']} edges, "
                         f"got {graph.n_nodes} / {graph.n_edges}")
    model = WeightedRentalGNN(graph=graph, **saved["init_args"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["metadata"]
