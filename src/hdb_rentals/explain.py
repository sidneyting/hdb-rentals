"""Integrated Gradients for the block GNNs, computed on each rental's exact receptive field.

A rental's prediction depends only on blocks within ``hops`` of its own block (two message-passing
layers, plus any one-hop channel read after them, stay within two hops). ``SubgraphExplainer``
cuts a trained model down to that neighbourhood by slicing its graph buffers:

- dense buffers with one row per block (e.g. ``premiums``, ``far_present``) keep the rows of the
  neighbourhood;
- sparse block x block buffers (``fixed_adjacency``, ``neighbour_adjacency``) keep the
  neighbourhood's sub-matrix. The rows of the rental's block and its direct neighbours are complete
  (all their edges stay inside a 2-hop neighbourhood), so their normalisation is unchanged; only
  the outer ring's rows are cut, and those never reach the rental;
- ``edge_index`` / ``edge_features`` / ``fixed_weights`` keep the edges inside the neighbourhood, so
  learned or masked edge weights normalise the same way for the same rows.

Integrated Gradients is then run with all steps batched as disjoint copies of the neighbourhood.
``check`` compares the cut model with the full one; notebooks assert on it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix


@dataclass
class Explanation:
    """Attributions for one rental, in model output units (sum ~ f(x) - f(baseline))."""

    row: int
    block: int
    nodes: np.ndarray            # block ids of the neighbourhood, sorted
    local: int                   # position of the rental's block in ``nodes``
    record: np.ndarray           # (record_dim,)
    node: np.ndarray             # (len(nodes), node_dim)
    extra: dict                  # buffer name -> (len(nodes), width) attributions
    output: float                # model output on the neighbourhood
    baseline: float              # model output at the baseline

    @property
    def own(self) -> np.ndarray:
        return self.node[self.local]

    @property
    def neighbours(self) -> np.ndarray:
        return np.delete(self.node, self.local, axis=0)

    @property
    def neighbour_blocks(self) -> np.ndarray:
        return np.delete(self.nodes, self.local)

    def total(self) -> float:
        return float(self.record.sum() + self.node.sum() + sum(v.sum() for v in self.extra.values()))


class SubgraphExplainer:
    def __init__(self, edge_index: torch.Tensor, n_nodes: int, hops: int = 2):
        self.recv, self.send = edge_index.numpy()
        self.n_nodes = n_nodes
        self.hops = hops
        self.adjacency = csr_matrix((np.ones(len(self.recv)), (self.recv, self.send)), shape=(n_nodes, n_nodes))

    def neighbourhood(self, block: int) -> np.ndarray:
        reached = np.array([block])
        frontier = reached
        for _ in range(self.hops):
            frontier = self.adjacency[frontier].indices
            reached = np.union1d(reached, frontier)
        return reached

    def hop(self, block: int, others: np.ndarray) -> np.ndarray:
        """1 for direct neighbours of ``block``, 2 for the rest of ``others``."""
        return np.where(np.isin(others, self.adjacency[block].indices), 1, 2)

    def submodel(self, model: torch.nn.Module, nodes: np.ndarray, copies: int = 1) -> torch.nn.Module:
        if getattr(model, "kind", None) == "GNN_multiscale":
            raise ValueError("GNN_multiscale reads a 1.5 km band, beyond the 2-hop neighbourhood")
        n, total = len(nodes), len(nodes) * copies
        local = np.full(self.n_nodes, -1)
        local[nodes] = np.arange(n)
        keep = (local[self.recv] >= 0) & (local[self.send] >= 0)
        keep_t = torch.from_numpy(keep)
        offsets = torch.arange(copies) * n
        idx = torch.from_numpy(nodes)
        sub = copy.deepcopy(model).eval()
        for name, buf in list(sub.named_buffers(recurse=False)):
            if buf.is_sparse and buf.shape == (self.n_nodes, self.n_nodes):
                setattr(sub, name, self._sparse_block(buf, local, copies, n))
            elif name == "edge_index":
                ei = torch.from_numpy(np.stack([local[self.recv[keep]], local[self.send[keep]]]))
                sub.edge_index = ei.repeat(1, copies) + offsets.repeat_interleave(int(keep.sum()))
            elif name in ("edge_features", "fixed_weights"):
                part = buf[keep_t]
                setattr(sub, name, part.repeat(copies, *([1] * (part.dim() - 1))))
            elif not buf.is_sparse and buf.dim() >= 1 and buf.shape[0] == self.n_nodes:
                setattr(sub, name, buf[idx].repeat(copies, *([1] * (buf.dim() - 1))))
        sub.n_nodes = total
        return sub

    @staticmethod
    def _sparse_block(matrix: torch.Tensor, local: np.ndarray, copies: int, n: int) -> torch.Tensor:
        matrix = matrix.coalesce()
        r, c = matrix.indices().numpy()
        inside = (local[r] >= 0) & (local[c] >= 0)
        rows = torch.from_numpy(local[r[inside]])
        cols = torch.from_numpy(local[c[inside]])
        vals = matrix.values()[torch.from_numpy(inside)]
        offsets = (torch.arange(copies) * n).repeat_interleave(len(rows))
        index = torch.stack([rows.repeat(copies) + offsets, cols.repeat(copies) + offsets])
        return torch.sparse_coo_tensor(index, vals.repeat(copies), (n * copies, n * copies), check_invariants=True).coalesce()

    @torch.no_grad()
    def output(self, model, nodes_x: torch.Tensor, record: torch.Tensor, block: int, extra: dict | None = None) -> float:
        """Model output for one rental, evaluated on its neighbourhood (``extra``: full-graph buffers)."""
        S = self.neighbourhood(block)
        sub = self.submodel(model, S)
        for name, value in (extra or {}).items():
            setattr(sub, name, value[torch.from_numpy(S)])
        lb = int(np.searchsorted(S, block))
        return float(sub(nodes_x[torch.from_numpy(S)], dict(ids=torch.tensor([lb]), record=record[None]))[0])

    def explain(self, model, nodes_x: torch.Tensor, record: torch.Tensor, block: int, *, row: int,
                node_base: torch.Tensor, record_base: torch.Tensor, extra: dict | None = None,
                steps: int = 32) -> Explanation:
        """Integrated Gradients from (node_base everywhere, record_base, extra baselines) to the inputs.

        ``extra`` maps a dense per-block buffer name to (full-graph values, baseline row); those
        buffers are attributed like node features.
        """
        extra = extra or {}
        S = self.neighbourhood(block)
        lb = int(np.searchsorted(S, block))
        idx = torch.from_numpy(S)
        sub = self.submodel(model, S, copies=steps)
        alphas = ((torch.arange(steps) + 0.5) / steps)[:, None, None]
        x = nodes_x[idx]
        X = (node_base + alphas * (x - node_base)).reshape(-1, x.shape[1]).requires_grad_()
        R = (record_base + alphas[:, :, 0] * (record - record_base)).requires_grad_()
        E = {}
        for name, (values, base) in extra.items():
            v = values[idx]
            E[name] = (base + alphas * (v - base)).reshape(-1, v.shape[1]).requires_grad_()
            setattr(sub, name, E[name])
        y = sub(X, dict(ids=torch.arange(steps) * len(S) + lb, record=R))
        grads = torch.autograd.grad(y.sum(), [X, R, *E.values()])
        node_attr = (x - node_base) * grads[0].view(steps, len(S), -1).mean(0)
        rec_attr = (record - record_base) * grads[1].mean(0)
        extra_attr = {}
        for (name, (values, base)), g in zip(extra.items(), grads[2:]):
            extra_attr[name] = ((values[idx] - base) * g.view(steps, len(S), -1).mean(0)).numpy()
        with torch.no_grad():
            ones = self.submodel(model, S)
            for name, (values, base) in extra.items():
                setattr(ones, name, base.expand(len(S), -1).clone())
            baseline = float(ones(node_base.expand(len(S), -1).clone(),
                                  dict(ids=torch.tensor([lb]), record=record_base[None]))[0])
        out = self.output(model, nodes_x, record, block, {k: v for k, (v, _) in extra.items()})
        return Explanation(row=row, block=block, nodes=S, local=lb, record=rec_attr.numpy(),
                           node=node_attr.numpy(), extra=extra_attr, output=out, baseline=baseline)
