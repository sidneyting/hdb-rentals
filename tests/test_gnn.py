"""Edge-mask and save/load checks for the shared GNN on a small synthetic block graph."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hdb_rentals.gnn import KINDS, build_graph, load_model, save_model, WeightedRentalGNN

GNN_KINDS = [k for k in KINDS if k != "MLP_matched"]


def toy_graph(n=12, seed=0):
    rng = np.random.default_rng(seed)
    blocks = pd.DataFrame(dict(
        LATITUDE=1.35 + rng.uniform(0, 0.01, n), LONGITUDE=103.8 + rng.uniform(0, 0.01, n),
        YEAR_COMPLETED=rng.integers(1970, 2020, n).astype(float), MAX_FLOOR=rng.integers(4, 40, n).astype(float),
        SUBZONE=rng.choice(["a", "b"], n), TOWN=rng.choice(["x", "y"], n), mrt_nearest_km=rng.uniform(0.1, 2, n),
    ))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (j - i) in (1, 2)]
    edges = pd.DataFrame(pairs, columns=["source", "target"])
    edges["distance_km"] = rng.uniform(0.05, 1.0, len(edges))
    return build_graph(blocks, edges)


def toy_inputs(graph, node_dim=5, record_dim=3, rows=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    nodes = torch.randn(graph.n_nodes, node_dim, generator=g)
    part = dict(ids=torch.randint(0, graph.n_nodes, (rows,), generator=g), record=torch.randn(rows, record_dim, generator=g))
    return nodes, part


def toy_model(kind, graph, seed=0):
    torch.manual_seed(seed)
    model = WeightedRentalGNN(kind, 5, 3, graph, hidden=8).eval()
    # Open the gates and give the learned-edge network a non-trivial kernel, so masks matter.
    with torch.no_grad():
        for gate in model.gates:
            gate.bias.fill_(2.0)
        if kind == "GNN_learned_edges":
            torch.nn.init.normal_(model.edge_network[-1].weight)
    return model


class EdgeMaskTests(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph()
        self.nodes, self.part = toy_inputs(self.graph)

    def test_mask_of_ones_reproduces_unmasked_prediction(self):
        ones = torch.ones(self.graph.n_edges)
        for kind in KINDS:
            model = toy_model(kind, self.graph)
            with torch.no_grad():
                torch.testing.assert_close(model(self.nodes, self.part, ones), model(self.nodes, self.part), msg=kind)

    def test_mask_changes_gnn_but_not_mlp(self):
        mask = torch.ones(self.graph.n_edges)
        mask[: self.graph.n_edges // 2] = 0.0
        for kind in KINDS:
            model = toy_model(kind, self.graph)
            with torch.no_grad():
                moved = (model(self.nodes, self.part, mask) - model(self.nodes, self.part)).abs().max().item()
            if kind == "MLP_matched":
                self.assertEqual(moved, 0.0)
            else:
                self.assertGreater(moved, 1e-6, kind)

    def test_gradient_flows_to_edge_mask(self):
        for kind in GNN_KINDS:
            model = toy_model(kind, self.graph)
            mask = torch.ones(self.graph.n_edges, requires_grad=True)
            model(self.nodes, self.part, mask).sum().backward()
            self.assertTrue(torch.isfinite(mask.grad).all(), kind)
            self.assertGreater(mask.grad.abs().sum().item(), 0.0, kind)

    def test_wrong_mask_shape_raises(self):
        model = toy_model("GNN_uniform", self.graph)
        with self.assertRaises(ValueError):
            model(self.nodes, self.part, torch.ones(self.graph.n_edges + 1))


class SaveLoadTests(unittest.TestCase):
    def test_round_trip_is_exact_for_every_kind(self):
        graph = toy_graph()
        nodes, part = toy_inputs(graph)
        with tempfile.TemporaryDirectory() as tmp:
            for kind in KINDS:
                model = toy_model(kind, graph, seed=3)
                path = save_model(model, Path(tmp) / f"{kind}.pt", seed=3, experiment="toy")
                loaded, meta = load_model(path, graph)
                self.assertEqual(meta, dict(seed=3, experiment="toy"))
                with torch.no_grad():
                    torch.testing.assert_close(loaded(nodes, part), model(nodes, part), rtol=0, atol=0, msg=kind)

    def test_load_rejects_a_different_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(toy_model("GNN_uniform", toy_graph(n=12)), Path(tmp) / "m.pt")
            with self.assertRaises(ValueError):
                load_model(path, toy_graph(n=13))


if __name__ == "__main__":
    unittest.main()
