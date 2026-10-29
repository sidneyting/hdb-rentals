"""Exactness and completeness of the subgraph Integrated Gradients explainer on a toy block graph."""

import unittest

import numpy as np
import torch
from torch import nn

from hdb_rentals.explain import SubgraphExplainer
from hdb_rentals.gnn import WeightedRentalGNN
from test_gnn import toy_graph, toy_inputs, toy_model

KINDS = ["MLP_matched", "GNN_uniform", "GNN_distance", "GNN_similarity", "GNN_learned_edges"]


class PremiumChannelGNN(WeightedRentalGNN):
    """Shaped like scripts/gnn_neighbour_premiums.py: a one-hop channel of neighbours' (never own) values."""

    def __init__(self, graph, premiums):
        super().__init__("GNN_uniform", 5, 3, graph, hidden=8)
        self.register_buffer("premiums", premiums)
        self.register_buffer("neighbour_adjacency", self.normalize(torch.ones(self.edge_index.shape[1])))
        self.premium_in = nn.Linear(premiums.shape[1], 8)
        self.premium_gate = nn.Linear(16, 8)

    def forward(self, nodes, part, edge_mask=None):
        z = torch.relu(self.projection(nodes))
        for root, message, gate in zip(self.roots, self.messages, self.gates):
            incoming = message(torch.sparse.mm(self.fixed_adjacency, z))
            z = torch.relu(root(z) + torch.sigmoid(gate(torch.cat([z, incoming], 1))) * incoming)
        h = torch.sparse.mm(self.neighbour_adjacency, torch.tanh(self.premium_in(self.premiums)))
        z = z + torch.sigmoid(self.premium_gate(torch.cat([z, h], 1))) * h
        return self.head(torch.cat([z[part["ids"]], part["record"]], 1)).squeeze(-1)


class ExplainerTests(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph(n=14)
        self.nodes, self.part = toy_inputs(self.graph, rows=8)
        self.explainer = SubgraphExplainer(self.graph.edge_index, self.graph.n_nodes)
        self.node_base = self.nodes.mean(0)
        self.record_base = self.part["record"].mean(0)

    def test_neighbourhood_is_smaller_than_the_graph(self):
        self.assertLess(len(self.explainer.neighbourhood(7)), self.graph.n_nodes)

    def test_subgraph_output_matches_full_graph(self):
        for kind in KINDS:
            model = toy_model(kind, self.graph)
            with torch.no_grad():
                full = model(self.nodes, self.part).numpy()
            for i, b in enumerate(self.part["ids"].tolist()):
                out = self.explainer.output(model, self.nodes, self.part["record"][i], b)
                self.assertAlmostEqual(out, float(full[i]), places=5, msg=kind)

    def test_completeness_and_graph_free_control(self):
        for kind in KINDS:
            model = toy_model(kind, self.graph)
            for i, b in enumerate(self.part["ids"].tolist()):
                e = self.explainer.explain(model, self.nodes, self.part["record"][i], b, row=i, steps=128,
                                           node_base=self.node_base, record_base=self.record_base)
                gap = e.output - e.baseline
                self.assertAlmostEqual(e.total(), gap, delta=0.02 * max(1.0, abs(gap)), msg=kind)
                if kind == "MLP_matched":
                    self.assertEqual(float(np.abs(e.neighbours).sum()), 0.0)

    def test_premium_channel_is_exact_and_never_reads_own_block(self):
        torch.manual_seed(0)
        premiums = torch.randn(self.graph.n_nodes, 2)
        model = PremiumChannelGNN(self.graph, premiums).eval()
        with torch.no_grad():
            full = model(self.nodes, self.part).numpy()
        base = premiums.mean(0)
        for i, b in enumerate(self.part["ids"].tolist()):
            self.assertAlmostEqual(self.explainer.output(model, self.nodes, self.part["record"][i], b),
                                   float(full[i]), places=5)
            e = self.explainer.explain(model, self.nodes, self.part["record"][i], b, row=i, steps=128,
                                       node_base=self.node_base, record_base=self.record_base,
                                       extra={"premiums": (premiums, base)})
            prem = e.extra["premiums"]
            self.assertEqual(float(np.abs(prem[e.local]).sum()), 0.0)      # own premium never used
            hop2 = self.explainer.hop(b, e.nodes) == 2
            hop2[e.local] = False
            self.assertEqual(float(np.abs(prem[hop2]).sum()), 0.0)         # channel is one hop
            gap = e.output - e.baseline
            self.assertAlmostEqual(e.total(), gap, delta=0.02 * max(1.0, abs(gap)))

    def test_multiscale_is_refused(self):
        with self.assertRaises(ValueError):
            self.explainer.submodel(toy_model("GNN_multiscale", self.graph), self.explainer.neighbourhood(3))


if __name__ == "__main__":
    unittest.main()
