"""Graph checks against the supplied station data and geometric behavior."""

import unittest

import numpy as np
import pandas as pd

from hdb_rentals import config
from hdb_rentals.amenity_graph import build_mrt_2021, distance_features, proximity_edges


class AmenityGraphTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(config.PROJECT_ROOT / "cleaned/sg_mrt_stations_cleaned.csv")

    def test_interchanges_branches_and_historical_exclusions(self):
        nodes, edges, excluded = build_mrt_2021(self.raw)
        by_code = {code: row.amenity_id for row in nodes.itertuples() for code in row.codes.split(";")}
        self.assertEqual(by_code["ew13"], by_code["ns25"])
        self.assertEqual(by_code["cg1"], by_code["dt35"])
        for code in ["cg", "ns3a", "ns6", "dt4", "cc18", "te4", "cr2", "ne18"]:
            self.assertNotIn(code, by_code)
        pairs = set(zip(edges.code_a, edges.code_b))
        for pair in [("ew5", "ew6"), ("ew4", "cg1"), ("cg1", "cg2"), ("cc4", "cc34"),
                     ("cc34", "cc33"), ("ne1", "ne3"), ("ns3", "ns4"),
                     ("ns5", "ns7"), ("dt3", "dt5"), ("cc17", "cc19")]:
            self.assertIn(pair, pairs)
        self.assertNotIn(("cc29", "cc33"), pairs)
        self.assertTrue(nodes.NAME.is_unique)
        self.assertTrue((edges.source != edges.target).all())
        self.assertIn("cr2", set(excluded.CODE))  # Marked open but future at this snapshot.

    def test_planned_intermediate_service_is_not_bridged(self):
        frame = self.raw.copy()
        frame.loc[frame.CODE.eq("ew5"), "STATUS"] = "planned"
        nodes, edges, _ = build_mrt_2021(frame)
        self.assertNotIn("bedok mrt station", set(nodes.NAME))
        self.assertNotIn(("ew4", "ew6"), set(zip(edges.code_a, edges.code_b)))

    def test_proximity_and_counts_agree_with_distance_units(self):
        blocks = pd.DataFrame({"LATITUDE": [1.30, 1.40], "LONGITUDE": [103.80, 103.90]})
        amenities = pd.DataFrame({"LATITUDE": [1.30, 1.3001], "LONGITUDE": [103.80, 103.8001]})
        edges = proximity_edges(blocks, amenities, 1.0)
        self.assertEqual(set(edges.block_id), {0})
        self.assertEqual(len(edges), 2)
        self.assertLess(edges.distance_km.max(), .02)
        features = distance_features(blocks, amenities, "poi")
        np.testing.assert_array_equal(features.poi_within_1km, [2, 0])
        self.assertEqual(features.poi_nearest_km.iloc[0], 0)
        self.assertGreater(features.poi_nearest_km.iloc[1], 10)


if __name__ == "__main__":
    unittest.main()
