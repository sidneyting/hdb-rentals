"""Spatial relations and a conservative MRT network for graph experiments.

The rail manifest describes services operating by January 2021. It intentionally
does not trust the auxiliary snapshot's many future stations marked ``open``.
Later openings require a dated network rather than extending this manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088


def _codes(prefix, start, end):
    return [f"{prefix}{i}" for i in range(start, end + 1)]


# Explicit operational adjacency: reserved/infill stations are absent, not
# inferred as active merely because their code falls between other stations.
MRT_ROUTES_2021 = {
    "EW": _codes("ew", 1, 33),
    "NS": _codes("ns", 1, 5) + _codes("ns", 7, 28),
    "NE": ["ne1"] + _codes("ne", 3, 17),
    "DT": _codes("dt", 1, 3) + _codes("dt", 5, 35),
    "CC": _codes("cc", 1, 17) + _codes("cc", 19, 29),
    # Dataset aliases: cc34 = Bayfront (CE1); cc33 = Marina Bay (CE2).
    "CC_branch": ["cc4", "cc34", "cc33"],
    "CG_branch": ["ew4", "cg1", "cg2"],
    "TE": ["te1", "te2", "te3"],
}


def build_mrt_2021(raw: pd.DataFrame):
    """Return physical station nodes, rail provenance, and excluded services.

    Apply the source's planned-service exclusion as well as the historical
    allowlist. A planned intermediate service breaks its route connections;
    no bypass is inferred by connecting filtered survivors.
    """
    frame = raw.copy()
    for col in ["CODE", "NAME", "STATUS"]:
        frame[col] = frame[col].astype(str).str.strip().str.lower()
    if frame["CODE"].duplicated().any():
        raise ValueError("station service codes must be unique")
    supported = set().union(*(set(route) for route in MRT_ROUTES_2021.values()))
    include = frame["STATUS"].eq("open") & frame["CODE"].isin(supported)
    active = frame.loc[include].copy()
    if active.empty:
        raise ValueError("no operating MRT services match the January 2021 manifest")
    active["line"] = active["CODE"].str.extract(r"^([a-z]+)")[0].replace({"cg": "ew"})
    nodes = active.groupby("NAME", sort=True).agg(
        LATITUDE=("LATITUDE", "mean"), LONGITUDE=("LONGITUDE", "mean"),
        codes=("CODE", lambda x: ";".join(sorted(x))),
        lines=("line", lambda x: ";".join(sorted(set(x)))),
    ).reset_index()
    nodes["amenity_id"] = np.arange(len(nodes))
    nodes["n_lines"] = nodes["lines"].str.split(";").map(len)
    nodes["is_interchange"] = (nodes["n_lines"] > 1).astype(int)
    for line in ["ns", "ew", "ne", "cc", "dt", "te"]:
        nodes[f"line_{line}"] = nodes["lines"].str.split(";").map(lambda xs: int(line in xs))
    name_to_id = nodes.set_index("NAME")["amenity_id"].to_dict()
    by_code = {row.CODE: name_to_id[row.NAME] for row in active.itertuples()}
    records = []
    for route, sequence in MRT_ROUTES_2021.items():
        for a, b in zip(sequence, sequence[1:]):
            if a in by_code and b in by_code and by_code[a] != by_code[b]:
                src, dst = sorted((by_code[a], by_code[b]))
                records.append((src, dst, route, a, b))
    edges = pd.DataFrame(records, columns=["source", "target", "route", "code_a", "code_b"])
    excluded = frame.loc[~include].copy()
    excluded["exclusion_reason"] = np.where(
        excluded["STATUS"].ne("open"), "not_open_in_source", "outside_operational_2021_manifest"
    )
    return nodes, edges, excluded


def coordinates_radians(frame: pd.DataFrame):
    coords = frame[["LATITUDE", "LONGITUDE"]].to_numpy(float)
    if not np.isfinite(coords).all():
        raise ValueError("graph nodes require finite coordinates")
    if not (((coords[:, 0] >= 1.13) & (coords[:, 0] <= 1.47)) &
            ((coords[:, 1] >= 103.6) & (coords[:, 1] <= 104.1))).all():
        raise ValueError("node coordinates fall outside Singapore bounds")
    return np.radians(coords)


def proximity_edges(blocks: pd.DataFrame, amenities: pd.DataFrame, radius_km: float):
    """One row per block/amenity pair inside radius, with distance in km."""
    if radius_km <= 0 or amenities.empty:
        raise ValueError("a positive radius and nonempty amenity table are required")
    tree = BallTree(coordinates_radians(amenities), metric="haversine")
    indices, distances = tree.query_radius(
        coordinates_radians(blocks), r=radius_km / EARTH_KM,
        return_distance=True, sort_results=True,
    )
    return pd.DataFrame(
        [(b, int(a), float(d * EARTH_KM)) for b, (ids, ds) in enumerate(zip(indices, distances))
         for a, d in zip(ids, ds)], columns=["block_id", "amenity_id", "distance_km"],
    )


def distance_features(blocks: pd.DataFrame, amenities: pd.DataFrame, prefix: str):
    """Target-free tabular control features, independently of graph learning."""
    if amenities.empty:
        raise ValueError("distance summaries require nonempty amenities")
    tree = BallTree(coordinates_radians(amenities), metric="haversine")
    coords = coordinates_radians(blocks)
    distances, _ = tree.query(coords, k=1)
    result = pd.DataFrame({f"{prefix}_nearest_km": distances[:, 0] * EARTH_KM}, index=blocks.index)
    for radius, suffix in [(0.5, "500m"), (1.0, "1km"), (2.0, "2km")]:
        result[f"{prefix}_within_{suffix}"] = tree.query_radius(coords, r=radius / EARTH_KM, count_only=True)
    return result
