"""Repeatable block holdouts combined with chronological rental evaluation."""

import numpy as np
import pandas as pd


def block_time_partitions(rentals, tune_cut, outer_cut, *, fraction=0.2, seed=17):
    """Choose blocks from inner-fit history; exclude all their labels from fitting.

    A zero fraction gives the ordinary temporal control. Selection uses block IDs
    from before tune_cut only, never future activity or rental prices. Auxiliary
    graph nodes can remain visible. Returned frames preserve input row order.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    tune_cut, outer_cut = pd.Timestamp(tune_cut), pd.Timestamp(outer_cut)
    if tune_cut >= outer_cut:
        raise ValueError("tune_cut must precede outer_cut")
    dates = pd.to_datetime(rentals["RENT_APPROVAL_DATE"])
    ids = rentals["block_id"]
    if ids.isna().any() or dates.isna().any():
        raise ValueError("Block IDs and dates must be complete")
    eligible = np.sort(ids[dates < tune_cut].unique())
    count = max(1, int(np.floor(len(eligible) * fraction))) if fraction else 0
    if fraction and count >= len(eligible):
        raise ValueError("Need at least two eligible blocks for a block holdout")
    held_ids = np.sort(np.random.default_rng(seed).choice(eligible, count, replace=False))
    retained = ~ids.isin(held_ids)
    frames = {
        "inner_fit": rentals[(dates < tune_cut) & retained].copy(),
        "inner_tune": rentals[(dates >= tune_cut) & (dates < outer_cut) & retained].copy(),
        "outer_fit": rentals[(dates < outer_cut) & retained].copy(),
        "outer_compare": rentals[dates >= outer_cut].copy(),
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Every temporal partition must contain rental rows")
    evaluation = frames["outer_compare"]
    fit_ids = set(frames["outer_fit"]["block_id"])
    evaluation["block_slice"] = np.where(
        evaluation["block_id"].isin(held_ids), "held_out",
        np.where(evaluation["block_id"].isin(fit_ids), "familiar", "naturally_unseen"),
    )
    return {"held_block_ids": held_ids, **frames}
