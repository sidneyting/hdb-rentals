"""Loading and normalisation for the HDB rental data.

The raw CSVs carry three quirks that will silently cost accuracy if they are
not handled at load time:

1. ``FLAT_TYPE`` is encoded two ways in train -- "4-room" (40,688 rows) and
   "4 room" (13,391 rows) are the same category. Test uses only the hyphenated
   form. Left alone, a model treats them as unrelated levels and the "4 room"
   rows teach it nothing usable at prediction time.
2. ``STREET`` is mixed case in both files ("Whampoa Drive" vs "whampoa drive"),
   which breaks the join onto the block geodata.
3. ``FEE`` and ``FURNISHED`` are constant, so they are dropped.

``load_train`` and ``load_test`` apply all of this; use them rather than
reading the CSVs directly.
"""

from __future__ import annotations

import pandas as pd

from . import config

_TEXT_COLUMNS = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED"]

# "4 room" -> "4-room". Applied after lowercasing.
_FLAT_TYPE_CANONICAL = {f"{n} room": f"{n}-room" for n in range(1, 6)}


def _normalise_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    if "FLAT_TYPE" in df.columns:
        df["FLAT_TYPE"] = df["FLAT_TYPE"].replace(_FLAT_TYPE_CANONICAL)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand RENT_APPROVAL_DATE ("2021-01") into usable time columns.

    ``MONTH_INDEX`` counts months from 2021-01 and is the column the
    time-ordered splits in validation.py sort on.
    """
    period = pd.PeriodIndex(df["RENT_APPROVAL_DATE"], freq="M")
    origin = pd.Period("2021-01", freq="M")
    df["APPROVAL_PERIOD"] = period
    df["YEAR"] = period.year
    df["MONTH"] = period.month
    df["MONTH_INDEX"] = (
        period.year * 12 + period.month - (origin.year * 12 + origin.month)
    )
    # Age of the flat's lease at the time of the rental approval, in years.
    df["FLAT_AGE"] = df["YEAR"] - df["LEASE_COMMENCE_DATE"]
    return df


def _load_split(path, *, drop_constant: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalise_text(df)
    df = _add_time_features(df)
    if drop_constant:
        df = df.drop(columns=[c for c in config.CONSTANT_COLUMNS if c in df.columns])
    return df


def load_train(*, drop_constant: bool = True) -> pd.DataFrame:
    """Training rows, normalised, sorted oldest first."""
    df = _load_split(config.TRAIN_CSV, drop_constant=drop_constant)
    return df.sort_values("MONTH_INDEX", kind="stable").reset_index(drop=True)


def load_test(*, drop_constant: bool = True) -> pd.DataFrame:
    """Test rows in their original file order.

    Order is preserved because submissions are matched by row position.
    """
    return _load_split(config.TEST_CSV, drop_constant=drop_constant)


def load_auxiliary(name: str) -> pd.DataFrame:
    """Load one auxiliary table by key, e.g. ``load_auxiliary("mrt")``."""
    if name not in config.AUX_CSV:
        raise KeyError(
            f"unknown auxiliary table {name!r}; "
            f"expected one of {sorted(config.AUX_CSV)}"
        )
    df = pd.read_csv(config.AUX_CSV[name])
    return _normalise_text(df)


def attach_block_geodata(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join latitude/longitude and block attributes onto rental rows.

    The join key is unique in sg-hdb-block.csv and covers 100% of both splits,
    so this neither drops nor duplicates rows -- both are asserted.
    """
    blocks = load_auxiliary("hdb_block")
    n_before = len(df)
    merged = df.merge(blocks, on=config.BLOCK_KEY, how="left", validate="many_to_one")
    assert len(merged) == n_before, "block join changed the row count"
    missing = merged["LATITUDE"].isna().sum()
    assert missing == 0, f"{missing} rows failed to match a block"
    return merged
