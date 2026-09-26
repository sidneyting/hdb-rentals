"""Monthly macro covariates from the COE and stock-price tables.

These two auxiliary tables are unlike the other four. They carry no per-flat
key -- they join on the month alone -- and they are the only inputs in the
project with **observed values inside the test window**: both run through
2026-07, while the labelled data stops at 2025-03.

That matters because of the extrapolation cliff documented in validation.py.
A tree fed ``MONTH_INDEX`` returns its last leaf for every month past training,
so all 50,000 test rows are predicted at the 2025-03 level. A macro series can
carry a time signal across that boundary instead -- but only if its test-window
values fall *inside the range the model saw in training*. A series that climbs
past its own training maximum re-creates the same cliff under a different name,
and is worse than useless because it looks like it should help.

So the loader gates on range, not on correlation. Measured over 2025-04..2026-07
against the 2021-01..2025-03 range:

    coe cat b   1.00 inside      corr with mean monthly rent  +0.84
    coe cat e   1.00                                          +0.82
    coe cat c   0.76                                          +0.91
    coe cat a   0.29                                          +0.94
    stocks      0.00-0.25 for most names                      up to +0.95

Note the inversion: the highest correlations are the least usable. Cat A and
almost every stock spend the test window above their training maximum. The gate
keeps what can actually carry signal forward, and ``macro_coverage`` reports the
fraction for everything so the decision stays visible rather than silent.

One caveat the gate cannot check. Both series and rent rose together through
mid-2023 and flattened together after, so a strong fit may be nothing more than
two trends sharing a period. That is what the plateau-only and placebo checks
are for; see DECISIONS.md. Range is a necessary condition, not a sufficient one.
"""

from __future__ import annotations

import pandas as pd

from . import config

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}

# Equal-weighted groups. Names listed after 2021-02 are excluded everywhere, so
# an index never changes composition mid-window -- a new listing appearing in
# 2024 would otherwise show up as a jump that has nothing to do with the market.
_BANKS = ["DBS Group", "OCBC Bank", "UOB"]
_PROPERTY = [
    "Ascendas Reit", "CapitaLand Ascott Trust", "CapitaLand Investment Limited",
    "CapitaLand Mall Trust", "City Developments",
    "Frasers Logistics & Industrial Trust", "Hotel Properties Limited",
    "Keppel REIT", "Mapletree Commercial Trust", "Mapletree Industrial Trust",
    "Mapletree Logistics Trust", "Pacific Century Regional Developments",
    "Singapore Land", "Suntec REIT", "UOL Group", "Yanlord",
]

_ORIGIN = pd.Period("2021-01", freq="M")


def _month_index(period: pd.PeriodIndex) -> pd.Index:
    return (period.year * 12 + period.month) - (_ORIGIN.year * 12 + _ORIGIN.month)


def _coe_panel() -> pd.DataFrame:
    """Mean quota premium per month per category, one column each.

    Two bidding rounds run each month. They are averaged rather than taking the
    later round: both are realised clearing prices within the month, and the
    mean is the less arbitrary summary. Stated here because the choice is not
    obviously right and changing it changes every number downstream.
    """
    coe = pd.read_csv(config.AUX_CSV["coe"])
    premium = coe["QUOTA_PREMIUM"].replace(r"[\$,]", "", regex=True).astype(float)
    period = pd.PeriodIndex(
        coe["YEAR"].astype(str)
        + "-"
        + coe["MONTH"].str.lower().map(_MONTHS).astype("Int64").astype(str).str.zfill(2),
        freq="M",
    )
    panel = (
        pd.DataFrame({"period": period, "cat": coe["CATEGORY"], "premium": premium})
        .pivot_table(index="period", columns="cat", values="premium", aggfunc="mean")
    )
    panel.columns = [f"COE_{c.upper()}" for c in panel.columns]
    return panel


def _stock_panel() -> pd.DataFrame:
    """Equal-weighted indices rebased to 100 at 2021-01.

    Levels are rebased so names with different share prices contribute equally;
    an unrebased mean is dominated by whichever stock happens to trade highest.
    """
    st = pd.read_csv(config.AUX_CSV["stocks"], parse_dates=["DATE"])
    st["period"] = st["DATE"].dt.to_period("M")
    monthly = st.pivot_table(
        index="period", columns="NAME", values="ADJUSTED_CLOSE", aggfunc="mean"
    )
    full = monthly.columns[monthly.iloc[0].notna() & monthly.notna().all()]
    rebased = monthly[full] / monthly[full].iloc[0] * 100.0

    groups = {
        "STOCK_BANKS": [n for n in _BANKS if n in rebased.columns],
        "STOCK_PROPERTY": [n for n in _PROPERTY if n in rebased.columns],
        "STOCK_BROAD": list(rebased.columns),
    }
    return pd.DataFrame({k: rebased[v].mean(axis=1) for k, v in groups.items()})


def macro_coverage() -> pd.DataFrame:
    """Per-series diagnostic: can this series carry signal past the cliff?

    ``frac_inside`` is the share of test-window months whose value falls within
    the training-window min/max. Anything below 1.0 is partly frozen: for those
    months the tree returns its edge leaf, exactly as it does for MONTH_INDEX.
    """
    panel = _raw_panel()
    train_end = pd.Period(config.TRAIN_PERIOD[1], freq="M")
    intrain = panel[panel.index <= train_end]
    intest = panel[panel.index > train_end]
    rows = []
    for col in panel.columns:
        lo, hi = intrain[col].min(), intrain[col].max()
        post = intest[col].dropna()
        rows.append(
            {
                "series": col,
                "train_min": round(float(lo), 1),
                "train_max": round(float(hi), 1),
                "test_min": round(float(post.min()), 1),
                "test_max": round(float(post.max()), 1),
                "frac_inside": round(float(((post >= lo) & (post <= hi)).mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("frac_inside", ascending=False)


def _raw_panel() -> pd.DataFrame:
    panel = _coe_panel().join(_stock_panel(), how="outer").sort_index()
    first = pd.Period(config.TRAIN_PERIOD[0], freq="M")
    last = pd.Period(config.TEST_PERIOD[1], freq="M")
    panel = panel.loc[first:last]
    expected = pd.period_range(first, last, freq="M")
    missing = expected.difference(panel.index)
    assert len(missing) == 0, f"macro panel has gaps: {list(missing)}"
    assert not panel.isna().any().any(), "macro panel has missing values"
    return panel


def load_macro(*, gate: bool = True, min_inside: float = 1.0) -> pd.DataFrame:
    """Monthly macro panel indexed by ``MONTH_INDEX``, 2021-01 to 2026-07.

    Args:
        gate: Drop series whose test-window values leave the training range.
            On by default -- an ungated series silently reintroduces the
            extrapolation cliff it was meant to solve.
        min_inside: Required share of test months inside the training range.
            1.0 means no month may fall outside.
    """
    panel = _raw_panel()
    if gate:
        cov = macro_coverage().set_index("series")["frac_inside"]
        keep = [c for c in panel.columns if cov[c] >= min_inside]
        assert keep, f"range gate at {min_inside} kept no series"
        panel = panel[keep]
    panel.index = _month_index(panel.index)
    panel.index.name = "MONTH_INDEX"
    return panel


def attach_macro(df: pd.DataFrame, *, gate: bool = True, min_inside: float = 1.0):
    """Left-join the macro panel onto rental rows on ``MONTH_INDEX``.

    Returns ``(frame, columns)``. Like ``attach_block_geodata`` this asserts no
    row loss and no unmatched row, because a silent NaN here would show up as a
    quiet accuracy loss rather than an error.
    """
    panel = load_macro(gate=gate, min_inside=min_inside)
    n_before = len(df)
    merged = df.merge(panel, on="MONTH_INDEX", how="left", validate="many_to_one")
    assert len(merged) == n_before, "macro join changed the row count"
    unmatched = merged[list(panel.columns)].isna().any(axis=1).sum()
    assert unmatched == 0, f"{unmatched} rows fell outside the macro panel"
    return merged, list(panel.columns)
