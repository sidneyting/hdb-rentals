"""Fold-local rent adjustment shared by tabular and graph models.

Only ``fit`` reads rent labels. ``multipliers`` accepts month indices, so
forecasting validation/test scales cannot depend on their rental targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET
from .market_level import forecast_level, hedonic_level


class MarketRentAdjustment:
    """Express rents in the first fitted month's market, then restore SGD.

    ``method`` controls extrapolation of the fitted log market effects.
    Months already fitted use their observed effects, including test records
    in the final labeled month. Damped forecasts require contiguous history.
    """

    def __init__(self, method: str = "damped"):
        if method not in {"hold", "damped", "blend", "linear"}:
            raise ValueError("method must be hold, damped, blend, or linear; oracle is not a training policy")
        self.method = method

    @staticmethod
    def _months(months) -> np.ndarray:
        values = np.asarray(months)
        if values.ndim != 1 or not np.isfinite(values.astype(float)).all():
            raise ValueError("month indices must be a finite one-dimensional array")
        if not np.equal(values, values.astype(int)).all():
            raise ValueError("month indices must be integers")
        return values.astype(int)

    def fit(self, records: pd.DataFrame) -> "MarketRentAdjustment":
        if records.empty:
            raise ValueError("cannot fit a market index to empty records")
        months = self._months(records["MONTH_INDEX"])
        rents = records[TARGET].to_numpy(float)
        if not np.isfinite(rents).all() or (rents <= 0).any():
            raise ValueError("hedonic fitting requires finite positive rents")
        unique = np.unique(months)
        if self.method in ("damped", "blend") and (len(unique) < 3 or np.any(np.diff(unique) != 1)):
            raise ValueError("damped forecasting requires at least three contiguous training months")
        # Reset arbitrary/nonunique external indices before dummy construction.
        fit = records.reset_index(drop=True)
        self.log_level_ = hedonic_level(fit, np.arange(len(fit)))
        if not np.isfinite(self.log_level_.to_numpy()).all():
            raise ValueError("hedonic model produced non-finite month effects")
        self.first_month_ = int(unique[0])
        self.last_month_ = int(unique[-1])
        return self

    def multipliers(self, months) -> np.ndarray:
        """Return positive market factors in input row order, without labels."""
        if not hasattr(self, "log_level_"):
            raise RuntimeError("fit the market adjustment before transforming months")
        months = self._months(months)
        if not len(months):
            return np.empty(0, dtype=float)
        level = self.log_level_
        if months.max() > self.last_month_:
            # forecast_level counts supplied future entries as steps. Supply
            # every intervening month so sparse/repeated requests stay aligned
            # to calendar horizons, not to the number of requested months.
            future = np.arange(self.last_month_ + 1, int(months.max()) + 1)
            level = forecast_level(level, future, self.method)
        log_scale = level.reindex(months).to_numpy(float)
        if not np.isfinite(log_scale).all():
            raise ValueError("requested historical month was absent from the fitted index")
        scale = np.exp(log_scale)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("market forecast produced invalid multipliers")
        return scale

    def adjusted_rents(self, records: pd.DataFrame) -> np.ndarray:
        """Adjust supplied labels using the already fitted market policy."""
        return records[TARGET].to_numpy(float) / self.multipliers(records["MONTH_INDEX"])

    def restore(self, adjusted_predictions, months) -> np.ndarray:
        predictions = np.asarray(adjusted_predictions, dtype=float)
        scale = self.multipliers(months)
        if predictions.shape != scale.shape:
            raise ValueError("predictions and months must have matching one-dimensional shapes")
        return predictions * scale
