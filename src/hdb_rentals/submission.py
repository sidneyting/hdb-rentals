"""Writing competition submissions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def write_submission(
    predictions, *, name: str, n_expected: int = 50_000, index_name: str = "Id"
) -> Path:
    """Write predictions to ``submissions/<timestamp>_<name>.csv``.

    Predictions must be in the row order of ``data/test.csv``; the file is
    written with a 0-based index column, which is the usual Kaggle shape.
    Adjust ``index_name`` if the competition asks for a different header.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    if len(preds) != n_expected:
        raise ValueError(f"expected {n_expected} predictions, got {len(preds)}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN or inf")

    out = pd.DataFrame({index_name: np.arange(len(preds)), config.TARGET: preds})
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SUBMISSIONS_DIR / f"{stamp}_{name}.csv"
    out.to_csv(path, index=False)
    return path
