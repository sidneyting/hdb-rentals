"""Append-only experiment log, shared through git.

Every run appends one JSON object to ``experiments.jsonl`` at the repo root.
The log is the team's shared record of what has been tried, so the rule is:
if a run produced a score you would quote to a teammate, log it.

Why one line of JSON per run rather than a Markdown table: two people editing
the same table conflict on nearly every merge. Appended lines are handled by
the ``merge=union`` rule in ``.gitattributes``, which keeps both sides'
entries instead of asking anyone to resolve them.

Each entry records the git commit and whether the tree was dirty. A score
without the code that produced it cannot be reproduced or trusted, and on a
team it cannot even be attributed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config

LOG_PATH = config.PROJECT_ROOT / "experiments.jsonl"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_state() -> dict[str, Any]:
    """Commit, branch and cleanliness, so a result can be traced to its code.

    The log file itself is excluded from the dirty check. ``dirty`` is meant to
    answer "could I check this code out again and re-run it", and appending a
    line to the log is not a code change. Without the exclusion the first entry
    of a session records clean and every entry after it records dirty, because
    the log dirtied the tree by writing to itself.
    """
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", f":(exclude){LOG_PATH.name}")),
        "author": _git("config", "user.name") or _git("config", "user.email"),
    }


def log_experiment(
    *,
    name: str,
    score: float,
    validation: str,
    features: Sequence[str],
    model: str,
    params: dict[str, Any] | None = None,
    fold_scores: Sequence[float] | None = None,
    metric: str = "rmse",
    notes: str = "",
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one run to the experiment log and return the entry.

    Args:
        name: Short handle for the run, e.g. ``"lgbm-mrt-distance"``.
        score: The headline number, lower-is-better for RMSE.
        validation: Which split produced it. Scores from different schemes
            are not comparable, so this is required rather than optional --
            see the fold-to-fold spread noted in validation.py.
        features: Columns fed to the model.
        model: Estimator name, e.g. ``"LGBMRegressor"``.
        params: Hyperparameters worth recording.
        fold_scores: Per-fold numbers behind ``score``. Keep these: comparing
            two runs fold by fold cancels fold difficulty and detects far
            smaller differences than comparing the means.
        notes: What you were testing and what you concluded. The field that
            makes the log worth reading in a month.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "name": name,
        "metric": metric,
        "score": round(float(score), 4),
        "fold_scores": [round(float(s), 4) for s in fold_scores] if fold_scores else None,
        "validation": validation,
        "model": model,
        "n_features": len(features),
        "features": sorted(features),
        "params": params or {},
        "notes": notes,
        **_git_state(),
        **(extra or {}),
    }
    target = path or LOG_PATH
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=False) + "\n")
    return entry


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every logged run. Bad lines are skipped rather than raising, so a
    half-written entry from an interrupted run cannot block the whole team."""
    target = path or LOG_PATH
    if not target.exists():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
