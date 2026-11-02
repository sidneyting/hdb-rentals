# hdb-rentals

Rent prediction for Singapore HDB flats (CS5228 final project).

## Setup

```
uv sync
```

Put the Kaggle files in `data/` (`train.csv`, `test.csv`, `auxiliary/`).

## Run order

Run the notebooks in `notebooks/` top to bottom, in this order. Each one maps to a report section.

| Notebook | Report section | Contents |
|---|---|---|
| `1_eda_preprocessing.ipynb` | EDA & Preprocessing | EDA of train and auxiliary data, cleaning (writes `cleaned/`), feature checks, key findings |
| `2_models.ipynb` | Data Mining Methods | Baseline pipeline, model families and tuning, block premiums and final LightGBM, GNNs (runs `scripts/*gnn*.py`) |
| `3_evaluation.ipynb` | Evaluation & Interpretation | Market-level robustness, GNN interpretability, neighbour premiums |

`overall_idea.md` is the guiding story for the report.

`outputs/08_graph_amenities/`, `outputs/09_graph_block_evaluation/` and the saved models under `outputs/11_interpretability/` and `outputs/12_interpretability_premiums/` are committed inputs: Part 2.4 reads the first two, and Parts 3.2 and 3.3 load the saved models unless `RETRAIN = True`. `outputs/report_level/`, written by `scripts/report_level_eval.py`, is also committed: Part 3.1 Appendix D reads it for the report's market-level numbers. `outputs/13_interpretability_table2/`, written by `scripts/explain_table2_gnn.py`, is committed too: the end of Part 3.3 reads it.

Shared code is in `src/hdb_rentals/` (installed by `uv sync`), GNN experiments in `scripts/`, tests in `tests/`.

## Submission

`outputs/submission_10_features.csv` is the submitted file (S1, public leaderboard 436.8). `notebooks/2_models.ipynb` writes it.

## GNN scripts

`notebooks/2_models.ipynb` runs these in Part 2.4, so you only need these commands to run them on their own. Run them from the repo root, in this order, after notebook 1. Run times are given only where the notebook records them.

| Order | Command | Run time |
|---|---|---|
| 1 | `uv run python scripts/compare_gnn_lightgbm.py` | Not recorded |
| 2 | `uv run python scripts/gnn_premiums_blend.py` | Not recorded |
| 3 | `uv run python scripts/gnn_premiums_blend.py --level blend --same-level --premiums row --seed <0, 1, 2>`, once per seed | A few minutes per seed on a laptop CPU |
| 4 | `uv run python scripts/gnn_neighbour_premiums.py` | Not recorded |
| 5 (optional) | `uv run python scripts/gnn_graph_tests.py <unseen, shuffled, planted> --seed <0, 1, 2>`, for each test and seed, then `spatial`, then `summary` | About 25 minutes on a laptop CPU in total |

Step 3 writes the `predictions.csv` files that Part 2.4 reads. They are committed, so the notebook also works if you skip it. Part 2.4 reads saved graph test results unless `RUN_GRAPH_TESTS = True`.

## Tests

```
uv run --with pytest pytest
```

`pytest` is not a project dependency, so this installs it for that one run.

## Report

The LaTeX source is in `report_final/`, with `cs5228.tex` as the main file.
