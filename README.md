# hdb-rentals

Rent prediction for Singapore HDB flats (CS5228 final project).

## Setup

```
uv sync
```

Put the Kaggle files in `data/` (`train.csv`, `test.csv`, `auxiliary/`).

## Layout

| Folder | Contents |
|---|---|
| `data/` | Kaggle inputs; `cleaned/` (written by notebook 1); `graph/` (the block graph and its feature lists); `transfer/` (graph inputs for the geographic transfer experiments) |
| `notebooks/` | Everything that runs |
| `outputs/` | The submitted file, the experiment log, and the frozen results the report quotes: `gnn/`, `interpretability/`, `market_level/`. `intermediate/` holds scratch files the notebooks regenerate (git-ignored) |

## Run order

Run the notebooks in `notebooks/` top to bottom, in this order.

| Notebook | Contents |
|---|---|
| `0_shared_code.ipynb` | Code the other three share (data loading, market level, GNNs, explainer, experiment runs). Each of them loads it with `%run`, so you never run it yourself |
| `1_eda_preprocessing.ipynb` | EDA of train and auxiliary data, cleaning (writes `data/cleaned/`), feature checks, key findings |
| `2_models.ipynb` | Baseline pipeline, model families and tuning, block premiums and final LightGBM, GNNs |
| `3_evaluation.ipynb` | Market-level robustness, GNN interpretability, neighbour premiums, geographic transfer (whole subzones and MRT catchments) |

## Frozen results

The GNN experiments take hours on a laptop CPU, and neural training is not exactly repeatable, so the notebooks read
their saved results from `outputs/` by default instead of retraining. These flags retrain, and overwrite what they
retrain:

| Flag | Notebook | Retrains | Writes |
|---|---|---|---|
| `RERUN_GNN` | 2, Part 2.4 | GNN vs LightGBM, premiums and blends (3 seeds), neighbour premiums | `outputs/gnn/vs_lightgbm/`, `outputs/gnn/premiums_blend/`, `outputs/gnn/neighbour_premiums/` |
| `RUN_EDGE_WEIGHT_EXPERIMENT` | 2, Part 2.4 | Edge-weight comparison (results not used elsewhere) | `outputs/gnn/edge_weights/` (git-ignored) |
| `RUN_GRAPH_TESTS` | 2, Part 2.4 | Unseen blocks, shuffled edges, planted sweep, spatial correlation (about 25 minutes) | `outputs/gnn/graph_tests/` |
| `RUN_TRANSFER_EXPERIMENTS` | 2, end of Part 2.4 | Whole-subzone and MRT-catchment transfer | `outputs/gnn/transfer/runs/` (git-ignored; never the frozen results) |
| `RUN_LEVEL_EVAL` | 3, Appendix D | The report's market-level numbers and figure | `outputs/market_level/` |
| `RETRAIN` | 3, Parts 3.2 and 3.3 | The GNNs being explained (otherwise loaded from `models/`) | `outputs/interpretability/gnn/`, `outputs/interpretability/premiums/` |
| `RUN_TABLE2_EXPLAIN` | 3, end of Part 3.3 | The report's main GNN and its explanations | `outputs/interpretability/table2/` |

`data/graph/` (the block graph and its feature lists) is an input that no notebook writes. `outputs/experiments.jsonl`
is the experiment log that notebook 1 reads.
The `level-hold` subfolders are the hold-level runs the report's findings cite.

## Submission

`outputs/submission_10_features.csv` is the submitted file (S1, public leaderboard 436.8). `notebooks/2_models.ipynb`
writes it.
