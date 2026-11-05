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
| `data/` | Kaggle inputs; `cleaned/` (written by notebook 1); `transfer/` (graph inputs for the geographic transfer experiments) |
| `notebooks/` | Everything that runs |
| `outputs/` | Frozen results the report quotes, and the submitted file |

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
| `RERUN_GNN` | 2, Part 2.4 | GNN vs LightGBM, premiums and blends (3 seeds), neighbour premiums | `outputs/gnn_vs_lightgbm/`, `outputs/gnn_premiums_blend/`, `outputs/gnn_neighbour_premiums/` |
| `RUN_EDGE_WEIGHT_EXPERIMENT` | 2, Part 2.4 | Edge-weight comparison (results not used elsewhere) | `outputs/12_weighted_gnns/` (git-ignored) |
| `RUN_GRAPH_TESTS` | 2, Part 2.4 | Unseen blocks, shuffled edges, planted sweep, spatial correlation (about 25 minutes) | `outputs/gnn_graph_tests/` |
| `RUN_TRANSFER_EXPERIMENTS` | 2, end of Part 2.4 | Whole-subzone and MRT-catchment transfer | `outputs/gnn_transfer/runs/` (git-ignored; never the frozen results) |
| `RUN_LEVEL_EVAL` | 3, Appendix D | The report's market-level numbers and figure | `outputs/report_level/` |
| `RETRAIN` | 3, Parts 3.2 and 3.3 | The GNNs being explained (otherwise loaded from `models/`) | `outputs/11_interpretability/`, `outputs/12_interpretability_premiums/` |
| `RUN_TABLE2_EXPLAIN` | 3, end of Part 3.3 | The report's main GNN and its explanations | `outputs/13_interpretability_table2/` |

`outputs/08_graph_amenities/` (the block graph) and `outputs/09_graph_block_evaluation/config.json` (its feature lists)
are inputs that no notebook writes. `outputs/experiments.jsonl` is the experiment log that notebook 1 reads.
The `level-hold` subfolders are the hold-level runs the report's findings cite.

## Submission

`outputs/submission_10_features.csv` is the submitted file (S1, public leaderboard 436.8). `notebooks/2_models.ipynb`
writes it.
