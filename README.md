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
