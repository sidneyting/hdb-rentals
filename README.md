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

Shared code is in `src/hdb_rentals/` (installed by `uv sync`), GNN experiments in `scripts/`, tests in `tests/`.
`notebooks/archive/` holds the original per-topic notebooks the three above were merged from; they are kept for history and are not meant to be run from there.
