# Geographic transfer experiments

Whole-subzone and whole-MRT-catchment stress tests for the GNNs. Every fitting and tuning rent in entire areas is withheld while their building attributes stay visible, and the models are scored on those areas' January–June 2024 rentals. This is a different setting from the local-block forecasting in the rest of the project, so compare these numbers only with each other.

`notebooks/3_evaluation.ipynb` shows the results (end of the notebook) and `notebooks/2_models.ipynb` has the optional reproduction cell (end of Part 2.4, off by default).

## Files

| File | Purpose |
|---|---|
| `run.py` | Loading, preprocessing, label withholding, training and scoring |
| `models.py` | Hierarchy and MRT models, with their matched controls |
| `inputs/` | Block features, MRT station nodes, rail edges and the feature specification |
| `results/` | Frozen results shown in `3_evaluation.ipynb` |
| `provenance.json` | Hashes of the data, the original source artifacts and these files |

`run.py` needs only `data/train.csv` and `src/hdb_rentals/` from this repo. The graph inputs are bundled here.

## Reproduce

From the repo root:

```bash
# Quick execution check: 700 rows per partition, three epochs, one seed (seconds).
uv run python gnn_transfer_addon/run.py --experiment both --smoke --seeds 0

# Full experiments, two seeds (a few minutes each on a laptop CPU; they can run in parallel).
uv run python gnn_transfer_addon/run.py --experiment hierarchy
uv run python gnn_transfer_addon/run.py --experiment mrt
```

`--models` runs a subset, for example `--models Catchment_only,Rail_3`. `--root` points at another project location.

Outputs go to `gnn_transfer_addon/runs/smoke/` or `runs/full/` (git-ignored) and never overwrite `results/`. Each run writes the split membership, held groups, configuration, predictions, selected epochs, ensemble scores and the cluster bootstrap. Model selection uses the inner split, followed by a fresh fit for the selected number of epochs.

## Setup

| | Hierarchy | MRT catchments |
|---|---|---|
| Hidden labels | 31 entire subzones | 18 entire nearest-station catchments |
| Selection | Seed 17; observed before July 2023; keep an observed sister subzone in each planning area | Seed 417; observed before July 2023; keep an observed adjacent catchment |
| Inner fit / tuning | Before July 2023 / July–December 2023 | Same |
| Outer fit / evaluation | Before January 2024 / January–June 2024 | Same |
| Seeds | 0, 1 | 0, 1 |
| Width / dropout | 32 / 0.1 | 32 / 0.1 |
| Maximum epochs / patience | 512 / 35 | 640 / 35 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.001 | Same |
| Market level / loss | Fold-local damped forecast / dollar-weighted squared error | Same |

Withheld rents are excluded from fitting, preprocessing, the market level and tuning. Known building nodes stay visible, and the models pass building attributes, not rents, through their pooled representations. A block's station is the nearest by straight-line distance, with no radius cutoff (not a walking-time catchment). The station network is frozen at January 2021.

The hierarchy compares a matched MLP with subzone → planning-area pooling. MRT compares the local MLP, catchment pooling without rail, one and three rail steps, shuffled rail connections, and geographic connections (each station linked to its two nearest, symmetrized: 151 edges versus 137 rail edges).

## Results

Held-area RMSE, predictions averaged over the two seeds before scoring:

| Model | RMSE |
|---|---|
| *31 subzones, 3,836 rentals* | |
| Matched MLP (`MLP_geo`) | 532.3 |
| Hierarchical pooling (`Hierarchy_static`) | **525.4** |
| *18 MRT catchments, 4,429 rentals* | |
| Matched MLP (`MLP_station`) | 509.8 |
| Catchment pooling without rail (`Catchment_only`) | 509.9 |
| One rail step (`Rail_1`) | 508.0 |
| Geographic connections (`Geographic_3`) | 506.6 |
| Three rail steps (`Rail_3`) | **505.5** |
| Shuffled rail connections (`Shuffled_3`) | 515.0 |

- The hierarchy beats its MLP by 6.8, but the planning-area bootstrap range (−23.1 to +1.2) includes zero.
- Three rail steps beat catchment pooling alone (−4.4, range −7.6 to −1.3), and shuffling the rail connections makes things worse. Geographic connections recover most of the gain, so this does not show that rail topology itself matters.
- Against the matched MLP, three rail steps are not clearly better (−4.3, range −9.9 to +2.0).
- A fresh full rerun (PyTorch 2.15, 2026-11) gave 525.5 / 532.7 for the hierarchy and kept `Rail_3` best on MRT (506.0). Its lead over `MLP_station` shrank to 1.1, so treat the MRT margins as run-dependent.

These are exploratory backtests: one withheld-area assignment per experiment, on windows that were inspected during development. The bootstrap holds predictions fixed, so it does not include retraining or new area selections. The results suggest broader geographic connections may help where an area has no rental history. They do not show a reliable competition-score gain.

## Results files

- `table.csv`: held-area scores for all eight models.
- `ensemble_metrics.csv`: all-row and held-area scores.
- `seed_metrics.csv`, `selected_epochs.csv`, `held_groups.csv`: individual runs, chosen epoch counts and withheld groups.
- `bootstrap.csv`: cluster bootstrap ranges; negative differences favour the graph model. Hierarchy clusters are planning areas, MRT clusters are station catchments.
- `cluster_errors.csv`: per-group error totals that reconstruct the held-area RMSE, MAE and bias and support the bootstrap.
- `context_metrics.csv`: ordinary-forecasting and random-block results for the same models, where neither approach consistently helps.
- `protocols.json`, `extraction_checks.json`, `smoke_checks.json`: original settings and the checks that this code reproduces the original experiments.

The models were extracted from the original exploratory scripts, which are not part of this repo (`provenance.json` lists their hashes). The checks confirmed identical held-row selections, preprocessing, parameter initialization and forward outputs, with gradients matching within float32 tolerance.
