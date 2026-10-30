# Overall idea

The guiding story for the report. Every section should fit into it; anything that contradicts it gets reworded or flagged.

## The model

**Rent = market level x flat premium.**

- **Market level:** one number per month. Test months are in the future, so the level has to be carried forward. This is the hard part of the task. Forecasting it (the `blend` level) fixes the far-ahead under-prediction on the holdout, but the leaderboard disagrees (S3 439.5 vs S1 436.8 with the frozen level), so the level stays a known limitation.
- **Flat premium:** stable over time (street premiums r = 0.83 quarter to quarter), so it can be learned once from flat attributes and location. The strongest location signal is the block's own past rentals (block premiums, Part 2.3 E1).

## The four steps

### 1. Normal models first, to see how things are

LightGBM on the tabular data (Parts 2.2, 2.3) sets the baseline and shows what carries signal: block premiums help (E1: -4.4 holdout), macro data does not, distance features help a little.

### 2. Graph models, to show how blocks are connected

Blocks are nodes, nearby blocks are joined by edges (Part 2.4). The GNN matches LightGBM on accuracy, but so does the graph-free MLP, so **the graph does not improve prediction**. What it adds is a view of structure:

- for blocks with no past rentals of that flat type, neighbours' premiums matter about as much as the block's own fallback premium (Part 3.3: $63 vs $68);
- once a block has history, its own premium dominates;
- learned edge weights show what the model treats as a useful neighbour.

That cold-start case is rare: 1.44% of holdout rows and 0.77% of test rows have no fit rentals of the same block + flat type (`notebooks/4_followup_experiments.ipynb`, X2). So it shows structure but cannot move RMSE.

Wording to use: "shows how blocks are connected". Wording to avoid: "the graph improves accuracy".

### 3. A blend, because the parts are complementary

**Unverified.** The numbers below compare models on different levels: the GNN uses `damped` (over-predicts on the holdout), E1 LightGBM uses the frozen `MONTH_INDEX`. On the `blend` level E1's bias is +3, not -37, so the "opposite errors" may be the levels. X1 in `notebooks/4_followup_experiments.ipynb` reruns both on the same level; rewrite this step from its result.

- The level model targets the level. LightGBM and the GNN both target the premium.
- They err in opposite directions: bias -37 (LightGBM) vs +19 (GNN). LightGBM is better on blocks with sparse history, the network on blocks with long history.
- GNN + E1 blend (0.8 / 0.2): 480.5 vs 483.1 for E1 alone. A small gain, and the graph-free MLP blends equally well (480.3).

### 4. Interpretability

At equal accuracy, the GNN is the model whose explanations can reach neighbouring blocks: which neighbours, and how much. The MLP cannot pose that question. Same method for both (Integrated Gradients); the GNN's explanations are **richer**, not easier to get.

- The explanations are checked: exactness, fidelity, seed stability, randomisation, and a planted neighbour effect (Part 3.2).
- Caveat: where neighbours are redundant with the block itself, neighbour attributions show how credit is shared, not that the neighbour is needed. They are most informative for blocks without history.
- The MLP is the control that shows neighbour attributions are wired correctly; LightGBM with TreeSHAP is the reference for feature rankings.

## Key numbers (holdout 2024-01 to 2025-03)

| Model | Level | RMSE | Bias | Source |
|---|---|---|---|---|
| E0 LightGBM | `MONTH_INDEX` | 487.5 | | `outputs/gnn_premiums_blend/summary.csv` |
| E1 LightGBM (block premiums) | `MONTH_INDEX` | 483.1 | -37 | same; bias from `outputs/gnn_neighbour_premiums/by_history.csv` |
| E1 LightGBM (block premiums) | `blend` | 481.7 | +3 | `2_models.ipynb` E6; bias from the X1 smoke run (its LightGBM uses all rows; only the neural part is sampled) |
| GNN_uniform + premiums | `damped` | 482.0 | +19 | `outputs/gnn_premiums_blend/summary.csv` |
| MLP_matched + premiums | `damped` | 481.6 | | same |
| GNN + E1 blend | mixed | 480.5 | | same |
| MLP + E1 blend | mixed | 480.3 | | same |

Only compare rows on the same level.

The `smoke/` subfolders are test runs; do not quote them.

## Words

"Blend" means three things in the notebooks. In the report:
- **booster average:** the average of the three tuned boosters (Part 2.2 E3; buys nothing, errors correlate 0.99);
- **`blend` level:** the market-level forecast that averages hold and damped trend (Part 2.3 E6, Part 3.1);
- **tree + GNN blend:** the weighted average of LightGBM and the GNN (Part 2.4).

## Open items

- Run X1 (`notebooks/4_followup_experiments.ipynb`); its "where it goes" notes list the cells to update, including Part 2.4's missing findings cells.
