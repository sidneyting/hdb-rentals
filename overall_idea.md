# Overall idea

The guiding story for the report. Every section should fit into it; anything that contradicts it gets reworded or flagged.

## The model

**Rent = market level x flat premium.**

- **Market level:** one number per month. Test months are in the future, so the level has to be carried forward. This is the hard part of the task. Forecasting it (the `blend` level) fixes the far-ahead under-prediction on the holdout, but the leaderboard disagrees (Part 2.3: S3 439.5 vs S1 436.8 with the frozen level; S2, the level alone, scored 439.7), so the level stays a known limitation and the validated submission is Part 2.3's S1, on the frozen level.
- **Flat premium:** stable over time (street premiums r = 0.83 quarter to quarter, Part 1.1 §7.3), so it can be learned once from flat attributes and location. Distance features give the first location gain (Part 1.7); the block's own past rentals (block + flat type premiums, Part 2.3 E1) add the largest gain on top of them.

## The four steps

### 1. Normal models first, to see how things are

LightGBM on the tabular data (Parts 2.2, 2.3) sets the baseline, and shows what carries signal: distance features help most from the base columns (496.4 to 489.6, Part 1.7), block premiums are the largest gain on top of them (Part 2.3 E1: -4.4 holdout, on the `MONTH_INDEX` level; on the `blend` level -1.8 holdout and no gain on tuning, Part 2.4), and macro data does not help (Part 1.8 §4-5, Part 2.1 §7).

### 2. Graph models, to show how blocks are connected

Blocks are nodes, nearby blocks are joined by edges (Part 2.4). The GNN matches LightGBM on accuracy, but so does the graph-free MLP, so **the graph does not improve prediction**. Shuffling the edges at random changes nothing either (482.0 both ways). Nearby blocks do have similar rents (premium correlation 0.69 under 100 m; within a town 0.45, gone by 500 m to 1 km), but a block's own features already carry that. What the graph adds is a way to test this:

- for blocks with no past rentals of that flat type, neighbours' premiums matter about as much as the block's own fallback premium (Part 3.3: $63 vs $68). The MLP with the same premium channel shows this too, so it comes from the premium input, not from message passing;
- once a block has history, its own premium dominates;
- on the Part 3 explanation of the scored model, neighbours share credit with a signal the block already has ($219 own block + neighbours vs $215 for the MLP's own block, on `hold`).

That cold-start case is rare: 1.44% of holdout rows and 0.77% of test rows have no fit rentals of the same block + flat type (Part 2.4, cold-start findings). So it shows structure but cannot move RMSE. On blocks with their rentals hidden (Part 2.4, unseen blocks), neighbours' premiums gain about 0.2% (clear for the MLP only), and message passing adds nothing. The neural models do beat LightGBM there by about 19, with or without a graph.

Wording to use: "tests whether the connections matter". Wording to avoid: "the graph improves accuracy", "reveals the spatial relationships of HDB rentals".

### 3. A blend, a small gain from averaging

Checked on one market level: Part 2.4's same-level run has LightGBM and the neural models learn rent / the `blend` level, over three seeds, so they differ only in the premium model.

- The level model targets the level. LightGBM and the GNN both target the premium.
- GNN + E1 blend: 480.3 vs 481.7 for E1 alone, with weights of 0.50 to 0.55 picked on the tuning split. The gain is above seed noise (twice the seed sd is 0.57) and shows on the tuning split too (593.95 to 592.6), but it is only 0.3%. It has no leaderboard score, and the `blend` level lost there.
- The graph-free MLP blends equally well (480.2), so the gain comes from averaging a tree with a neural model, not from the graph.

Wording to use: "averaging a tree and a neural model gives a small gain". Wording to avoid: "they err in opposite directions" as the reason. Holdout biases do differ in sign (+3 vs -9), but the earlier -37 vs +18 was mostly the levels, and both are negative on the tuning split, where the blend also gains. The gain is what averaging two equally accurate models with error correlation 0.988 gives. "LightGBM is better on sparse history, the network on long history" also does not hold on one level: LightGBM is ahead for 1 to 10 rentals, the GNN at 0 and above 10, all by about 2 or less.

### 4. Interpretability

The report compares and explains the neural models on the `hold` level, the closest to the submitted month-index model (`gnn_premiums_blend.py --level hold --same-level`, `gnn_graph_tests.py --level hold`, `explain_table2_gnn.py --level hold`, results under `level-hold`). The `blend` runs remain as a robustness check. On `hold`: E1 482.6, GNN 483.6, MLP 483.5, shuffled 483.5, blends 481.6 and 481.5; no neighbour rule clears, so "neighbours' premiums help unseen blocks" holds on `blend` only. Parts 3.2 and 3.3 explain siblings; use them for the planted-effect check of the method and the neighbour-premium side test.

At equal accuracy (with premiums, Part 2.4; without them the GNN is about 5 behind its LightGBM reference, Part 3.2), the GNN is the model whose explanations can reach neighbouring blocks: how much credit neighbours share. The MLP explains features and location equally well; only the GNN can split credit among neighbouring blocks' features (never their rents in the scored model; ranking of single neighbours is weak, $8.2 vs $5.9 for a random one). Same method for both (Integrated Gradients); the GNN's explanations are **richer**, not easier to get.

- The explanations are checked: exactness, fidelity, seed stability, randomisation, and a planted neighbour effect (Part 3.2).
- Caveat: where neighbours are redundant with the block itself, neighbour attributions show how credit is shared, not that the neighbour is needed. They are most informative for blocks without history.
- The MLP is the control that shows neighbour attributions are wired correctly; LightGBM with TreeSHAP is the reference for feature rankings.
- The submitted model (Part 2.3 S1) is explained with TreeSHAP in Part 2.3: time 46%, flat attributes 31%, block premiums 12%, distance 6%, town 5%. Its time contribution stops changing after the last training month, which shows the frozen level directly.

## Key numbers (holdout 2024-01 to 2025-03)

| Model | Level | RMSE | Bias | Source |
|---|---|---|---|---|
| E0 LightGBM | `MONTH_INDEX` | 487.5 | | `outputs/gnn_premiums_blend/summary.csv` |
| E1 LightGBM (block premiums) | `MONTH_INDEX` | 483.1 | -37 | same; bias from `outputs/gnn_neighbour_premiums/by_history.csv` |
| E1 LightGBM (block premiums) | `blend` | 481.7 | +3 | `2_models.ipynb` E6; bias from the Part 2.4 same-level run's `results.csv` |
| GNN_uniform + premiums | `damped` | 482.0 | +18 | `outputs/gnn_premiums_blend/summary.csv`; bias from `results.csv` |
| MLP_matched + premiums | `damped` | 481.5 | | same |
| GNN + E1 blend | mixed | 480.5 | | same |
| MLP + E1 blend | mixed | 480.2 | | same |
| GNN_uniform + premiums | `blend` | 482.0 | -9 | `outputs/gnn_premiums_blend/level-blend_same_seed*/results.csv` (mean of 3 seeds, sd 0.29) |
| MLP_matched + premiums | `blend` | 481.8 | -9 | same (sd 0.17) |
| GNN + E1 blend | `blend` | 480.3 | -3 | same |
| MLP + E1 blend | `blend` | 480.2 | -3 | same |
| GNN_uniform + premiums, shuffled edges | `blend` | 482.0 | -8 | `outputs/gnn_graph_tests/shuffled_summary.csv` (mean of 3 seeds, sd 0.39) |

Only compare rows on the same level.

The `smoke/` subfolders are test runs; do not quote them. Neural numbers move by up to about 0.2 between runs, so quote them to the nearest whole number or with the seed sd.

## Words

"Blend" means three things in the notebooks. In the report:
- **booster average:** the tuned-weight average of the three tuned boosters (Part 2.2 E3; buys nothing, errors correlate 0.99);
- **`blend` level:** the market-level forecast that averages hold and damped trend (Part 2.3 E6, Part 3.1);
- **tree + neural blend:** the weighted average of LightGBM and the GNN or MLP (Part 2.4).
