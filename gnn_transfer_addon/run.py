"""Run the report's whole-subzone or whole-MRT-catchment experiments.

Example: python gnn_transfer_addon/run.py --experiment hierarchy --smoke
Requires the project's data/train.csv and src/hdb_rentals/rent_adjustment.py.
Saved report results are separate from newly generated runs/ outputs.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch  # Import before libraries that may load another OpenMP runtime.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__:
    from .models import HierarchicalRentalModel, StationModel
else:
    from models import HierarchicalRentalModel, StationModel

HERE = Path(__file__).resolve().parent
KEY = ['TOWN', 'BLOCK', 'STREET']
MODELS = {'hierarchy': ['MLP_geo', 'Hierarchy_static'],
          'mrt': ['MLP_station', 'Catchment_only', 'Rail_1', 'Rail_3',
                  'Shuffled_3', 'Geographic_3']}


def load_data(root):
    sys.path.insert(0, str(root / 'src'))
    blocks = pd.read_csv(HERE / 'inputs/block_nodes.csv.gz').sort_values('block_id').reset_index(drop=True)
    assert np.array_equal(blocks.block_id, np.arange(len(blocks)))
    for c in ['MAX_FLOOR', 'YEAR_COMPLETED']:
        blocks[c] = pd.to_numeric(blocks[c], errors='coerce').replace(-1, np.nan)
    rent = pd.read_csv(root / 'data/train.csv')
    for c in KEY:
        rent[c] = rent[c].astype('string').str.strip().str.lower()
        blocks[c] = blocks[c].astype('string').str.strip().str.lower()
    rent['row_id'] = np.arange(len(rent))
    rent['RENT_APPROVAL_DATE'] = pd.to_datetime(rent.RENT_APPROVAL_DATE, format='%Y-%m')
    rent['MONTH_INDEX'] = (rent.RENT_APPROVAL_DATE.dt.year-2021)*12 + rent.RENT_APPROVAL_DATE.dt.month-1
    rent['FLAT_AGE'] = rent.RENT_APPROVAL_DATE.dt.year-rent.LEASE_COMMENCE_DATE
    rent['FLAT_TYPE'] = rent.FLAT_TYPE.str.lower().str.strip().str.replace(r'[\s-]+', '-', regex=True)
    rent['FLAT_MODEL'] = rent.FLAT_MODEL.str.lower().str.strip()
    rent = rent.merge(blocks[KEY+['block_id']], on=KEY, how='left', validate='m:1', sort=False)
    assert rent.block_id.notna().all() and rent.row_id.tolist() == list(range(len(rent)))
    return blocks, rent


def build_graph(blocks, experiment):
    if experiment == 'hierarchy':
        subzones, areas = sorted(blocks.SUBZONE.unique()), sorted(blocks.PLANNING_AREA.unique())
        assert blocks.groupby('SUBZONE').PLANNING_AREA.nunique().eq(1).all()
        mapping = blocks[['SUBZONE', 'PLANNING_AREA']].drop_duplicates().set_index('SUBZONE').loc[subzones]
        return dict(block_sub=torch.tensor(pd.Categorical(blocks.SUBZONE, categories=subzones).codes.astype('int64')),
                    sub_area=torch.tensor(pd.Categorical(mapping.PLANNING_AREA, categories=areas).codes.astype('int64')),
                    n_subzones=len(subzones), n_areas=len(areas))
    stations = pd.read_csv(HERE / 'inputs/mrt_nodes.csv').sort_values('amenity_id').reset_index(drop=True)
    edges = pd.read_csv(HERE / 'inputs/mrt_rail_edges.csv')
    xy = np.radians(stations[['LATITUDE', 'LONGITUDE']].to_numpy())
    distance, ids = BallTree(xy, metric='haversine').query(
        np.radians(blocks[['LATITUDE', 'LONGITUDE']]), k=1)
    ids, distance = ids[:, 0], distance[:, 0]*6371.0088
    n = len(stations)
    rail = np.zeros((n, n), dtype='float32')
    rail[edges.source, edges.target] = rail[edges.target, edges.source] = 1
    _, neighbours = BallTree(xy, metric='haversine').query(xy, k=3)
    geographic = np.zeros_like(rail)
    geographic[np.arange(n)[:, None], neighbours[:, 1:]] = 1
    geographic = np.maximum(geographic, geographic.T)
    perm = np.random.default_rng(731).permutation(n)
    def normalize(a):
        return torch.tensor(a / np.maximum(a.sum(1, keepdims=True), 1))
    membership = torch.tensor(ids, dtype=torch.long)
    return dict(stations=stations, ids=ids, distance=distance, rail=rail,
                membership=membership, n_stations=n,
                counts=torch.bincount(membership, minlength=n).float().clamp_min(1).unsqueeze(1),
                adjacency={'Rail_1': normalize(rail), 'Rail_3': normalize(rail),
                           'Shuffled_3': normalize(rail[perm][:, perm]),
                           'Geographic_3': normalize(geographic)})


def split_data(blocks, rent, graph, experiment):
    """Choose held groups from pre-July-2023 activity, never future rents."""
    date = rent.RENT_APPROVAL_DATE
    observed = blocks[blocks.block_id.isin(rent.loc[date < '2023-07-01', 'block_id'])]
    if experiment == 'hierarchy':
        mapping = observed[['SUBZONE', 'PLANNING_AREA']].drop_duplicates().sort_values('SUBZONE')
        remaining = mapping.groupby('PLANNING_AREA').SUBZONE.nunique().to_dict()
        held = []
        for row in mapping.iloc[np.random.default_rng(17).permutation(len(mapping))].itertuples(index=False):
            if remaining[row.PLANNING_AREA] > 1:
                held.append(row.SUBZONE)
                remaining[row.PLANNING_AREA] -= 1
            if len(held) == int(np.floor(.2*len(mapping))):
                break
        held_blocks = blocks.loc[blocks.SUBZONE.isin(held), 'block_id'].to_numpy()
    else:
        observed_stations = np.unique(graph['ids'][observed.block_id])
        candidates = [int(i) for i in observed_stations if np.any(graph['rail'][i, observed_stations])]
        held = []
        for station in np.random.default_rng(417).permutation(candidates):
            proposed = set(held+[int(station)])
            remaining = np.array([i for i in observed_stations if i not in proposed])
            if all(np.any(graph['rail'][i, remaining]) for i in proposed):
                held.append(int(station))
            if len(held) == int(np.floor(.2*len(candidates))):
                break
        held_blocks = np.flatnonzero(np.isin(graph['ids'], held))
    retained = ~rent.block_id.isin(held_blocks)
    frames = [rent[(date < '2023-07-01') & retained].copy(),
              rent[(date >= '2023-07-01') & (date < '2024-01-01') & retained].copy(),
              rent[(date < '2024-01-01') & retained].copy(),
              rent[(date >= '2024-01-01') & (date < '2024-07-01')].copy()]
    for f in frames[:3]:
        assert not f.block_id.isin(held_blocks).any()
    assert frames[0].RENT_APPROVAL_DATE.max() < frames[1].RENT_APPROVAL_DATE.min()
    assert frames[2].RENT_APPROVAL_DATE.max() < frames[3].RENT_APPROVAL_DATE.min()
    frames[3]['held_out'] = frames[3].block_id.isin(held_blocks)
    return frames, held


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ], sparse_threshold=0)


def prepare(blocks, fit, val, graph, experiment):
    from hdb_rentals.rent_adjustment import MarketRentAdjustment
    spec = json.loads((HERE / 'inputs/feature_spec.json').read_text())
    bp = preprocessor(spec['numeric'], spec['categorical'])
    fit_blocks = np.sort(fit.block_id.unique())
    bp.fit(blocks.iloc[fit_blocks])
    nodes, extra = bp.transform(blocks), {}
    if experiment == 'mrt':
        stations, ids = graph['stations'], graph['ids']
        columns = ['LATITUDE', 'LONGITUDE', 'n_lines', 'is_interchange'] + [c for c in stations if c.startswith('line_')]
        sx = stations[columns].to_numpy(float)
        visible = np.unique(ids[fit.block_id.unique()])
        sx = (sx-sx[visible].mean(0))/np.maximum(sx[visible].std(0), 1e-6)
        access = np.column_stack([sx[ids], graph['distance']])
        # Preserve original row order in these means for numerical parity.
        access_fit = access[fit.block_id.unique()]
        access = (access-access_fit.mean(0))/np.maximum(access_fit.std(0), 1e-6)
        nodes = np.column_stack([nodes, access])
        extra['station'] = torch.tensor(sx.astype('float32'))
    rp = preprocessor(['FLOOR_AREA_SQM', 'FLAT_AGE'], ['FLAT_TYPE', 'FLAT_MODEL'])
    rp.fit(fit)
    adjustment = MarketRentAdjustment('damped').fit(fit)
    scale = adjustment.multipliers(fit.MONTH_INDEX)
    adjusted = fit.MONTHLY_RENT.to_numpy()/scale
    mean, std = float(adjusted.mean()), float(adjusted.std(ddof=1))
    def pack(frame):
        s, y = adjustment.multipliers(frame.MONTH_INDEX), frame.MONTHLY_RENT.to_numpy(float)
        return dict(ids=torch.tensor(frame.block_id.to_numpy(int)),
                    record=torch.tensor(rp.transform(frame).astype('float32')),
                    y=y, scale=s, scaled_y=torch.tensor(((y/s-mean)/std).astype('float32')),
                    weight=torch.tensor((s*s/np.mean(scale*scale)).astype('float32')))
    return dict(nodes=torch.tensor(nodes.astype('float32')), parts=[pack(fit), pack(val)],
                mean=mean, std=std, **extra)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, bundle):
    model.eval()
    val = bundle['parts'][1]
    return np.maximum(model(bundle, val).numpy()*bundle['std']+bundle['mean'], 0)*val['scale']


def fit_model(kind, bundle, graph, experiment, seed, epochs, tune=False):
    seed_all(seed)
    factory = HierarchicalRentalModel if experiment == 'hierarchy' else StationModel
    model = factory(kind, bundle, graph)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.001)
    fit, best, best_epoch, stale, history = bundle['parts'][0], float('inf'), epochs, 0, []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(bundle, fit)-fit['scaled_y']).square()*fit['weight']).mean()
        if experiment == 'hierarchy':
            loss = loss+.1*(model.sub_embedding.square().mean()+model.area_embedding.square().mean())
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()))
        if tune:
            score = float(np.sqrt(np.mean((predict(model, bundle)-bundle['parts'][1]['y'])**2)))
            row['inner_rmse'] = score
            if score < best-.01:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(row)
        if tune and stale >= 35:
            break
    return model, best_epoch, history


def score_predictions(predictions):
    rows = []
    for model, f in predictions.groupby('model'):
        for label, g in [('all', f), ('held_out', f[f.held_out])]:
            error = g.prediction-g.MONTHLY_RENT
            rows.append(dict(model=model, slice=label, n_rows=len(g),
                             rmse=np.sqrt(np.mean(error**2)), mae=error.abs().mean(), bias=error.mean()))
    return pd.DataFrame(rows)


def cluster_errors(ensemble, blocks, graph, experiment):
    """Sufficient statistics for the held-area scores and fixed-prediction bootstrap."""
    f = ensemble[ensemble.held_out].copy()
    groups = blocks.PLANNING_AREA.to_numpy() if experiment == 'hierarchy' else graph['ids']
    f['cluster'] = groups[f.block_id]
    error = f.prediction-f.MONTHLY_RENT
    f['n'], f['sse'], f['sae'], f['error_sum'] = 1, error**2, error.abs(), error
    return f.groupby(['model', 'cluster'], as_index=False)[['n', 'sse', 'sae', 'error_sum']].sum()


def bootstrap(stats, experiment):
    """Resample groups, keeping trained predictions fixed; negative delta improves."""
    model = 'Hierarchy_static' if experiment == 'hierarchy' else 'Rail_3'
    references = ['MLP_geo'] if experiment == 'hierarchy' else [
        'MLP_station', 'Catchment_only', 'Shuffled_3', 'Geographic_3']
    rng, rows = np.random.default_rng(981), []
    for reference in references:
        if not {model, reference} <= set(stats.model):
            continue
        a = stats[stats.model == model].set_index('cluster')
        b = stats[stats.model == reference].set_index('cluster').loc[a.index]
        np.testing.assert_array_equal(a.n, b.n)
        values = np.column_stack([a.n, a.sse, b.sse])
        draws = values[rng.integers(0, len(a), size=(2000, len(a)))].sum(1)
        delta = np.sqrt(draws[:, 1]/draws[:, 0])-np.sqrt(draws[:, 2]/draws[:, 0])
        rows.append(dict(model=model, reference=reference, clusters=len(a),
                         delta=np.sqrt(a.sse.sum()/a.n.sum())-np.sqrt(b.sse.sum()/b.n.sum()),
                         lower=np.quantile(delta, .025), upper=np.quantile(delta, .975)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', choices=['hierarchy', 'mrt', 'both'], default='both')
    ap.add_argument('--root', type=Path, default=HERE.parent)
    ap.add_argument('--output', type=Path, default=HERE / 'runs')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--models', help='Optional comma-separated subset of model names')
    ap.add_argument('--smoke', action='store_true', help='700 rows per split, three epochs; execution check only')
    args = ap.parse_args()
    blocks, rent = load_data(args.root.resolve())
    experiments = ['hierarchy', 'mrt'] if args.experiment == 'both' else [args.experiment]
    for experiment in experiments:
        threads, epochs = (2, 512) if experiment == 'hierarchy' else (4, 640)
        torch.set_num_threads(threads)
        graph = build_graph(blocks, experiment)
        frames, held = split_data(blocks, rent, graph, experiment)
        out = args.output / ('smoke' if args.smoke else 'full') / experiment
        out.mkdir(parents=True, exist_ok=True)
        membership = rent[['row_id', 'block_id', 'RENT_APPROVAL_DATE']].copy()
        for name, f in zip(['inner_fit', 'inner_tune', 'outer_fit', 'outer_compare'], frames):
            membership[name] = membership.row_id.isin(f.row_id)
        membership.to_csv(out / 'membership.csv', index=False)
        kinds = args.models.split(',') if args.models else MODELS[experiment]
        assert set(kinds) <= set(MODELS[experiment])
        config = dict(experiment=experiment, smoke=args.smoke, seeds=args.seeds, models=kinds,
                      held_groups=held, threads=threads, max_epochs=3 if args.smoke else epochs,
                      market='fold-local damped', patience=35, hidden=32,
                      held_labels_in_fit_or_tune=0)
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        if args.smoke:
            frames = [f.sample(min(len(f), 700), random_state=0).sort_index() for f in frames]
        inner = prepare(blocks, *frames[:2], graph, experiment)
        outer = prepare(blocks, *frames[2:], graph, experiment)
        predictions, selections = [], []
        for kind in kinds:
            for seed in map(int, args.seeds.split(',')):
                _, count, history = fit_model(kind, inner, graph, experiment, seed, config['max_epochs'], tune=True)
                fitted, _, _ = fit_model(kind, outer, graph, experiment, seed, count)
                f = frames[3][['row_id', 'block_id', 'MONTHLY_RENT', 'held_out']].copy()
                f['prediction'], f['model'], f['seed'] = predict(fitted, outer), kind, seed
                predictions.append(f)
                selections.append(dict(model=kind, seed=seed, epochs=count, tuning_epochs=len(history),
                                       parameters=sum(p.numel() for p in fitted.parameters() if p.requires_grad)))
                print(experiment, kind, seed, 'epochs', count, flush=True)
        predictions = pd.concat(predictions, ignore_index=True)
        assert not predictions.duplicated(['model', 'seed', 'row_id']).any()
        predictions.to_csv(out / 'predictions.csv.gz', index=False)
        pd.DataFrame(selections).to_csv(out / 'selected_epochs.csv', index=False)
        ensemble = predictions.groupby(['model', 'row_id', 'block_id', 'held_out'], as_index=False).agg(
            MONTHLY_RENT=('MONTHLY_RENT', 'first'), prediction=('prediction', 'mean'))
        scores = score_predictions(ensemble)
        scores.to_csv(out / 'ensemble_metrics.csv', index=False)
        stats = cluster_errors(ensemble, blocks, graph, experiment)
        stats.to_csv(out / 'cluster_errors.csv', index=False)
        bootstrap(stats, experiment).to_csv(out / 'bootstrap.csv', index=False)
        print(scores[scores['slice'] == 'held_out'].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
