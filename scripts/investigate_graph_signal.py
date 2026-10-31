"""Leakage-controlled diagnostics for the earlier graph experiments.

Run from the repository root with PYTHONPATH=src. Requires the saved 08/09
artifacts. No neural models are trained; LightGBM baselines are reconstructed
using the same preprocessing and parameters. Outputs go to a separate directory.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hdb_rentals.rent_adjustment import MarketRentAdjustment

OUT = ROOT / 'outputs/10_graph_signal_investigation'
OUT.mkdir(parents=True, exist_ok=True)
PREVIOUS = ROOT / 'outputs/09_graph_block_evaluation'
CONFIG = json.loads((PREVIOUS / 'config.json').read_text())
TARGET = 'MONTHLY_RENT'
KEY = ['TOWN', 'BLOCK', 'STREET']
BLOCK_NUM = CONFIG['block_numeric_features']
BLOCK_CAT = CONFIG['block_categorical_features']
blocks = pd.read_csv(ROOT / 'outputs/08_graph_amenities/block_nodes.csv').sort_values('block_id').reset_index(drop=True)
assert np.array_equal(blocks.block_id, np.arange(len(blocks)))
for c in ['MAX_FLOOR', 'YEAR_COMPLETED']:
    blocks[c] = pd.to_numeric(blocks[c], errors='coerce').replace(-1, np.nan)

rent = pd.read_csv(ROOT / 'data/train.csv')
for c in KEY:
    rent[c] = rent[c].astype('string').str.strip().str.lower()
    blocks[c] = blocks[c].astype('string').str.strip().str.lower()
rent['row_id'] = np.arange(len(rent))
rent['RENT_APPROVAL_DATE'] = pd.to_datetime(rent.RENT_APPROVAL_DATE, format='%Y-%m')
rent['MONTH_INDEX'] = (rent.RENT_APPROVAL_DATE.dt.year - 2021) * 12 + rent.RENT_APPROVAL_DATE.dt.month - 1
rent['FLAT_AGE'] = rent.RENT_APPROVAL_DATE.dt.year - rent.LEASE_COMMENCE_DATE
rent['FLAT_TYPE'] = rent.FLAT_TYPE.str.lower().str.strip().str.replace(r'[\s-]+', '-', regex=True)
rent['FLAT_MODEL'] = rent.FLAT_MODEL.str.lower().str.strip()
rent = rent.merge(blocks[KEY + ['block_id']], on=KEY, how='left', validate='many_to_one', sort=False)
assert rent.row_id.tolist() == list(range(len(rent))) and rent.block_id.notna().all()
N = len(blocks)
edges = pd.read_csv(ROOT / 'outputs/08_graph_amenities/block_edges.csv')
u, v = edges.source.to_numpy(), edges.target.to_numpy()
assert np.all(u < v)


def preprocessor(num, cat):
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
    ], sparse_threshold=0)


def bundle(fit, val, extra_subzone=False):
    adjustment = MarketRentAdjustment('damped').fit(fit)
    bp = preprocessor(BLOCK_NUM, BLOCK_CAT + (['SUBZONE'] if extra_subzone else []))
    rp = preprocessor(['FLOOR_AREA_SQM', 'FLAT_AGE'], ['FLAT_TYPE', 'FLAT_MODEL'])
    bx = bp.fit(blocks.iloc[np.sort(fit.block_id.unique())]).transform(blocks).astype('float32')
    rp.fit(fit)
    def pack(frame):
        scale = adjustment.multipliers(frame.MONTH_INDEX)
        return dict(X=np.concatenate([bx[frame.block_id.to_numpy(int)], rp.transform(frame).astype('float32')], axis=1),
                    y=frame[TARGET].to_numpy(float)/scale, scale=scale, frame=frame)
    a, b = pack(fit), pack(val)
    a['weight'] = a['scale']**2 / np.mean(a['scale']**2)
    return a, b


def model(count):
    return LGBMRegressor(n_estimators=count, learning_rate=.05, num_leaves=31,
                         random_state=0, verbosity=-1, n_jobs=1, metric='None')


def fit_predict(fit, val, count, subzone=False, tune=False):
    a,b = bundle(fit,val,subzone)
    m = model(800 if tune else count)
    options = {}
    if tune:
        def metric(y, pred):
            return 'sgd_rmse', float(np.sqrt(np.mean(((np.maximum(pred,0)-y)*b['scale'])**2))), False
        options = dict(eval_set=[(b['X'],b['y'])], eval_metric=metric,
                       callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
    m.fit(a['X'],a['y'],sample_weight=a['weight'],**options)
    return np.maximum(m.predict(b['X']),0), b, m.best_iteration_ if tune else count


def metrics(y,pred):
    e=np.asarray(pred)-np.asarray(y)
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)))


def source_stats(source):
    # Remove fold-wide monthly forecast bias, leaving spatial/flat-specific errors.
    source=source.copy()
    source['centered_residual']=source.residual-source.groupby('MONTH_INDEX').residual.transform('mean')
    g=source.groupby('block_id').centered_residual.agg(['sum','count'])
    sums=np.zeros(N);counts=np.zeros(N)
    sums[g.index]=g['sum'];counts[g.index]=g['count']
    return sums,counts,source


def correction(sums,counts,kind,tau):
    if kind=='own_block':
        return sums/(counts+tau)
    if kind=='subzone':
        f=pd.DataFrame({'subzone':blocks.SUBZONE,'sums':sums,'counts':counts})
        sg=f.groupby('subzone')[['sums','counts']].transform('sum')
        return sg['sums'].to_numpy()/(sg['counts'].to_numpy()+tau)
    weights = np.ones(len(edges)) if kind=='neighbors' else np.exp(-edges.distance_km.to_numpy()/.3)
    adjacency=csr_matrix((np.r_[weights,weights],(np.r_[u,v],np.r_[v,u])),shape=(N,N))
    # Weighted mean of block residual means, downweighting poorly observed blocks.
    reliability=counts/(counts+tau)
    means=np.divide(sums,counts,out=np.zeros(N),where=counts>0)
    den=adjacency @ reliability
    return np.divide(adjacency @ (reliability*means),den,out=np.zeros(N),where=den>0)


def choose_correction(source, frame, base, scale):
    sums,counts,_=source_stats(source)
    rows=[]
    for kind in ['own_block','subzone','neighbors','distance_neighbors']:
        for tau in [5.,20.]:
            delta=correction(sums,counts,kind,tau)[frame.block_id.to_numpy(int)]
            for strength in [0.,.25,.5,1.]:
                pred=np.maximum(base+strength*delta,0)*scale
                rows.append(dict(kind=kind,tau=tau,strength=strength,**metrics(frame[TARGET],pred)))
    return pd.DataFrame(rows).sort_values(['rmse','strength','tau']).iloc[0],pd.DataFrame(rows)


def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else np.nan


def spatial_diagnostic(source, evaluation, base, scale, experiment):
    sums,counts,source=source_stats(source)
    means=np.divide(sums,counts,out=np.full(N,np.nan),where=counts>0)
    valid=np.isfinite(means[u]) & np.isfinite(means[v])
    a,b=u[valid],v[valid]
    observed=corr(means[a],means[b])
    rng=np.random.default_rng(123)
    ids=np.flatnonzero(np.isfinite(means));permuted=[]
    for _ in range(199):
        shuffled=means.copy();shuffled[ids]=rng.permutation(shuffled[ids])
        permuted.append(corr(shuffled[a],shuffled[b]))
    f=pd.DataFrame({'subzone':blocks.SUBZONE,'mean':means})
    within=means-f.groupby('subzone')['mean'].transform('mean').to_numpy()
    ev=evaluation[['row_id','block_id','MONTH_INDEX',TARGET,'block_slice']].copy()
    ev['residual']=ev[TARGET].to_numpy()/scale-base
    ev['centered']=ev.residual-ev.groupby('MONTH_INDEX').residual.transform('mean')
    em=ev.groupby('block_id').centered.mean()
    common=em.index[np.isfinite(means[em.index])]
    delta=correction(sums,counts,'distance_neighbors',20.)
    rows=[]
    for slice_name,g in [('all',ev)]+list(ev.groupby('block_slice')):
        bm=g.groupby('block_id').centered.mean()
        rows.append(dict(experiment=experiment,slice=slice_name,n_blocks=len(bm),
                         historical_neighbor_vs_future_residual_corr=corr(delta[bm.index],bm.to_numpy())))
    diag=dict(experiment=experiment,n_observed_blocks=len(ids),n_observed_edges=len(a),
              historical_edge_corr=observed,permutation_mean=float(np.mean(permuted)),
              permutation_p=(1+sum(x>=observed for x in permuted))/200,
              within_subzone_edge_corr=corr(within[a],within[b]),
              same_block_history_vs_future_corr=corr(means[common],em.loc[common]),
              same_block_comparison_blocks=len(common))
    return diag,rows


if __name__=='__main__':
    results=[];selected=[];tuning=[];diagnostics=[];predictive=[];predictions=[];source_audits=[]
    saved_counts=pd.read_csv(PREVIOUS/'selected_iterations.csv')
    for experiment in ['temporal','held_out_17','held_out_29','held_out_43']:
        print('START',experiment,flush=True)
        membership=pd.read_csv(PREVIOUS/f'{experiment}_label_membership.csv')
        assert membership.row_id.tolist()==rent.row_id.tolist()
        assert np.array_equal(membership.block_id,rent.block_id)
        frames={name:rent.loc[membership[name]].copy() for name in ['inner_fit','inner_tune','outer_fit','outer_compare']}
        evaluation=frames['outer_compare']
        held=set(pd.read_csv(PREVIOUS/'block_membership.csv').query('experiment == @experiment and held_out').block_id)
        evaluation['block_slice']=np.where(evaluation.block_id.isin(held),'held_out',
                                         np.where(evaluation.block_id.isin(frames['outer_fit'].block_id),'familiar','naturally_unseen'))
        count=int(saved_counts[(saved_counts.experiment==experiment)&saved_counts.model.eq('LightGBM_geo')].selected_iterations.iloc[0])
        # Fixed 250-tree source models: no outer-score or future-period tuning.
        source_parts=[]
        for cutoff,end in [('2022-07-01','2023-01-01'),('2023-01-01','2023-07-01'),('2023-07-01','2024-01-01')]:
            fit=frames['outer_fit'][frames['outer_fit'].RENT_APPROVAL_DATE<pd.Timestamp(cutoff)]
            val=frames['outer_fit'][(frames['outer_fit'].RENT_APPROVAL_DATE>=pd.Timestamp(cutoff)) &
                                    (frames['outer_fit'].RENT_APPROVAL_DATE<pd.Timestamp(end))]
            assert not fit.block_id.isin(held).any() and not val.block_id.isin(held).any()
            assert fit.RENT_APPROVAL_DATE.max()<val.RENT_APPROVAL_DATE.min()
            pred,packed,_=fit_predict(fit,val,250)
            s=val[['row_id','block_id','MONTH_INDEX','RENT_APPROVAL_DATE']].copy()
            s['residual']=packed['y']-pred
            s['source_origin']=cutoff
            source_parts.append(s)
            source_audits.append(dict(experiment=experiment,origin=cutoff,fit_max=str(fit.RENT_APPROVAL_DATE.max()),
                                      source_min=str(val.RENT_APPROVAL_DATE.min()),source_max=str(val.RENT_APPROVAL_DATE.max()),
                                      fit_rows=len(fit),source_rows=len(val),held_rows=0))
            print('OOF source',cutoff,len(val),flush=True)
        source=pd.concat(source_parts,ignore_index=True)
        source.to_csv(OUT/f'{experiment}_historical_oof_residuals.csv',index=False)
        inner_source=source[source.RENT_APPROVAL_DATE<pd.Timestamp(CONFIG['tune_cut'])]
        assert inner_source.RENT_APPROVAL_DATE.max()<frames['inner_tune'].RENT_APPROVAL_DATE.min()
        base_inner,inner_pack,_=fit_predict(frames['inner_fit'],frames['inner_tune'],count)
        choice,grid=choose_correction(inner_source,frames['inner_tune'],base_inner,inner_pack['scale'])
        grid['experiment']=experiment;tuning.append(grid)
        selected.append(dict(experiment=experiment,**choice.to_dict()))
        base,pack,_=fit_predict(frames['outer_fit'],evaluation,count)
        diag,pred_corr=spatial_diagnostic(source,evaluation,base,pack['scale'],experiment)
        diagnostics.append(diag);predictive.extend(pred_corr)
        sums,counts,_=source_stats(source)
        delta=correction(sums,counts,choice['kind'],choice.tau)[evaluation.block_id.to_numpy(int)]
        corrected=np.maximum(base+choice.strength*delta,0)*pack['scale']
        variants={'baseline':base*pack['scale'],'inner_selected_residual_correction':corrected}
        # Also expose each predeclared correction, independently inner-tuned; no outer winner selected.
        for kind in ['own_block','subzone','neighbors','distance_neighbors']:
            ch=grid[grid.kind.eq(kind)].sort_values(['rmse','strength','tau']).iloc[0]
            dd=correction(sums,counts,kind,ch.tau)[evaluation.block_id.to_numpy(int)]
            variants[f'correction_{kind}']=np.maximum(base+ch.strength*dd,0)*pack['scale']
        sub_inner,_,sub_count=fit_predict(frames['inner_fit'],frames['inner_tune'],count,subzone=True,tune=True)
        sub_pred,subpack,_=fit_predict(frames['outer_fit'],evaluation,sub_count,subzone=True)
        variants['baseline_plus_subzone']=sub_pred*subpack['scale']
        # Per-month bias removal uses evaluation labels: diagnostic only, never a model candidate.
        oracle_error=(base*pack['scale']-evaluation[TARGET].to_numpy())
        oracle_bias=pd.Series(oracle_error,index=evaluation.index).groupby(evaluation.MONTH_INDEX).transform('mean').to_numpy()
        variants['ORACLE_monthly_bias_removed']=base*pack['scale']-oracle_bias
        for name,pred in variants.items():
            output=evaluation[['row_id','block_id','MONTH_INDEX','block_slice',TARGET]].copy()
            output['prediction']=pred;output['model']=name;output['experiment']=experiment
            predictions.append(output)
            for slice_name,g in [('all',output)]+list(output.groupby('block_slice')):
                results.append(dict(experiment=experiment,model=name,slice=slice_name,n_rows=len(g),n_blocks=g.block_id.nunique(),
                                    **metrics(g[TARGET],g.prediction)))
        for name,data in [('metrics',results),('selected_corrections',selected),('spatial_diagnostics',diagnostics),
                          ('predictive_correlations',predictive),('oof_source_audit',source_audits)]:
            pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
        pd.concat(tuning,ignore_index=True).to_csv(OUT/'inner_correction_grid.csv',index=False)
        print(pd.DataFrame(results).query('experiment == @experiment and slice == "all"')[['model','rmse','bias']].to_string(index=False),flush=True)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'config.json').write_text(json.dumps(dict(source_model_trees=250,source_origins=['2022-07','2023-01','2023-07'],
        source_period='2022-07 through 2023-12',source_residuals='adjusted units; centered within source month',
        correction_grid=dict(kinds=['own_block','subzone','neighbors','distance_neighbors'],tau=[5,20],strength=[0,.25,.5,1]),
        distance_decay_km=.3,baseline='earlier LightGBM selected tree counts',
        subzone='extra categorical feature; tree count independently inner-tuned',
        outer_warning='previously inspected comparison window, not an untouched final test'),indent=2))
    print('COMPLETE',flush=True)
