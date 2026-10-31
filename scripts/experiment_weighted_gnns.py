"""Controlled GNN experiments with explicit and learned geographic edge weights.

PYTHONPATH=src python scripts/experiment_weighted_gnns.py
Set HDB_WEIGHTED_GNN_SMOKE=1 for a separate short execution check.
"""
# Import torch first to avoid mixed OpenMP import-order crashes.
import torch
from torch import nn
from pathlib import Path
import os,json,gc
import numpy as np
import pandas as pd
from investigate_graph_signal import ROOT,blocks,rent,edges,N,BLOCK_NUM,BLOCK_CAT,preprocessor,metrics
from hdb_rentals.rent_adjustment import MarketRentAdjustment
from hdb_rentals.block_evaluation import block_time_partitions
from hdb_rentals import gnn

SMOKE=os.environ.get('HDB_WEIGHTED_GNN_SMOKE','0')=='1'
OUT=Path(os.environ['HDB_WEIGHTED_GNN_OUTPUT']) if 'HDB_WEIGHTED_GNN_OUTPUT' in os.environ else ROOT/'outputs'/('12_weighted_gnns_smoke' if SMOKE else '12_weighted_gnns')
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[0] if SMOKE else [0,1]
KINDS=['MLP_matched','GNN_uniform','GNN_distance','GNN_similarity','GNN_learned_edges','GNN_multiscale']
if os.environ.get('HDB_WEIGHTED_GNN_KINDS'):
    requested=os.environ['HDB_WEIGHTED_GNN_KINDS'].split(',')
    assert set(requested)<=set(KINDS)
    KINDS=requested
MAX_EPOCHS=3 if SMOKE else int(os.environ.get('HDB_WEIGHTED_GNN_MAX_EPOCHS','320'))
PATIENCE=2 if SMOKE else 30
HIDDEN=64
DEVICE=torch.device('cpu')
torch.set_num_threads(int(os.environ.get('HDB_WEIGHTED_GNN_THREADS','4')))

# Graph tensors and the model live in hdb_rentals.gnn; these names stay for other scripts.
GRAPH=gnn.build_graph(blocks,edges)
EDGE_INDEX=GRAPH.edge_index;EDGE_FEATURES=GRAPH.edge_features
FAR_ADJACENCY=GRAPH.far_adjacency;FAR_PRESENT=GRAPH.far_present
WeightedRentalGNN=gnn.WeightedRentalGNN;predict=gnn.predict;seed_everything=gnn.seed_everything


def prepare(fit,*others):
    adjustment=MarketRentAdjustment('damped').fit(fit)
    bp=preprocessor(BLOCK_NUM,BLOCK_CAT+['SUBZONE'])
    bp.fit(blocks.iloc[np.sort(fit.block_id.unique())])
    node_x=torch.tensor(bp.transform(blocks).astype('float32'))
    rp=preprocessor(['FLOOR_AREA_SQM','FLAT_AGE'],['FLAT_TYPE','FLAT_MODEL']);rp.fit(fit)
    fit_scale=adjustment.multipliers(fit.MONTH_INDEX)
    adjusted=fit.MONTHLY_RENT.to_numpy()/fit_scale
    mean,std=float(adjusted.mean()),float(adjusted.std(ddof=1))
    normalizer=np.mean(fit_scale**2)
    def pack(frame):
        scale=adjustment.multipliers(frame.MONTH_INDEX)
        y=frame.MONTHLY_RENT.to_numpy(float)
        return dict(frame=frame,ids=torch.tensor(frame.block_id.to_numpy(int)),
            record=torch.tensor(rp.transform(frame).astype('float32')),
            y=y,scale=scale,scaled_y=torch.tensor(((y/scale-mean)/std).astype('float32')),
            weight=torch.tensor((scale**2/normalizer).astype('float32')))
    return dict(nodes=node_x,parts=[pack(f) for f in (fit,*others)],mean=mean,std=std)


def train(kind,bundle,seed,epochs,tune=False,save_to=None,**metadata):
    model,best_epoch,history=gnn.train(kind,bundle,GRAPH,seed,epochs,tune=tune,patience=PATIENCE,hidden=HIDDEN)
    if save_to is not None:
        gnn.save_model(model,save_to,kind=kind,seed=seed,epochs=epochs,best_epoch=best_epoch,**metadata)
    return model,best_epoch,history


def main():
    experiments=[('2023-01-01',0),('2024-01-01',0),('2025-01-01',0),('2024-01-01',17)]
    if os.environ.get('HDB_WEIGHTED_GNN_ORIGIN'):
        experiments=[job for job in experiments if job[0]==os.environ['HDB_WEIGHTED_GNN_ORIGIN'] and job[1]==0]
    if os.environ.get('HDB_WEIGHTED_GNN_SEEDS'):
        SEEDS[:]=[int(seed) for seed in os.environ['HDB_WEIGHTED_GNN_SEEDS'].split(',')]
    if SMOKE:experiments=[('2024-01-01',17)]
    config=dict(smoke=SMOKE,kinds=KINDS,seeds=SEEDS,experiments=experiments,max_epochs=MAX_EPOCHS,
        patience=PATIENCE,hidden=HIDDEN,dropout=.2,learning_rate=.001,gate_initial_bias=-2.,
        distance_decay_km=.3,similarity_age_scale_years=15,similarity_height_scale_floors=8,
        multiscale_far_band_km=[.25,1.5],multiscale_far_neighbors=8,multiscale_far_decay_km=.75,
        edge_features=['distance_km','completion_year_gap/20','max_floor_gap/10','same_subzone','same_town','nearest_mrt_distance_gap_km'],
        block_categories=BLOCK_CAT+['SUBZONE'],holdout_fraction=.2,holdout_seed=17,
        market_policy='damped',objective='SGD MSE via multiplier squared weights',
        comparison='Same active self/message/gate/head architecture; MLP uses identity adjacency',
        learned_weights='positive distance decay times learned sigmoid kernel, normalized per receiving node',
        limitations='Exploratory comparisons on previously inspected labels; metadata snapshot; fixed historical MRT policy; no actual walking distances',
        torch_version=torch.__version__,threads=torch.get_num_threads())
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    results=[];predictions=[];selections=[];histories=[];audits=[]
    for origin,split_seed in experiments:
        cutoff=pd.Timestamp(origin);inner_cut=cutoff-pd.DateOffset(months=6)
        end=min(cutoff+pd.DateOffset(months=6),rent.RENT_APPROVAL_DATE.max()+pd.DateOffset(months=1))
        partition=block_time_partitions(rent,inner_cut,cutoff,fraction=.2 if split_seed else 0,seed=split_seed)
        fit=partition['outer_fit'];val=partition['outer_compare']
        val=val[val.RENT_APPROVAL_DATE<end].copy()
        inner_fit=partition['inner_fit'];inner_val=partition['inner_tune']
        held=set(partition['held_block_ids'])
        for frame in [fit,inner_fit,inner_val]:assert not frame.block_id.isin(held).any()
        if SMOKE:
            fit=fit.sample(min(3500,len(fit)),random_state=0).sort_index()
            inner_fit=inner_fit.sample(min(2500,len(inner_fit)),random_state=0).sort_index()
            inner_val=inner_val.sample(min(800,len(inner_val)),random_state=0).sort_index()
            val=pd.concat([g.sample(min(300,len(g)),random_state=0) for _,g in val.groupby('block_slice')]).sort_index()
        experiment=f'{cutoff:%Y%m}_'+('temporal' if not split_seed else f'held_{split_seed}')
        print('EXPERIMENT',experiment,'fit',len(fit),'evaluate',len(val),flush=True)
        pd.DataFrame({'block_id':np.arange(N),'held_out':np.isin(np.arange(N),list(held))}).to_csv(OUT/f'{experiment}_block_membership.csv',index=False)
        inner=prepare(inner_fit,inner_val);outer=prepare(fit,val)
        audits.append(dict(experiment=experiment,inner_fit_max=str(inner_fit.RENT_APPROVAL_DATE.max()),
            inner_val_min=str(inner_val.RENT_APPROVAL_DATE.min()),outer_fit_max=str(fit.RENT_APPROVAL_DATE.max()),
            outer_val_min=str(val.RENT_APPROVAL_DATE.min()),held_block_count=len(held),held_training_rows=0))
        for kind in KINDS:
            for seed in SEEDS:
                tags=dict(experiment=experiment,origin=origin,split_seed=split_seed,model=kind,seed=seed)
                tuned,count,history=train(kind,inner,seed,MAX_EPOCHS,tune=True)
                histories.extend([{**tags,**h} for h in history]);del tuned;gc.collect()
                fitted,_,_=train(kind,outer,seed,count,save_to=OUT/'models'/f'{experiment}_{kind}_seed{seed}.pt',
                                 experiment=experiment,origin=origin,split_seed=split_seed)
                pred=predict(fitted,outer,outer['parts'][1])
                frame=val[['row_id','block_id','RENT_APPROVAL_DATE','MONTHLY_RENT','block_slice']].copy()
                frame['prediction']=pred
                for name,value in tags.items():frame[name]=value
                predictions.append(frame)
                for slice_name,g in [('all',frame)]+list(frame.groupby('block_slice')):
                    results.append(dict(**tags,slice=slice_name,n_rows=len(g),n_blocks=g.block_id.nunique(),
                        **metrics(g.MONTHLY_RENT,g.prediction)))
                selections.append(dict(**tags,iterations=count,parameters=sum(p.numel() for p in fitted.parameters()),
                                       selected_inner_rmse=min(h['inner_rmse'] for h in history)))
                if kind=='GNN_learned_edges':
                    adj=fitted.adjacency().coalesce()
                    idx=adj.indices().numpy();weight=adj.values().detach().numpy()
                    pd.DataFrame(dict(source=idx[0],target=idx[1],normalized_weight=weight)).to_csv(
                        OUT/f'{experiment}_seed{seed}_learned_weights.csv',index=False)
                print('RESULT',tags,'epochs',count,metrics(val.MONTHLY_RENT,pred),flush=True)
                del fitted;gc.collect()
                pd.DataFrame(results).to_csv(OUT/'seed_metrics.csv',index=False)
                pd.DataFrame(selections).to_csv(OUT/'selected_iterations.csv',index=False)
                pd.DataFrame(histories).to_csv(OUT/'tuning_history.csv',index=False)
                pd.DataFrame(audits).to_csv(OUT/'isolation_audit.csv',index=False)
        del inner,outer;gc.collect()
        pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    print('WEIGHTED GNN EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
