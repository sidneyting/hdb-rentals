"""Controlled GNN experiments with explicit and learned geographic edge weights.

PYTHONPATH=src python scripts/experiment_weighted_gnns.py
Set HDB_WEIGHTED_GNN_SMOKE=1 for a separate short execution check.
"""
# Import torch first on this Mac to avoid mixed OpenMP import-order crashes.
import torch
from torch import nn
from pathlib import Path
import os,time,json,gc,random
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import BallTree
from investigate_graph_signal import ROOT,blocks,rent,edges,N,BLOCK_NUM,BLOCK_CAT,preprocessor,metrics
from hdb_rentals.rent_adjustment import MarketRentAdjustment
from hdb_rentals.block_evaluation import block_time_partitions

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

# Symmetric geographic edges with distances from the original 1 km/8-NN graph.
u=edges.source.to_numpy(int);v=edges.target.to_numpy(int)
row=np.r_[u,v];col=np.r_[v,u]
distance=np.tile(edges.distance_km.to_numpy(float),2)
year=blocks.YEAR_COMPLETED.fillna(blocks.YEAR_COMPLETED.median()).to_numpy()
floor=blocks.MAX_FLOOR.fillna(blocks.MAX_FLOOR.median()).to_numpy()
edge_x=np.column_stack([distance,abs(year[row]-year[col])/20.,abs(floor[row]-floor[col])/10.,
    (blocks.SUBZONE.iloc[row].to_numpy()==blocks.SUBZONE.iloc[col].to_numpy()).astype(float),
    (blocks.TOWN.iloc[row].to_numpy()==blocks.TOWN.iloc[col].to_numpy()).astype(float),
    abs(blocks.mrt_nearest_km.to_numpy()[row]-blocks.mrt_nearest_km.to_numpy()[col])]).astype('float32')
EDGE_INDEX=torch.tensor(np.stack([row,col]),dtype=torch.long)
EDGE_FEATURES=torch.tensor(edge_x)

# A separate neighborhood band tests whether the original graph is too local.
coords=np.radians(blocks[['LATITUDE','LONGITUDE']].to_numpy())
tree=BallTree(coords,metric='haversine')
far_ids,far_dist=tree.query_radius(coords,r=1.5/6371.0088,return_distance=True,sort_results=True)
far_rows=[];far_cols=[];far_weights=[]
for i,(ids,dists) in enumerate(zip(far_ids,far_dist)):
    km=dists*6371.0088
    valid=(km>=.25)&(km<=1.5)
    ids,km=ids[valid][:8],km[valid][:8]
    far_rows.extend([i]*len(ids));far_cols.extend(ids);far_weights.extend(np.exp(-km/.75))
far_sparse=coo_matrix((far_weights,(far_rows,far_cols)),shape=(N,N)).tocsr()
far_sparse=far_sparse.maximum(far_sparse.T).tocoo()
FAR_INDEX=torch.tensor(np.stack([far_sparse.row,far_sparse.col]),dtype=torch.long)
fw=torch.tensor(far_sparse.data,dtype=torch.float32)
fd=torch.zeros(N).scatter_add_(0,FAR_INDEX[0],fw)
FAR_ADJACENCY=torch.sparse_coo_tensor(FAR_INDEX,fw/fd[FAR_INDEX[0]].clamp_min(1e-10),(N,N),check_invariants=True).coalesce()
FAR_PRESENT=fd.gt(0).float().unsqueeze(1)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


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


class WeightedRentalGNN(nn.Module):
    def __init__(self,kind,node_dim,record_dim):
        super().__init__();self.kind=kind
        self.projection=nn.Linear(node_dim,HIDDEN)
        self.roots=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN) for _ in range(2)])
        self.messages=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN,bias=False) for _ in range(2)])
        self.gates=nn.ModuleList([nn.Linear(2*HIDDEN,HIDDEN) for _ in range(2)])
        for gate in self.gates:nn.init.constant_(gate.bias,-2.)
        self.drop=nn.Dropout(.2)
        self.head=nn.Sequential(nn.Linear(HIDDEN+record_dim,HIDDEN),nn.ReLU(),nn.Dropout(.2),nn.Linear(HIDDEN,1))
        if kind=='GNN_learned_edges':
            self.edge_network=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,1))
            nn.init.zeros_(self.edge_network[-1].weight);nn.init.zeros_(self.edge_network[-1].bias)
            self.log_tau=nn.Parameter(torch.tensor(float(np.log(np.expm1(.3)))))
        if kind=='GNN_multiscale':
            self.scale_logits=nn.Parameter(torch.zeros(2))
            self.register_buffer('far_adjacency',FAR_ADJACENCY)
            self.register_buffer('far_present',FAR_PRESENT)
        self.register_buffer('edge_index',EDGE_INDEX)
        self.register_buffer('edge_features',EDGE_FEATURES)
        fixed=torch.ones(len(row))
        if kind=='GNN_distance':fixed=torch.exp(-self.edge_features[:,0]/.3)
        if kind=='GNN_similarity':
            fixed=torch.exp(-self.edge_features[:,0]/.3-self.edge_features[:,1]*(20/15)-self.edge_features[:,2]*(10/8))
        self.register_buffer('fixed_weights',fixed)
        if kind not in ['MLP_matched','GNN_learned_edges']:
            self.register_buffer('fixed_adjacency',self.normalize(fixed))

    def normalize(self,w):
        degree=torch.zeros(N,dtype=w.dtype).scatter_add_(0,self.edge_index[0],w)
        w=w/degree[self.edge_index[0]].clamp_min(1e-10)
        return torch.sparse_coo_tensor(self.edge_index,w,(N,N),check_invariants=True).coalesce()

    def adjacency(self):
        if self.kind=='MLP_matched':return None
        if self.kind=='GNN_learned_edges':
            tau=torch.nn.functional.softplus(self.log_tau).clamp(.03,3.)
            w=torch.exp(-self.edge_features[:,0]/tau)*torch.sigmoid(self.edge_network(self.edge_features).squeeze(-1))
            return self.normalize(w)
        return self.fixed_adjacency

    def forward(self,nodes,part):
        z=self.drop(torch.relu(self.projection(nodes)))
        adjacency=self.adjacency()
        for layer,(root,message,gate) in enumerate(zip(self.roots,self.messages,self.gates)):
            # Identity adjacency in MLP keeps the same active layers and gates.
            aggregated=z if adjacency is None else torch.sparse.mm(adjacency,z)
            if self.kind=='GNN_multiscale':
                amount=torch.sigmoid(self.scale_logits[layer])*self.far_present
                aggregated=(1-amount)*aggregated+amount*torch.sparse.mm(self.far_adjacency,z)
            incoming=message(aggregated)
            amount=torch.sigmoid(gate(torch.cat([z,incoming],dim=1)))
            z=self.drop(torch.relu(root(z)+amount*incoming))
        return self.head(torch.cat([z[part['ids']],part['record']],dim=1)).squeeze(-1)


@torch.no_grad()
def predict(model,bundle,part):
    model.eval()
    adjusted=model(bundle['nodes'],part).numpy()*bundle['std']+bundle['mean']
    return np.maximum(adjusted,0)*part['scale']


def train(kind,bundle,seed,epochs,tune=False):
    seed_everything(seed)
    model=WeightedRentalGNN(kind,bundle['nodes'].shape[1],bundle['parts'][0]['record'].shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    fit=bundle['parts'][0];best=float('inf');best_epoch=epochs;stale=0;history=[]
    start=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        err=model(bundle['nodes'],fit)-fit['scaled_y']
        loss=(fit['weight']*err.square()).mean()
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        record=dict(epoch=epoch,train_objective=float(loss.detach()))
        if tune:
            score=metrics(bundle['parts'][1]['y'],predict(model,bundle,bundle['parts'][1]))['rmse']
            record['inner_rmse']=score
            if score<best-.01:best=score;best_epoch=epoch;stale=0
            else:stale+=1
        history.append(record)
        if epoch==1 or epoch%40==0:
            print(kind,seed,'tune' if tune else 'refit',epoch,'loss',round(record['train_objective'],4),
                  'seconds',round(time.monotonic()-start,1),flush=True)
        if tune and stale>=PATIENCE:break
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
                fitted,_,_=train(kind,outer,seed,count)
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
