#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, time, traceback
from pathlib import Path
from typing import Dict, List, Tuple
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def balanced_labels(n,k,rng):
    counts=np.full(k,n//k,dtype=int); counts[:n%k]+=1
    y=np.concatenate([np.full(counts[j],j,dtype=int) for j in range(k)])
    rng.shuffle(y); return y

def generate_matched_data(n,d,d_info,k,delta,signal_sd,noise_sd,seed):
    if not (0<d_info<d): raise ValueError('Require 0 < d_info < d')
    rng=np.random.default_rng(seed); y=balanced_labels(n,k,rng)
    X=rng.normal(0.0,noise_sd,size=(n,d)); support=np.arange(d_info,dtype=int)
    centers=np.linspace(-delta,delta,k)
    for c in range(k):
        rows=y==c
        X[np.ix_(rows,support)] = rng.normal(loc=centers[c],scale=signal_sd,size=(rows.sum(),d_info))
    perm=rng.permutation(n); return X[perm],y[perm],support

def fit_diag_gmm(Z,y,k,seed,n_init,max_iter):
    m=GaussianMixture(n_components=k,covariance_type='diag',n_init=n_init,max_iter=max_iter,reg_covar=1e-6,random_state=seed)
    lab=m.fit_predict(Z)
    return {'ARI':float(adjusted_rand_score(y,lab)),'NMI':float(normalized_mutual_info_score(y,lab)),'converged':bool(m.converged_),'n_iter':int(m.n_iter_)}

def laplacian_scores(X,n_neighbors=5,heat_scale='median'):
    n,d=X.shape
    nn=NearestNeighbors(n_neighbors=n_neighbors+1,metric='euclidean').fit(X)
    dist,ind=nn.kneighbors(X); dist=dist[:,1:]; ind=ind[:,1:]
    sq=dist**2; nz=sq[sq>0]
    t=1.0 if nz.size==0 else float(np.median(nz) if heat_scale=='median' else np.mean(nz)); t=max(t,1e-12)
    w=np.exp(-sq/t); rows=np.repeat(np.arange(n),n_neighbors); cols=ind.reshape(-1); vals=w.reshape(-1)
    W=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n)); W=W.maximum(W.T)
    deg=np.asarray(W.sum(axis=1)).ravel(); s=float(deg.sum())
    if s<=0: raise RuntimeError('zero-degree graph')
    mu=(deg[:,None]*X).sum(axis=0)/s; F=X-mu
    DF=deg[:,None]*F; denom=np.sum(F*DF,axis=0); numer=denom-np.sum(F*(W@F),axis=0)
    out=np.full(d,np.inf); ok=denom>1e-12; out[ok]=numer[ok]/denom[ok]; return out

def support_f1(selected,true_support,d):
    t=np.zeros(d,int); p=np.zeros(d,int); t[true_support]=1; p[selected]=1
    return float(f1_score(t,p,zero_division=0))

def run_pca_gmm(X,y,args,seed):
    q=min(args.pca_components,X.shape[0]-1,X.shape[1]); t0=time.perf_counter()
    pca=PCA(n_components=q,random_state=seed); Z=pca.fit_transform(X)
    r=fit_diag_gmm(Z,y,args.k,seed,args.gmm_n_init,args.gmm_max_iter)
    r.update(method='PCA+diag-GMM',dimensions_used=int(q),feature_f1=np.nan,runtime_sec=float(time.perf_counter()-t0),pca_variance_explained=float(pca.explained_variance_ratio_.sum()))
    return r

def run_ls_gmm(X,y,support,args,seed):
    m=min(args.ls_features,X.shape[1]); t0=time.perf_counter(); sc=laplacian_scores(X,args.ls_neighbors,args.ls_heat_scale)
    sel=np.argsort(sc)[:m]; Z=X[:,sel]
    r=fit_diag_gmm(Z,y,args.k,seed,args.gmm_n_init,args.gmm_max_iter)
    r.update(method='LaplacianScore+diag-GMM',dimensions_used=int(m),feature_f1=support_f1(sel,support,X.shape[1]),runtime_sec=float(time.perf_counter()-t0),mean_selected_score=float(np.mean(sc[sel])))
    return r

def summarize(raw):
    metrics=['ARI','NMI','runtime_sec','dimensions_used','feature_f1','converged','n_iter','pca_variance_explained','mean_selected_score']
    ex=[m for m in metrics if m in raw.columns]
    s=raw.groupby(['n','d','d_info','delta','method'],dropna=False)[ex].agg(['mean','std','count']).reset_index()
    s.columns=['_'.join(str(x) for x in c if str(x)!='').rstrip('_') if isinstance(c,tuple) else c for c in s.columns]
    return s

def make_latex_table(summary,path):
    rows=[r'\begin{table}[t]',r'\centering',r'\caption{Two-stage baselines on the matched synthetic benchmark. Values are mean (standard deviation) over independently generated datasets.}',r'\label{tab:two_stage_baselines}',r'\begin{tabular}{clcccc}',r'\toprule',r'$N$ & Method & ARI & NMI & Dim. used & Feature F1 \\',r'\midrule']
    for n in sorted(summary['n'].unique()):
        sub=summary[summary['n']==n]
        for _,row in sub.iterrows():
            def fmt(metric,digits=3):
                mean=row.get(f'{metric}_mean',np.nan); sd=row.get(f'{metric}_std',np.nan)
                if pd.isna(mean): return '--'
                if pd.isna(sd): return f'{mean:.{digits}f}'
                return f'{mean:.{digits}f} ({sd:.{digits}f})'
            rows.append(f"{int(n)} & {row['method']} & {fmt('ARI')} & {fmt('NMI')} & {fmt('dimensions_used',1)} & {fmt('feature_f1')} \\")
        rows.append(r'\addlinespace')
    rows += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(rows),encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--output_dir',default='/content/two_stage_baselines'); p.add_argument('--seeds',type=int,nargs='+',default=list(range(1,21))); p.add_argument('--sample_sizes',type=int,nargs='+',default=[200,1000]); p.add_argument('--quick',action='store_true')
    p.add_argument('--d',type=int,default=100); p.add_argument('--d_info',type=int,default=10); p.add_argument('--k',type=int,default=3); p.add_argument('--delta',type=float,default=2.0); p.add_argument('--signal_sd',type=float,default=1.0); p.add_argument('--noise_sd',type=float,default=3.0)
    p.add_argument('--pca_components',type=int,default=50); p.add_argument('--ls_features',type=int,default=10); p.add_argument('--ls_neighbors',type=int,default=5); p.add_argument('--ls_heat_scale',choices=['median','mean'],default='median')
    p.add_argument('--gmm_n_init',type=int,default=10); p.add_argument('--gmm_max_iter',type=int,default=300); return p.parse_args()

def main():
    args=parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    seeds=[1,2] if args.quick else args.seeds; sizes=[200] if args.quick else args.sample_sizes
    cfg=vars(args).copy(); cfg['effective_seeds']=seeds; cfg['effective_sample_sizes']=sizes; cfg['laplacian_score_direction']='lower_is_better'; cfg['ls_feature_budget_is_oracle']=bool(args.ls_features==args.d_info)
    (out/'two_stage_baselines_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    rec=[]
    for n in sizes:
      for seed in seeds:
        print(f'[run] n={n} d={args.d} d_info={args.d_info} delta={args.delta} seed={seed}',flush=True)
        try:
            X,y,sup=generate_matched_data(n,args.d,args.d_info,args.k,args.delta,args.signal_sd,args.noise_sd,seed); X=StandardScaler().fit_transform(X)
            base={'n':n,'seed':seed,'d':args.d,'d_info':args.d_info,'delta':args.delta,'status':'ok','error':''}
            rec.append({**base,**run_pca_gmm(X,y,args,seed)}); rec.append({**base,**run_ls_gmm(X,y,sup,args,seed)})
        except Exception as e:
            print(f'[ERROR] {e!r}',flush=True); traceback.print_exc(); rec.append({'n':n,'seed':seed,'d':args.d,'d_info':args.d_info,'delta':args.delta,'method':'FAILED','status':'failed','error':repr(e)})
    raw=pd.DataFrame(rec); raw.to_csv(out/'two_stage_baselines_raw.csv',index=False)
    ok=raw[raw['status']=='ok'].copy()
    if ok.empty: raise RuntimeError('All runs failed')
    s=summarize(ok); s.to_csv(out/'two_stage_baselines_summary.csv',index=False); make_latex_table(s,out/'two_stage_baselines_table.tex')
    cols=[c for c in ['n','method','ARI_mean','ARI_std','NMI_mean','NMI_std','dimensions_used_mean','feature_f1_mean','feature_f1_std','runtime_sec_mean','runtime_sec_std'] if c in s.columns]
    print('\n[done]'); print(s[cols].to_string(index=False)); print(f'\nResults written to: {out}')
if __name__=='__main__': main()
