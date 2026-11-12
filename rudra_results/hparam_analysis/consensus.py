import pandas as pd, numpy as np
pd.set_option("display.width",230,"display.max_columns",40,"display.max_rows",300)
t=pd.read_csv('/home/hp/Downloads/Reglo/rudra_results/full/trial_table/trials_full.csv'); t=t[t.status!='EXTRA_NOT_IN_CURRENT_GRID']
SRC=['svhn','cifar10','mnist','fmnist']
G=sorted(t.adamw_lr.dropna().unique())          # the shared 20-value grid
idx={round(v,6):i for i,v in enumerate(G)}
def ix(v): return idx[round(v,6)]

def grid2(x,a,b):
    """dict src -> 20x20 array of val_acc (nan where missing), a=row param, b=col param"""
    out={}
    for s in SRC:
        z=np.full((20,20),np.nan); y=x[x.source==s]
        for _,r in y.iterrows(): z[ix(r[a]),ix(r[b])]=r.val_acc
        out[s]=z
    return out
def grid1(x,a):
    out={}
    for s in SRC:
        z=np.full(20,np.nan); y=x[x.source==s]
        for _,r in y.iterrows(): z[ix(r[a])]=r.val_acc
        out[s]=z
    return out
def smooth(z,k=3):
    m=np.isnan(z); w=np.where(m,0,z); c=(~m).astype(float)
    def f(u):                                        # k-box sum with zero padding, any dimension
        r=k//2; p=np.pad(u,r); o=np.zeros_like(u,dtype=float)
        for sh in np.ndindex(*([k]*u.ndim)):
            sl=tuple(slice(a,a+n) for a,n in zip(sh,u.shape)); o+=p[sl]
        return o
    return np.where(f(c)>0,f(w)/np.maximum(f(c),1e-9),np.nan)
def pick(grids,names,topn=5,label=''):
    """grids: src->array (1D or 2D). regret in pp. returns table of best consensus cells"""
    present=np.all([~np.isnan(g) for g in grids.values()],axis=0)
    grids={s:np.where(present,g,np.nan) if False else g for s,g in grids.items()}
    R={s:(np.nanmax(g)-g)*100 for s,g in grids.items()}
    Sm={s:(np.nanmax(smooth(g))-smooth(g))*100 for s,g in grids.items()}
    mean_raw=np.nanmean([R[s] for s in SRC],axis=0); mean_sm=np.nanmean([Sm[s] for s in SRC],axis=0)
    worst=np.nanmax([R[s] for s in SRC],axis=0)
    ok=present&~np.isnan(mean_sm)
    order=np.argsort(np.where(ok,mean_sm,1e9),axis=None)[:topn]
    mm=np.argsort(np.where(present,worst,1e9),axis=None)[:3]
    rows=[]
    for o in order:
        i=np.unravel_index(o,mean_sm.shape)
        rows.append(dict(**{n:G[j] for n,j in zip(names,i)},meanRegret_smooth=mean_sm[i],meanRegret_raw=mean_raw[i],worstDataset_regret=worst[i],**{f'reg_{s}':R[s][i] for s in SRC}))
    d=pd.DataFrame(rows); print(f'\n=== {label}  (cells present in all 4 datasets: {int(present.sum())})'); print(d.round(3).to_string(index=False))
    r2=[]
    for o in mm:
        i=np.unravel_index(o,mean_sm.shape)
        r2.append(dict(**{n:G[j] for n,j in zip(names,i)},meanRegret_raw=mean_raw[i],worstDataset_regret=worst[i]))
    print('  minimax(worst-dataset regret):',pd.DataFrame(r2).round(3).to_dict('records')); return d

out={}
x=t[t.method=='M1']; out['M1']=pick(grid1(x,'adamw_lr'),['adamw_lr'],label='M1 AdamW lr')
x=t[t.method=='M3']; out['M3']=pick(grid1(x,'meta_lr'),['meta_lr'],label='M3 meta_lr')
x=t[t.method=='M4a']; out['M4a']=pick(grid2(x,'ls','meta_lr'),['ls','meta_lr'],label='M4a (λ_smooth x meta_lr)')
for reg in ['l1','l2']:
    x=t[(t.method=='M2')&(t.reg==reg)]; out['M2'+reg]=pick(grid2(x,'adamw_lr','lam'),['adamw_lr','lam'],label=f'M2 {reg} (lr x λ)')
for reg in ['sam','gsam','gam']:
    x=t[(t.method=='M2')&(t.reg==reg)&(t.phase=='m2')]; out['M2'+reg]=pick(grid2(x,'adamw_lr','lam'),['adamw_lr','lam'],label=f'M2 {reg} stage1 (lr x λ, default rho)')
    x=t[(t.method=='M2')&(t.reg==reg)&(t.phase=='m2rho')]; out['M2rho'+reg]=pick(grid2(x,'lam','rho'),['lam','rho'],label=f'M2 {reg} stage2 (λ x ρ at fixed lr)')
    print('stage-2 fixed lr per dataset:',x.groupby('source').adamw_lr.first().to_dict())
for reg in ['l1','l2','sam','gsam','gam']:
    x=t[(t.method=='M5')&(t.reg==reg)&(t.phase=='m5')]; out['M5'+reg]=pick(grid2(x,'ls','lam_reg'),['ls','lam_reg'],label=f'M5 {reg} stage1 (λ_smooth x λ_reg)')
    print('stage-1 base meta_lr per dataset:',x.groupby('source').meta_lr.first().to_dict())
    y=t[(t.method=='M5')&(t.reg==reg)&(t.phase=='m5mlr')]; print('stage-2 (ls,lam_reg) fixed per dataset:',y.groupby('source')[['ls','lam_reg']].first().to_dict('index'))
