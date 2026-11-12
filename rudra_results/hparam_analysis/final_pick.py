import pandas as pd, numpy as np, json
exec(open('consensus.py').read().split("\nout={}\nx=t")[0])      # reuse loaders + smooth()
def stats(grids):
    present=np.all([~np.isnan(g) for g in grids.values()],axis=0)
    R={s:(np.nanmax(g)-g)*100 for s,g in grids.items()}
    Sm={s:(np.nanmax(smooth(g))-smooth(g))*100 for s,g in grids.items()}
    return present,R,np.mean([R[s] for s in SRC],0),np.max([R[s] for s in SRC],0),np.mean([Sm[s] for s in SRC],0)
def choose(grids,names,frac=0.05,kmin=4,label='',show=3):
    """flat region (best `frac` of cells by neighbourhood-smoothed mean regret) -> minimise mean+worst-dataset regret"""
    present,R,mraw,worst,msm=stats(grids)
    ok=np.where(present,msm,np.inf).ravel()
    k=max(kmin,int(round(frac*present.sum())))
    cand=np.argsort(ok)[:k]
    sc=(mraw+worst).ravel()
    best=sorted(cand,key=lambda o:sc[o])[:show]
    rows=[]
    for o in best:
        i=np.unravel_index(o,mraw.shape)
        rows.append(dict(**{n:G[j] for n,j in zip(names,i)},mean_regret=mraw[i],worst_regret=worst[i],nbhd_regret=msm[i],**{s:R[s][i] for s in SRC}))
    d=pd.DataFrame(rows); print(f'\n=== {label}'); print(d.round(2).to_string(index=False)); return rows[0]
res={}
x=t[t.method=='M1']; res['M1']=choose(grid1(x,'adamw_lr'),['adamw_lr'],label='M1 lr')
x=t[t.method=='M3']; res['M3']=choose(grid1(x,'meta_lr'),['meta_lr'],label='M3 meta_lr')
x=t[t.method=='M4a']; res['M4a']=choose(grid2(x,'ls','meta_lr'),['ls','meta_lr'],label='M4a (λ_smooth,meta_lr)')
for reg in ['l1','l2']:
    x=t[(t.method=='M2')&(t.reg==reg)]; res['M2:'+reg]=choose(grid2(x,'adamw_lr','lam'),['adamw_lr','lam'],label=f'M2 {reg} (lr,λ)')
for reg in ['sam','gsam','gam']:
    x=t[(t.method=='M2')&(t.reg==reg)&(t.phase=='m2')]; r1=choose(grid2(x,'adamw_lr','lam'),['adamw_lr','lam'],label=f'M2 {reg} stage1 (lr,λ @default ρ)')
    x=t[(t.method=='M2')&(t.reg==reg)&(t.phase=='m2rho')]; r2=choose(grid2(x,'lam','rho'),['lam','rho'],label=f'M2 {reg} stage2 (λ,ρ @ per-dataset best lr)')
    res['M2:'+reg]=dict(adamw_lr=r1['adamw_lr'],lam=r2['lam'],rho=r2['rho'],stage1_lam=r1['lam'])
for reg in ['l1','l2','sam','gsam','gam']:
    x=t[(t.method=='M5')&(t.reg==reg)&(t.phase=='m5')]; r1=choose(grid2(x,'ls','lam_reg'),['ls','lam_reg'],label=f'M5 {reg} stage1 (λ_smooth,λ_reg)')
    # meta_lr profile: dataset's stage-1 best cell (its base meta_lr) + the 19 other meta_lr values
    prof={}
    for s in SRC:
        a=x[x.source==s]; b=a.loc[a.val_acc.idxmax()]
        m=t[(t.method=='M5')&(t.reg==reg)&(t.phase=='m5mlr')&(t.source==s)]
        z=np.full(20,np.nan); z[ix(b.meta_lr)]=b.val_acc
        for _,r in m.iterrows(): z[ix(r.meta_lr)]=r.val_acc
        prof[s]=z
    r2=choose(prof,['meta_lr'],label=f'M5 {reg} meta_lr profile',kmin=5)
    res['M5:'+reg]=dict(ls=r1['ls'],lam_reg=r1['lam_reg'],meta_lr=r2['meta_lr'])
    if reg in ('sam','gsam','gam'):
        prof={}
        for s in SRC:
            a=t[(t.method=='M5')&(t.reg==reg)&(t.source==s)&(t.phase.isin(['m5rho','m5mlr','m5']))]
            m=t[(t.method=='M5')&(t.reg==reg)&(t.phase=='m5rho')&(t.source==s)]
            z=np.full(20,np.nan)
            for _,r in m.iterrows(): z[ix(r.rho)]=r.val_acc
            prof[s]=z
        r3=choose(prof,['rho'],label=f'M5 {reg} rho profile',kmin=4)
        res['M5:'+reg]['rho']=r3['rho']
json.dump(res,open('final_pick.json','w'),indent=1,default=float)
print(json.dumps(res,indent=1,default=float))
