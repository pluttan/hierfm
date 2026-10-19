import numpy as np, pandas as pd
E=pd.read_pickle("/tmp/cov05sim/E.pkl")
geo={ # approx centroid lat,lon ; winter UTC offset  (hand-entered, approximate)
"AT":(47.5,14.5,1),"BE":(50.6,4.6,1),"BG":(42.7,25.2,2),"CH":(46.8,8.2,1),"CZ":(49.8,15.5,1),"DE":(51.0,10.3,1),
"DK":(56.0,9.5,1),"EE":(58.7,25.5,2),"ES":(40.2,-3.6,1),"FI":(64.0,26.0,2),"FR":(46.6,2.5,1),"GB_GBN":(53.0,-1.8,0),
"GR":(39.0,22.0,2),"HR":(45.2,15.6,1),"HU":(47.1,19.4,1),"IE":(53.2,-8.0,0),"IT":(42.8,12.5,1),"LT":(55.3,23.9,2),
"LV":(56.9,24.9,2),"NL":(52.2,5.5,1),"NO":(61.0,9.0,1),"PL":(52.0,19.4,1),"PT":(39.6,-8.0,0),"RO":(45.9,25.0,2),
"RS":(44.0,21.0,1),"SE":(62.0,15.0,1),"SI":(46.1,14.8,1),"SK":(48.7,19.5,1)}
C=[c for c in geo if c in E.columns]
X=E[C].copy()
X=X["2015-01-01":"2019-12-31"]
# remove per-node yearly mean bias and clip outliers at 5 sd
X=X.groupby(X.index.year).transform(lambda s:s-s.mean())
X=X.clip(-0.2,0.2).dropna()
print("country nodes",len(C),"hours",len(X))
R=np.corrcoef(X.values,rowvar=False)
lat=np.radians([geo[c][0] for c in C]); lon=np.radians([geo[c][1] for c in C]); tz=np.array([geo[c][2] for c in C])
def gc(i,j):
    a=np.sin((lat[j]-lat[i])/2)**2+np.cos(lat[i])*np.cos(lat[j])*np.sin((lon[j]-lon[i])/2)**2
    return 2*6371*np.arcsin(np.sqrt(a))
iu=np.triu_indices(len(C),1)
d=np.array([gc(i,j) for i,j in zip(*iu)]); r=R[iu]; st=(tz[iu[0]]==tz[iu[1]])
print("pairs",len(r),"mean corr %.3f median %.3f  q10 %.3f q90 %.3f"%(r.mean(),np.median(r),*np.quantile(r,[.1,.9])))
for lo,hi in [(0,600),(600,1200),(1200,2000),(2000,4000)]:
    m=(d>=lo)&(d<hi); print(f"dist {lo}-{hi} km: n={m.sum()} mean corr {r[m].mean():.3f}")
print("same tz mean %.3f diff tz mean %.3f"%(r[st].mean(),r[~st].mean()))
print("corr(distance, rho) = %.3f"%np.corrcoef(np.log(d),r)[0,1])
ev=np.linalg.eigvalsh(R)[::-1]; print("top eigenvalue shares of correlation matrix:",np.round(ev[:4]/len(C),3))
# window experiment: long-sample correlation as reference; windows of n days (hourly obs), compare estimators
off=~np.eye(len(C),dtype=bool); p=len(C)
dm=np.zeros((p,p)); 
for (i,j),v in zip(zip(*iu),d): dm[i,j]=dm[j,i]=v
def kern_target(r):
    best=None
    for l in [300,600,1000,1500,2500,4000]:
        k=np.exp(-dm/l)[off]; a=(k*r[off]).sum()/(k*k).sum(); e=((r[off]-a*k)**2).sum()
        if best is None or e<best[0]: best=(e,a,l)
    _,a,l=best; t=a*np.exp(-dm/l); np.fill_diagonal(t,1); return t
def const_target(r):
    t=np.full((p,p),r[off].mean()); np.fill_diagonal(t,1); return t
def ss(Xw,tf):
    n=len(Xw); Z=(Xw-Xw.mean(0))/Xw.std(0,ddof=1); rr=Z.T@Z/(n-1); np.fill_diagonal(rr,1)
    t=tf(rr); W=Z[:,:,None]*Z[:,None,:]; v=n/(n-1)**3*((W-W.mean(0))**2).sum(0)
    lam=min(1,max(0,v[off].sum()/((rr-t)[off]**2).sum())); return (1-lam)*rr+lam*t,lam
rng=np.random.default_rng(0); V=X.values; N=len(V)
print("\nWindow experiment (hourly errors, 28 countries; reference = full 2015-2019 correlation). rel. Frobenius error, mean over 60 random windows [lambda]")
print("window".ljust(14)+"sample".rjust(10)+"SS-const".rjust(16)+"SS-distKernel".rjust(16))
for days in [7,14,28,84,365]:
    n=days*24; res={"s":[],"c":[],"k":[]}; lam={"c":[],"k":[]}
    for _ in range(60):
        s=rng.integers(0,N-n); w=V[s:s+n]
        z=(w-w.mean(0))/w.std(0,ddof=1); rs=z.T@z/(n-1)
        res["s"].append(np.linalg.norm(rs-R)/np.linalg.norm(R))
        for k,tf in [("c",const_target),("k",kern_target)]:
            E_,l=ss(w,tf); res[k].append(np.linalg.norm(E_-R)/np.linalg.norm(R)); lam[k].append(l)
    print(f"{days} days".ljust(14)+f"{np.mean(res[chr(115)]):.3f}".rjust(10)+f"{np.mean(res[chr(99)]):.3f} [{np.mean(lam[chr(99)]):.2f}]".rjust(16)+f"{np.mean(res[chr(107)]):.3f} [{np.mean(lam[chr(107)]):.2f}]".rjust(16))
