import numpy as np, sys
from sklearn.covariance import LedoitWolf, OAS
rng=np.random.default_rng(1)
p=88
sizes=[22,14,12,11,10,10,9]
blk=np.repeat(np.arange(7),sizes)
centers=rng.uniform(0,10,(7,2))*np.array([3,1])  # elongated
xy=centers[blk]+rng.normal(0,0.8,(p,2))
d=np.sqrt(((xy[:,None]-xy[None])**2).sum(-1))
same=(blk[:,None]==blk[None]).astype(float)
def mk(case):
    if case=="A":
        R=np.where(same==1,0.5,0.2)
    else:
        K=np.exp(-d/6.0)
        R=0.5*np.where(same==1,0.5,0.2)+0.5*K
    np.fill_diagonal(R,1.0)
    return R
def frob(A,B): return np.linalg.norm(A-B,"fro")
def stein(S,T):
    M=np.linalg.solve(S,T); s,ld=np.linalg.slogdet(M); return np.trace(M)-ld-p
def ss_shrink(X,target_fn):
    n=len(X); Z=(X-X.mean(0))/X.std(0,ddof=1)
    r=Z.T@Z/(n-1); np.fill_diagonal(r,1)
    t=target_fn(r)
    W=Z[:,:,None]*Z[:,None,:]; wbar=W.mean(0)
    v=n/(n-1)**3*((W-wbar)**2).sum(0)
    off=~np.eye(p,dtype=bool)
    lam=v[off].sum()/max(((r-t)[off]**2).sum(),1e-12)
    lam=min(1,max(0,lam))
    return (1-lam)*r+lam*t, lam
def t_block(r):
    off=~np.eye(p,dtype=bool)
    a=r[(same==1)&off].mean(); b=r[(same==0)].mean()
    t=np.where(same==1,a,b); np.fill_diagonal(t,1); return t
def t_diag(r): return np.eye(p)
def t_const(r):
    off=~np.eye(p,dtype=bool); t=np.full((p,p),r[off].mean()); np.fill_diagonal(t,1); return t
def t_kernel(r):
    # fit amplitude a and length scale by grid on pooled r (exp decay), rho=a*exp(-d/l)
    off=~np.eye(p,dtype=bool); best=None
    for l in [2,4,6,8,12,20]:
        k=np.exp(-d/l)[off]; a=(k*r[off]).sum()/(k*k).sum(); e=((r[off]-a*k)**2).sum()
        if best is None or e<best[0]: best=(e,a,l)
    _,a,l=best; t=a*np.exp(-d/l); np.fill_diagonal(t,1); return t
def t_blockkernel(r):
    off=~np.eye(p,dtype=bool); best=None
    X=np.stack([same[off], np.exp(-d/6.0)[off], np.ones(off.sum())],1)
    c,*_=np.linalg.lstsq(X,r[off],rcond=None)
    t=np.zeros((p,p)); t[off]=X@c; np.fill_diagonal(t,1)
    # ensure PD by eigen clip
    w,V=np.linalg.eigh(t); w=np.maximum(w,0.05); t=(V*w)@V.T; dd=np.sqrt(np.diag(t)); return t/dd[:,None]/dd[None]
ests={"sample/pinv":None,"diag(SS-D)":t_diag,"const(SS-F)":t_const,"block":t_block,"kernel":t_kernel,"block+kernel":t_blockkernel}
ns=[7,14,28,56,112,365]
reps=int(sys.argv[1]) if len(sys.argv)>1 else 40
for case in ["A","B"]:
    R=mk(case); L=np.linalg.cholesky(R+1e-9*np.eye(p))
    print(f"\n=== truth {case}: p={p}, relative Frobenius error of correlation matrix (mean over {reps} reps); [lambda]")
    hdr="n".ljust(5)+"".join(k.rjust(18) for k in list(ests)+["LW(sklearn)","OAS(sklearn)"])
    print(hdr)
    for n in ns:
        res={k:[] for k in list(ests)+["LW(sklearn)","OAS(sklearn)"]}; lam={k:[] for k in ests}
        for _ in range(reps):
            X=rng.standard_normal((n,p))@L.T
            r=np.corrcoef(X,rowvar=False)
            res["sample/pinv"].append(frob(r,R)/frob(R,np.eye(p)*0))
            for k,f in ests.items():
                if f is None: continue
                E,l=ss_shrink(X,f); res[k].append(frob(E,R)/frob(R,np.eye(p)*0)); lam[k].append(l)
            Z=(X-X.mean(0))/X.std(0,ddof=1)
            for nm,M in [("LW(sklearn)",LedoitWolf()),("OAS(sklearn)",OAS())]:
                C=M.fit(Z).covariance_; dd=np.sqrt(np.diag(C)); C=C/dd[:,None]/dd[None]
                res[nm].append(frob(C,R)/frob(R,np.eye(p)*0))
        row=str(n).ljust(5)
        for k in res:
            m=np.mean(res[k]); row+=(f"{m:.3f}"+(f" [{np.mean(lam[k]):.2f}]" if k in lam and lam[k] else "")).rjust(18)
        print(row)
