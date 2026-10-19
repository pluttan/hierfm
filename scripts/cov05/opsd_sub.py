import numpy as np, pandas as pd
E=pd.read_pickle("/tmp/cov05sim/E.pkl")["2015-01-01":"2019-12-31"]
G={"DE":["DE_50hertz","DE_amprion","DE_tennet","DE_transnetbw"],"SE":["SE_1","SE_2","SE_3","SE_4"],
"NO":["NO_1","NO_2","NO_3","NO_4","NO_5"],"IT":["IT_CNOR","IT_CSUD","IT_NORD","IT_SARD","IT_SICI","IT_SUD"],"DK":["DK_1","DK_2"]}
allc=[c for g in G.values() for c in g]
X=E[allc].copy(); X=X.groupby(X.index.year).transform(lambda s:s-s.mean()).clip(-0.2,0.2).dropna()
print("hours",len(X))
R=pd.DataFrame(np.corrcoef(X.values,rowvar=False),index=allc,columns=allc)
gi={c:k for k,g in G.items() for c in g}
intra=[];inter=[]
for i,a in enumerate(allc):
    for b in allc[i+1:]:
        (intra if gi[a]==gi[b] else inter).append(R.loc[a,b])
print("zones within the same country: n=%d mean corr %.3f ; across countries: n=%d mean %.3f"%(len(intra),np.mean(intra),len(inter),np.mean(inter)))
for k,g in G.items():
    m=R.loc[g,g].values; iu=np.triu_indices(len(g),1); print(k,"mean intra corr %.3f min %.3f max %.3f"%(m[iu].mean(),m[iu].min(),m[iu].max()))
# daily-mean errors
D=X.resample("D").mean(); Rd=np.corrcoef(D.values,rowvar=False)
intra=[];inter=[]
for i,a in enumerate(allc):
    for j,b in enumerate(allc):
        if j>i: (intra if gi[a]==gi[b] else inter).append(Rd[i,j])
print("DAILY-mean errors: intra %.3f inter %.3f"%(np.mean(intra),np.mean(inter)))
# country level: total vs sum of zones consistency: error of country total correlated with zone errors
for k in ["DE","SE","NO","IT","DK"]:
    if k in E.columns:
        y=E[[k]+G[k]].dropna(); print(k,"corr of country-total error with zone errors:",np.round(np.corrcoef(y.clip(-.2,.2).values,rowvar=False)[0,1:],2))
