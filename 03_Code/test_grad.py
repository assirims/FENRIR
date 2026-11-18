import numpy as np, autograd as ag
ag.set_dtype(np.float64)
import models
rng=np.random.default_rng(0)
d,C,B=7,4,5
A=rng.random((d,d)); A=np.where(A>0.5,A,0); A+=np.eye(d); A/=A.sum(1,keepdims=True)
X=rng.normal(size=(B,d)); y=rng.integers(0,C,B); w=np.ones(C)
def check(m,name):
    m.drop=0
    def loss():
        return ag.cross_entropy(m.forward(X),y,w,0.1)
    for p in m.params: p.grad=None
    L=loss(); L.backward()
    maxrel=0
    for p in m.params[:]:
        g=p.grad if p.grad is not None else np.zeros_like(p.data)
        idx=[tuple(rng.integers(0,s) for s in p.data.shape) for _ in range(3)]
        for ix in idx:
            o=p.data[ix]; e=1e-6
            p.data[ix]=o+e; lp=float(loss().data); p.data[ix]=o-e; lm=float(loss().data); p.data[ix]=o
            num=(lp-lm)/(2*e); an=g[ix]
            rel=abs(num-an)/(abs(num)+abs(an)+1e-9)
            maxrel=max(maxrel,rel if abs(num)+abs(an)>1e-7 else 0)
    print(name, m.n_params(), 'max rel err', maxrel)
for nm in ['MLP','CNN1D','GCN','GAT','Transformer']:
    check(models.build(nm,d,C,A,{'D':8,'heads':2,'drop':0}),nm)
for fl in [{},{'periodic':False},{'dual_view':False},{'adj':'prior'},{'adj':'adaptive'},{'gate':False},{'use_G':False},{'use_S':False},{'chan_att':False},{'att_pool':False}]:
    check(models.build('TAGGNN',d,C,A,{'D':8,'heads':2,'drop':0,'flags':fl}),'TAGGNN'+str(fl))
