"""Interpretability analysis of the trained FENRIR model (seed 0): node importance, learned graph,
mixing coefficient lambda, gate statistics and per-head structural strengths gamma."""
import json, os
import numpy as np
import autograd as ag
import data, models
from exp_fl import fenrir_hp, selected

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    sel = selected()
    hp = fenrir_hp()
    names = [fd.features[i] for i in sel]
    A = data.prior_adjacency(fd.corr, sel, hp["topk"])
    m = models.build("TAGGNN", len(sel), len(fd.classes), A, hp, seed=0)
    m.set_flat(np.load(os.path.join(RES, "runs", "main", "FENRIR_s0_weights.npy")))
    m.train = False
    X, y = fd.X[np.ix_(fd.test, sel)], fd.y[fd.test]
    imps, chans, pools, gates = [], [], [], []
    for i in range(0, len(X), 1024):
        m.forward(X[i:i + 1024])
        ch = m.cache["chan"][-1]
        po = m.cache["pool"]
        imps.append(ch * po)
        chans.append(ch); pools.append(po)
        gates.append(m.cache.get("gate", [np.nan]))
    imp = np.concatenate(imps)
    imp = imp / imp.sum(1, keepdims=True)
    Ah = m.cache["A"]
    lam = float(1 / (1 + np.exp(-m.theta.data[0]))) if hasattr(m, "theta") else None
    gam = [lay["att"].m and lay["gam"].data.ravel().tolist() for lay in m.layers if "gam" in lay]
    per_class = {fd.classes[c]: imp[y == c].mean(0).tolist() for c in range(len(fd.classes)) if np.any(y == c)}
    out = dict(features=names, importance=imp.mean(0).tolist(), per_class=per_class, A_hat=Ah.tolist(),
               A_prior=A.tolist(), lam=lam, gamma=gam, gate_mean=np.nanmean(np.array(gates), 0).tolist())
    json.dump(out, open(os.path.join(RES, "interpretability.json"), "w"))
    o = np.argsort(-imp.mean(0))
    print("lambda", lam, "gamma", gam, "gate", out["gate_mean"])
    print([(names[i], round(float(imp.mean(0)[i]), 4)) for i in o[:10]])


if __name__ == "__main__":
    main()
