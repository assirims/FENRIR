"""
Federated learning engine: local proximal training, robust aggregation (incl. the proposed
Median-Anchored Trust-weighted Aggregation, MATA), Byzantine attack simulation and evaluation.
"""
import time
import numpy as np
import autograd as ag
from autograd import Adam
import models
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score)


# ----------------------------------------------------------------------------- metrics
def evaluate(y, P, C):
    yp = P.argmax(1)
    out = dict(acc=accuracy_score(y, yp),
               prec=precision_score(y, yp, average="macro", zero_division=0),
               rec=recall_score(y, yp, average="macro", zero_division=0),
               f1=f1_score(y, yp, average="macro", zero_division=0),
               f1w=f1_score(y, yp, average="weighted", zero_division=0),
               mcc=matthews_corrcoef(y, yp))
    try:
        Pn = P / P.sum(1, keepdims=True)
        out["auc"] = roc_auc_score(y, Pn, multi_class="ovr", average="macro", labels=list(range(C)))
    except Exception:
        out["auc"] = float("nan")
    return out


# ----------------------------------------------------------------------------- local training
def local_update(model, w_glob, X, y, epochs, bs, lr, mu, cw, rng, label_flip=False, C=None):
    model.set_flat(w_glob)
    shapes = [p.data.shape for p in model.params]
    gparts, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        gparts.append(w_glob[i:i + n].reshape(s))
        i += n
    opt = Adam(model.params, lr=lr, wd=1e-5)
    yy = (C - 1 - y) if label_flip else y
    model.train = True
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for b in range(0, len(X), bs):
            idx = perm[b:b + bs]
            if len(idx) < 2:
                continue
            opt.zero()
            loss = ag.cross_entropy(model.forward(X[idx]), yy[idx], cw, smoothing=0.05)
            loss.backward()
            extra = [mu * (p.data - g) for p, g in zip(model.params, gparts)] if mu > 0 else None
            opt.step(extra)
    return model.get_flat() - w_glob


# ----------------------------------------------------------------------------- aggregation
def agg_fedavg(U, n, **kw):
    w = n / n.sum()
    return (w[:, None] * U).sum(0), w


def agg_median(U, n, **kw):
    return np.median(U, 0), np.ones(len(U)) / len(U)


def agg_trimmed(U, n, beta=0.2, **kw):
    k = int(np.floor(beta * len(U)))
    S = np.sort(U, 0)
    return S[k:len(U) - k].mean(0), np.ones(len(U)) / len(U)


def agg_krum(U, n, f=2, **kw):
    K = len(U)
    D2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    m = max(1, K - f - 2)
    sc = np.array([np.sort(D2[i][np.arange(K) != i])[:m].sum() for i in range(K)])
    sel = np.argsort(sc)[:max(1, K - f)]  # multi-Krum
    w = np.zeros(K)
    w[sel] = 1 / len(sel)
    return U[sel].mean(0), w


def agg_mata(U, n, tau=3.0, **kw):
    """Median-Anchored Trust-weighted Aggregation (proposed).
    1) robust anchor m = coordinate-wise median; 2) norm clipping to the median norm;
    3) directional trust t_k = exp(-tau(1-cos(u_k, m))) * 1[cos>0]; 4) data-size weighting."""
    m = np.median(U, 0)
    nr = np.linalg.norm(U, axis=1) + 1e-12
    nmed = np.median(nr)
    Uc = U * np.minimum(1.0, nmed / nr)[:, None]
    cos = (Uc @ m) / (np.linalg.norm(Uc, axis=1) * (np.linalg.norm(m) + 1e-12) + 1e-12)
    t = np.exp(-tau * (1 - cos)) * (cos > 0)
    if t.sum() <= 0:
        return m, np.ones(len(U)) / len(U)
    w = n * t
    w = w / w.sum()
    return (w[:, None] * Uc).sum(0), w


AGGS = {"FedAvg": agg_fedavg, "FedProx": agg_fedavg, "Median": agg_median, "TrimmedMean": agg_trimmed,
        "Krum": agg_krum, "MATA": agg_mata}


# ----------------------------------------------------------------------------- driver
def run_fl(fd, sel, model_name, hp, rounds=30, local_epochs=1, bs=128, agg="MATA", mu=None,
           frac=1.0, seed=0, attack=None, n_byz=0, eval_every=5, client_subsample=1.0, local_frac=0.5,
           use_val=False, verbose=False, return_model=False):
    """Simulate cross-silo FL over the clients in `fd` using feature subset `sel`."""
    import data as dmod
    rng = np.random.default_rng(seed)
    C = len(fd.classes)
    A = dmod.prior_adjacency(fd.corr, sel, hp.get("topk", 6))
    model = models.build(model_name, len(sel), C, A, hp, seed=seed)
    w = model.get_flat()
    cw = dmod.class_weights(fd.class_counts, hp.get("cw_pow", 0.5))
    mu = hp.get("mu", 0.0) if mu is None else mu
    if agg == "FedAvg":
        mu = 0.0
    K = len(fd.clients)
    byz = set(range(K - n_byz, K)) if n_byz > 0 else set()  # last clients are adversarial
    Xs, ys = [], []
    for c in fd.clients:
        idx = c["tr"]
        if client_subsample < 1.0:
            idx = rng.choice(idx, max(64, int(len(idx) * client_subsample)), replace=False)
        Xs.append(fd.X[np.ix_(idx, sel)]); ys.append(fd.y[idx])
    nk = np.array([len(v) for v in ys], float)
    if use_val:
        ev_idx = np.concatenate([c["va"] for c in fd.clients])
    else:
        ev_idx = fd.test
    Xe, ye = fd.X[np.ix_(ev_idx, sel)], fd.y[ev_idx]
    hist, wtrace = [], []
    t0 = time.time()
    lr = hp.get("lr", 3e-3)
    for r in range(rounds):
        part = np.arange(K) if frac >= 1 else np.sort(rng.choice(K, max(2, int(frac * K)), replace=False))
        U = []
        for k in part:
            lf = (attack == "labelflip" and k in byz)
            nloc = len(ys[k])
            li = rng.choice(nloc, max(32, int(local_frac * nloc)), replace=False) if local_frac < 1 else np.arange(nloc)
            u = local_update(model, w, Xs[k][li], ys[k][li], local_epochs, bs, lr, mu, cw, rng, label_flip=lf, C=C)
            if k in byz and attack == "signflip":
                u = -3.0 * u
            elif k in byz and attack == "gaussian":
                u = rng.normal(0, 3.0 * np.linalg.norm(u) / np.sqrt(u.size), u.size).astype(np.float32)
            elif k in byz and attack == "scaling":
                u = 10.0 * u
            U.append(u)
        U = np.stack(U)
        g, wk = AGGS[agg](U, nk[part], tau=hp.get("tau", 3.0))
        w = w + g.astype(np.float32)
        wtrace.append(dict(zip(part.tolist(), np.round(wk, 4).tolist())))
        if (r + 1) % eval_every == 0 or r == rounds - 1:
            model.set_flat(w)
            P = model.predict_proba(Xe)
            m = evaluate(ye, P, C)
            m["round"] = r + 1
            m["time"] = time.time() - t0
            hist.append(m)
            if verbose:
                print(f"  r{r+1:3d} acc={m['acc']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f} t={m['time']:.0f}s", flush=True)
    model.set_flat(w)
    res = dict(hist=hist, final=hist[-1], n_params=model.n_params(), weights=wtrace,
               comm_MB=2 * model.n_params() * 4 * K * rounds / 1e6, train_time=time.time() - t0)
    if return_model:
        res["model"] = model
        res["Xe"], res["ye"] = Xe, ye
    return res
