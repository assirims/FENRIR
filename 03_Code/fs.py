"""
Privacy-preserving federated wrapper feature selection.
The server runs a population-based optimiser over binary feature masks; each client scores a
candidate mask on its own data (local proxy learner trained on local-train, scored on local-val)
and returns only a scalar macro-F1. No raw data or gradients leave the clients.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


class FedFSFitness:
    def __init__(self, fd, alpha=0.95, max_tr=2500, seed=0):
        rng = np.random.default_rng(seed)
        self.alpha, self.D = alpha, fd.X.shape[1]
        self.cl = []
        for c in fd.clients:
            tr = c["tr"]
            if len(tr) > max_tr:
                tr = np.sort(rng.choice(tr, max_tr, replace=False))
            va = c["va"]
            if len(va) == 0:
                continue
            self.cl.append((fd.X[tr], fd.y[tr], fd.X[va], fd.y[va], len(c["tr"])))
        self.cache, self.evals = {}, 0

    def mask(self, x):
        m = x > 0.5
        if not m.any():
            m[np.argmax(x)] = True
        return m

    def proxy_f1(self, m):
        f, n = 0.0, 0
        for Xt, yt, Xv, yv, nk in self.cl:
            clf = DecisionTreeClassifier(min_samples_leaf=2, random_state=0, class_weight="balanced")
            clf.fit(Xt[:, m], yt)
            p = clf.predict(Xv[:, m])
            f += nk * f1_score(yv, p, average="macro", labels=np.unique(yv), zero_division=0)
            n += nk
        return f / n

    def __call__(self, x):
        m = self.mask(x)
        key = m.tobytes()
        if key not in self.cache:
            self.evals += 1
            f1 = self.proxy_f1(m)
            self.cache[key] = self.alpha * (1 - f1) + (1 - self.alpha) * m.sum() / self.D
        return self.cache[key]
