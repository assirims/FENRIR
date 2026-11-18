"""Experiment 9: parameters, multiply-accumulate operations (MACs), CPU latency and communication cost."""
import json, os, time
import numpy as np
import autograd as ag
import models, data
from exp_fl import fenrir_hp, BASE_HP, selected

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def count_macs(model, d):
    orig = ag.matmul
    tot = [0]

    def counting(a, b):
        out = orig(a, b)
        A = a.data if isinstance(a, ag.Tensor) else np.asarray(a)
        tot[0] += int(np.prod(out.shape)) * int(A.shape[-1])
        return out
    ag.matmul = counting
    try:
        model.train = False
        model.forward(np.zeros((1, d), np.float32))
    finally:
        ag.matmul = orig
        model.train = True
    return tot[0]


def latency(model, d, bs, reps=5):
    X = np.random.default_rng(0).normal(size=(bs, d)).astype(np.float32)
    model.train = False
    model.forward(X)
    t = time.perf_counter()
    for _ in range(reps):
        model.forward(X)
    model.train = True
    return (time.perf_counter() - t) / reps / bs * 1000.0


def main():
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    sel = selected()
    C = len(fd.classes)
    out = {}
    hpF = fenrir_hp()
    for name, mname, hp, cols in [("FENRIR (TAG-GNN)", "TAGGNN", hpF, sel), ("MLP", "MLP", BASE_HP, sel),
                                  ("CNN1D", "CNN1D", BASE_HP, sel), ("GCN", "GCN", BASE_HP, sel),
                                  ("GAT", "GAT", BASE_HP, sel), ("Transformer", "Transformer", BASE_HP, sel),
                                  ("TAG-GNN, all 92 features", "TAGGNN", hpF, list(range(fd.X.shape[1])))]:
        d = len(cols)
        A = data.prior_adjacency(fd.corr, cols, hp.get("topk", 6))
        m = models.build(mname, d, C, A, hp, seed=0)
        P = m.n_params()
        r = dict(params=P, macs=count_macs(m, d), size_kb=P * 4 / 1024,
                 lat_b256_ms=latency(m, d, 256), lat_b1_ms=latency(m, d, 1, reps=50),
                 comm_round_MB=2 * P * 4 * 10 / 1e6, d=d)
        out[name] = r
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "complexity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
