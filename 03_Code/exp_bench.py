"""Experiment 1: numerical analysis of LC-GW2O on shifted benchmark functions (+ component ablation)."""
import json, os, time, functools
import numpy as np
from scipy import stats
import optimizers as O

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
D, N, BUDGET, RUNS, ABL_RUNS = 30, 30, 15000, 20, 10


def trace_points(tr, npts=60):
    tr = np.asarray(tr)
    idx = np.linspace(0, len(tr) - 1, npts).astype(int)
    return tr[idx].tolist()


def run(alg, f, runs):
    finals, traces = [], []
    for r in range(runs):
        ob = O.Objective(f["f"], BUDGET)
        alg(ob, D, f["lb"], f["ub"], N, np.random.default_rng(1000 + r))
        finals.append(float(ob.best))
        traces.append(trace_points(ob.trace))
    return finals, np.mean(traces, 0).tolist()


def main():
    S = O.benchmark_suite(D)
    res = {"cfg": dict(D=D, N=N, budget=BUDGET, runs=RUNS), "main": {}, "ablation": {}}
    t0 = time.time()
    for fn, f in S.items():
        res["main"][fn] = {}
        for an, alg in O.ALGOS.items():
            fin, tr = run(alg, f, RUNS)
            res["main"][fn][an] = dict(finals=fin, mean=float(np.mean(fin)), std=float(np.std(fin)),
                                       best=float(np.min(fin)), trace=tr)
        # Wilcoxon signed-rank: proposed vs each competitor
        for an in O.ALGOS:
            if an == "LCGW2O":
                continue
            a, b = np.array(res["main"][fn]["LCGW2O"]["finals"]), np.array(res["main"][fn][an]["finals"])
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = 1.0
            sign = "+" if (p < 0.05 and np.median(a) < np.median(b)) else ("-" if p < 0.05 else "=")
            res["main"][fn][an]["wilcoxon_p"], res["main"][fn][an]["sign"] = float(p), sign
        print(fn, {k: round(v["mean"], 4) for k, v in res["main"][fn].items()}, f"{time.time()-t0:.0f}s", flush=True)
    # Friedman test on mean errors
    algs = list(O.ALGOS)
    M = np.array([[res["main"][fn][a]["mean"] for a in algs] for fn in S])
    ranks = np.array([stats.rankdata(r) for r in M]).mean(0)
    fr = stats.friedmanchisquare(*M.T)
    res["friedman"] = dict(algos=algs, avg_rank=ranks.tolist(), stat=float(fr.statistic), p=float(fr.pvalue))
    print("Friedman", dict(zip(algs, np.round(ranks, 2))), fr.pvalue, flush=True)
    # component ablation
    variants = {"LC-GW2O (full)": O.lcgw2o,
                "w/o Levy flight": functools.partial(O.lcgw2o, use_levy=False),
                "w/o chaotic maps": functools.partial(O.lcgw2o, use_chaos=False),
                "w/o opposition learning": functools.partial(O.lcgw2o, use_obl=False),
                "w/o whale spiral": functools.partial(O.lcgw2o, use_spiral=False),
                "w/o inheritance crossover": functools.partial(O.lcgw2o, cr=0.0),
                "w/o elite local search": functools.partial(O.lcgw2o, elite_ls=0)}
    for vn, alg in variants.items():
        res["ablation"][vn] = {}
        for fn, f in S.items():
            fin, _ = run(alg, f, ABL_RUNS)
            res["ablation"][vn][fn] = dict(mean=float(np.mean(fin)), std=float(np.std(fin)))
        print(vn, {k: round(v["mean"], 3) for k, v in res["ablation"][vn].items()}, flush=True)
    V = list(variants)
    M2 = np.array([[res["ablation"][v][fn]["mean"] for v in V] for fn in S])
    res["ablation_rank"] = dict(zip(V, np.array([stats.rankdata(r) for r in M2]).mean(0).tolist()))
    json.dump(res, open(os.path.join(OUT, "bench.json"), "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
