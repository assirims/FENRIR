"""Experiment 2: privacy-preserving federated feature selection with LC-GW2O vs. competitors."""
import json, os, time
import numpy as np
import data, fs, optimizers as O

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N, ITERS, RUNS = 12, 25, 10


def main(algos=None, tag="all"):
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    fit = fs.FedFSFitness(fd, alpha=0.95)
    D = fit.D
    full = np.ones(D, bool)
    res = {"cfg": dict(N=N, budget=N * ITERS, runs=RUNS, alpha=0.95, D=D),
           "all_features": dict(f1=fit.proxy_f1(full), fitness=float(fit(np.ones(D)))), "algos": {}}
    t0 = time.time()
    for an, alg in O.ALGOS.items():
        if algos and an not in algos:
            continue
        runs = []
        for r in range(RUNS):
            fit.cache, fit.evals = {}, 0
            ob = O.Objective(fit, N * ITERS)
            ts = time.time()
            alg(ob, D, 0.0, 1.0, N, np.random.default_rng(500 + r))
            m = fit.mask(ob.bx)
            tr = np.asarray(ob.trace)
            runs.append(dict(fitness=float(ob.best), nfeat=int(m.sum()), f1=float(fit.proxy_f1(m)),
                             mask=m.astype(int).tolist(), time=time.time() - ts,
                             trace=tr[np.linspace(0, len(tr) - 1, 50).astype(int)].tolist()))
        res["algos"][an] = runs
        print(an, "fit=%.4f nfeat=%.1f f1=%.4f" % (np.mean([x["fitness"] for x in runs]),
                                                  np.mean([x["nfeat"] for x in runs]),
                                                  np.mean([x["f1"] for x in runs])), f"{time.time()-t0:.0f}s", flush=True)
    if tag != "all":
        json.dump(res, open(os.path.join(OUT, f"fs_part_{tag}.json"), "w"))
        return
    finalize(res, fd)


def finalize(res, fd):
    best = min(res["algos"]["LCGW2O"], key=lambda x: x["fitness"])
    sel = [i for i, v in enumerate(best["mask"]) if v]
    res["selected"] = dict(idx=sel, names=[fd.features[i] for i in sel], fitness=best["fitness"], f1=best["f1"])
    freq = np.mean([x["mask"] for x in res["algos"]["LCGW2O"]], 0)
    res["lcgw2o_selection_freq"] = dict(zip(fd.features, freq.tolist()))
    # Kuncheva-style pairwise stability (Jaccard) of selected subsets per optimiser
    for an, runs in res["algos"].items():
        Ms = [np.array(x["mask"], bool) for x in runs]
        J = [np.sum(a & b) / max(1, np.sum(a | b)) for i, a in enumerate(Ms) for b in Ms[i + 1:]]
        res.setdefault("stability_jaccard", {})[an] = float(np.mean(J))
    json.dump(res, open(os.path.join(OUT, "fs.json"), "w"))
    json.dump(res["selected"], open(os.path.join(OUT, "fs_selected.json"), "w"), indent=1)
    print("selected", len(sel), res["selected"]["names"])


def merge():
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    parts = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith("fs_part_")]
    res = parts[0]
    for p in parts[1:]:
        res["algos"].update(p["algos"])
    res["algos"] = {k: res["algos"][k] for k in O.ALGOS if k in res["algos"]}
    finalize(res, fd)


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "merge":
        merge()
    else:
        main(sys.argv[2].split(","), sys.argv[1])
