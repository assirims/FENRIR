"""Experiment 3: federated hyper-parameter optimisation of FENRIR with LC-GW2O vs. random search.
Fitness = 1 - macro-F1 on the union of client-held validation sets (computable from aggregated
per-client confusion matrices); the global test set is never touched."""
import json, os, sys, time, functools
import numpy as np
import data, fl, optimizers as O

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BUDGET, ROUNDS, LFRAC = 40, 6, 0.25


DIM = 6


def decode(x):
    # The class-weight exponent is fixed to 0.5 (as for all baselines): client validation sets hold only one
    # sample of the two rarest attacks, so validation macro-F1 cannot measure the effect of re-weighting.
    return dict(lr=float(10 ** (-3.3 + 1.3 * x[0])), D=16 if x[1] < 0.5 else 24, heads=2 if x[2] < 0.5 else 4,
                drop=float(0.3 * x[3]), mu=float(0.05 * x[4]), topk=int(round(3 + 7 * x[5])), cw_pow=0.5, tau=3.0)


def main(method):
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    sel = json.load(open(os.path.join(OUT, "fs_selected.json")))["idx"]
    log = []

    def f(x):
        hp = decode(x)
        t = time.time()
        r = fl.run_fl(fd, sel, "TAGGNN", hp, rounds=ROUNDS, eval_every=ROUNDS, seed=7, agg="MATA",
                      use_val=True, local_frac=LFRAC)
        v = 1 - r["final"]["f1"]
        log.append(dict(hp=hp, val_f1=r["final"]["f1"], time=time.time() - t))
        print(len(log), hp, round(r["final"]["f1"], 4), f"{time.time()-t:.0f}s", flush=True)
        return v

    ob = O.Objective(f, BUDGET)
    rng = np.random.default_rng(2024)
    if method == "LCGW2O":
        O.lcgw2o(ob, DIM, 0.0, 1.0, 6, rng, elite_ls=1)
    else:
        X = rng.random((BUDGET, DIM))
        ob(X)
    res = dict(method=method, best_val_f1=1 - ob.best, best_hp=decode(ob.bx), trace=list(1 - np.array(ob.trace)), log=log)
    json.dump(res, open(os.path.join(OUT, f"hpo_{method}.json"), "w"), indent=1)
    print("best", res["best_val_f1"], res["best_hp"])


if __name__ == "__main__":
    main(sys.argv[1])
