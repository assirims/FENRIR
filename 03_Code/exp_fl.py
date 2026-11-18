"""Experiments 4-8: federated detection (main comparison, ablation, Byzantine robustness, non-IID
sensitivity, centralised / local-only references). Usage:
    python exp_fl.py <group> <worker_id> <n_workers>
Each job writes results/runs/<group>/<job>.json (resumable)."""
import json, os, sys, time
import numpy as np
import data, fl, models, autograd as ag

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEEDS = [0, 1, 2]
BASE_HP = dict(D=24, heads=4, drop=0.1, lr=3e-3, cw_pow=0.5, topk=6, mu=0.0, tau=3.0)


def fenrir_hp():
    p = os.path.join(RES, "hpo_LCGW2O.json")
    hp = json.load(open(p))["best_hp"] if os.path.exists(p) else dict(BASE_HP, heads=2, mu=0.01)
    return hp


def selected():
    return json.load(open(os.path.join(RES, "fs_selected.json")))["idx"]


def jobs(group):
    J = []
    if group == "main":
        for s in SEEDS:
            J.append((f"FENRIR_s{s}", dict(model="TAGGNN", hp="fenrir", agg="MATA", seed=s, rounds=30)))
            for m in ["MLP", "CNN1D", "GCN", "GAT", "Transformer"]:
                J.append((f"{m}_s{s}", dict(model=m, hp="base", agg="FedAvg", seed=s, rounds=30)))
    elif group == "ablation":
        V = {"full": {}, "noHPO": {"hp": "base_tag"}, "noG": {"flags": {"use_G": False}},
             "noS": {"flags": {"use_S": False}}, "noGate": {"flags": {"gate": False}},
             "noChan": {"flags": {"chan_att": False}}, "priorOnly": {"flags": {"adj": "prior"}},
             "adaptOnly": {"flags": {"adj": "adaptive"}}, "noPeriodic": {"flags": {"periodic": False}},
             "noDual": {"flags": {"dual_view": False}}, "FedAvgAgg": {"agg": "FedAvg"}}
        for s in [0, 1, "noFS", 2]:  # seeds 0-1 first, then the costly all-feature variant, then seed 2
            if s == "noFS":
                J.append(("noFS_s0", dict(model="TAGGNN", hp="fenrir", agg="MATA", seed=0, rounds=20, allfeat=True)))
                continue
            for v, c in V.items():
                cfg = dict(model="TAGGNN", hp="fenrir", agg="MATA", seed=s, rounds=20)
                cfg.update(c)
                J.append((f"{v}_s{s}", cfg))
    elif group == "robust":
        for atk in ["none", "signflip", "gaussian", "labelflip"]:
            for agg in ["FedAvg", "Median", "TrimmedMean", "Krum", "MATA"]:
                J.append((f"{agg}_{atk}", dict(model="TAGGNN", hp="fenrir", agg=agg, seed=0, rounds=20,
                                               attack=None if atk == "none" else atk, n_byz=0 if atk == "none" else 3)))
    elif group == "noniid":
        for a in ["0.1", "0.3", "1.0", "iid"]:
            J.append((f"FENRIR_a{a}", dict(model="TAGGNN", hp="fenrir", agg="MATA", seed=0, rounds=30, alpha=a)))
            J.append((f"MLP_a{a}", dict(model="MLP", hp="base", agg="FedAvg", seed=0, rounds=30, alpha=a)))
            J.append((f"Transformer_a{a}", dict(model="Transformer", hp="base", agg="FedAvg", seed=0, rounds=30, alpha=a)))
        J.append(("FENRIRfedavg_a0.1", dict(model="TAGGNN", hp="fenrir", agg="FedAvg", seed=0, rounds=30, alpha="0.1")))
    elif group == "central":
        for s in SEEDS:
            J.append((f"central_TAGGNN_s{s}", dict(kind="central", model="TAGGNN", hp="fenrir", seed=s, epochs=15)))
            J.append((f"central_MLP_s{s}", dict(kind="central", model="MLP", hp="base", seed=s, epochs=15)))
        J.append(("local_TAGGNN_s0", dict(kind="local", model="TAGGNN", hp="fenrir", seed=0, epochs=15)))
        J.append(("classic_ml", dict(kind="classic")))
    return J


def get_hp(tag):
    if tag == "fenrir":
        return dict(fenrir_hp())
    if tag == "base_tag":
        return dict(BASE_HP, mu=0.0)
    return dict(BASE_HP)


def run_central(fd, sel, cfg, local=False):
    C = len(fd.classes)
    hp = get_hp(cfg["hp"])
    rng = np.random.default_rng(cfg["seed"])
    A = data.prior_adjacency(fd.corr, sel, hp.get("topk", 6))
    cw = data.class_weights(fd.class_counts, hp.get("cw_pow", 0.5))
    Xe, ye = fd.X[np.ix_(fd.test, sel)], fd.y[fd.test]
    groups = [np.concatenate([c["tr"] for c in fd.clients])] if not local else [c["tr"] for c in fd.clients]
    out = []
    for gi, idx in enumerate(groups):
        m = models.build(cfg["model"], len(sel), C, A, hp, seed=cfg["seed"] + gi)
        w = m.get_flat()
        X, y = fd.X[np.ix_(idx, sel)], fd.y[idx]
        hist = []
        t0 = time.time()
        for ep in range(cfg["epochs"]):
            w = w + fl.local_update(m, w, X, y, 1, 128, hp.get("lr", 3e-3), 0.0, cw, rng)
            if (ep + 1) % 5 == 0 or ep == cfg["epochs"] - 1:
                m.set_flat(w)
                r = fl.evaluate(ye, m.predict_proba(Xe), C)
                r["epoch"] = ep + 1
                hist.append(r)
        out.append(dict(hist=hist, final=hist[-1], train_time=time.time() - t0, n_params=m.n_params()))
        print("  group", gi, {k: round(v, 4) for k, v in hist[-1].items()}, flush=True)
    if local:
        keys = out[0]["final"].keys()
        return dict(per_client=[o["final"] for o in out],
                    final={k: float(np.mean([o["final"][k] for o in out])) for k in keys},
                    final_std={k: float(np.std([o["final"][k] for o in out])) for k in keys})
    return out[0]


def run_classic(fd, sel):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    C = len(fd.classes)
    tr = np.concatenate([c["tr"] for c in fd.clients])
    res = {}
    for fs_name, cols in [("selected", sel), ("all", list(range(fd.X.shape[1])))]:
        Xt, yt = fd.X[np.ix_(tr, cols)], fd.y[tr]
        Xe, ye = fd.X[np.ix_(fd.test, cols)], fd.y[fd.test]
        for nm, clf in [("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                        ("RandomForest", RandomForestClassifier(300, n_jobs=1, random_state=0, class_weight="balanced")),
                        ("ExtraTrees", ExtraTreesClassifier(300, n_jobs=1, random_state=0, class_weight="balanced")),
                        ("HistGB", HistGradientBoostingClassifier(random_state=0, class_weight="balanced"))]:
            t = time.time()
            clf.fit(Xt, yt)
            P = np.zeros((len(Xe), C))
            P[:, clf.classes_] = clf.predict_proba(Xe)
            r = fl.evaluate(ye, P, C)
            r["time"] = time.time() - t
            res[f"{nm}_{fs_name}"] = r
            print(nm, fs_name, {k: round(v, 4) for k, v in r.items()}, flush=True)
    return res


def main(group, wid, nw, rev=False):
    outdir = os.path.join(RES, "runs", group)
    os.makedirs(outdir, exist_ok=True)
    sel_fs = selected()
    cache = {}
    JJ = list(enumerate(jobs(group)))
    for j, (name, cfg) in (reversed(JJ) if rev else JJ):
        if j % nw != wid:
            continue
        fp = os.path.join(outdir, name + ".json")
        lock = fp + ".lock"
        if os.path.exists(fp) or os.path.exists(lock):
            continue
        open(lock, "w").write(str(os.getpid()))
        alpha = cfg.get("alpha", "0.5")
        a = None if alpha == "iid" else float(alpha)
        if alpha not in cache:
            cache[alpha] = data.prepare(K=10, alpha=a, seed=42)
        fd = cache[alpha]
        sel = list(range(fd.X.shape[1])) if cfg.get("allfeat") else sel_fs
        print(f"[{group}] {name} start", flush=True)
        t0 = time.time()
        kind = cfg.get("kind", "fl")
        if kind == "central":
            r = run_central(fd, sel, cfg)
        elif kind == "local":
            r = run_central(fd, sel, cfg, local=True)
        elif kind == "classic":
            r = run_classic(fd, sel)
        else:
            hp = get_hp(cfg["hp"])
            if "flags" in cfg:
                hp["flags"] = cfg["flags"]
            r = fl.run_fl(fd, sel, cfg["model"], hp, rounds=cfg["rounds"], agg=cfg["agg"], seed=cfg["seed"],
                          attack=cfg.get("attack"), n_byz=cfg.get("n_byz", 0), eval_every=5, return_model=True)
            m = r.pop("model")
            Xe, ye = r.pop("Xe"), r.pop("ye")
            P = m.predict_proba(Xe)
            r["pred"] = P.argmax(1).astype(int).tolist()
            if group == "main" and name.startswith("FENRIR"):
                np.save(os.path.join(outdir, name + "_proba.npy"), P.astype(np.float32))
                np.save(os.path.join(outdir, name + "_weights.npy"), m.get_flat())
            r["hp"] = {k: v for k, v in hp.items()}
        r["cfg"], r["wall"] = cfg, time.time() - t0
        json.dump(r, open(fp, "w"))
        if os.path.exists(lock):
            os.remove(lock)
        fin = r.get("final", {})
        print(f"[{group}] {name} done {r['wall']:.0f}s", {k: round(v, 4) for k, v in fin.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), rev=len(sys.argv) > 4)
