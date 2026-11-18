"""Builds every results table / number used in the manuscript directly from results/*.json, so the
paper text and the raw results cannot diverge. Writes paper/generated.json (placeholder -> text)."""
import json, os, glob, sys
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, f1_score
import data

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.join(HERE, "..", "paper")
sys.path.insert(0, PAPER)
import static_blocks as SB  # noqa

G = {}


def pm(v, d=4, pct=False):
    v = np.asarray(v, float)
    if pct:
        return f"{100*v.mean():.2f} ± {100*v.std():.2f}"
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}"


def sci(x):
    if x == 0:
        return "0"
    if abs(x) >= 1000 or abs(x) < 1e-2:
        return f"{x:.2e}"
    return f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"


def load_group(group):
    R = {}
    for f in glob.glob(os.path.join(RES, "runs", group, "*.json")):
        R[os.path.basename(f)[:-5]] = json.load(open(f))
    return R


SHORT = {"ARP_poisioning": "ARP poisoning", "DDOS_Slowloris": "DDoS Slowloris", "DOS_SYN_Hping": "DoS SYN flood",
         "MQTT_Publish": "MQTT publish (benign)", "Metasploit_Brute_Force_SSH": "SSH brute force (Metasploit)",
         "NMAP_FIN_SCAN": "Nmap FIN scan", "NMAP_OS_DETECTION": "Nmap OS detection", "NMAP_TCP_scan": "Nmap TCP scan",
         "NMAP_UDP_SCAN": "Nmap UDP scan", "NMAP_XMAS_TREE_SCAN": "Nmap XMAS-tree scan",
         "Thing_Speak": "ThingSpeak (benign)", "Wipro_bulb": "Wipro bulb (benign)"}


def table3(fd):
    import pandas as pd
    raw = pd.read_csv(os.path.join(HERE, "..", "data", "RT_IOT2022"))["Attack_type"].value_counts()
    C = len(fd.classes)
    tot = np.bincount(fd.y, minlength=C)
    te = np.bincount(fd.y[fd.test], minlength=C)
    tr = sum(np.bincount(fd.y[c["tr"]], minlength=C) for c in fd.clients)
    va = sum(np.bincount(fd.y[c["va"]], minlength=C) for c in fd.clients)
    ncl = [sum(1 for c in fd.clients if np.any(fd.y[c["tr"]] == k)) for k in range(C)]
    rows = ["Table: **Table 3.** RT-IoT2022 class distribution, processed counts and splits (clients = number of the 10 clients holding the class).", "",
            "| Class | Type | Original | Processed | Train (clients) | Validation (clients) | Test | Clients |",
            "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for k, c in enumerate(fd.classes):
        typ = "Benign" if c in ("MQTT_Publish", "Thing_Speak", "Wipro_bulb") else "Attack"
        rows.append(f"| {SHORT[c]} | {typ} | {raw[c]:,} | {tot[k]:,} | {tr[k]:,} | {va[k]:,} | {te[k]:,} | {ncl[k]} |")
    rows.append(f"| **Total** | | **{raw.sum():,}** | **{tot.sum():,}** | **{tr.sum():,}** | **{va.sum():,}** | **{te.sum():,}** | |")
    G["TABLE3"] = "\n".join(rows)
    G["N_TRAIN"], G["N_VAL"], G["N_TEST"] = f"{tr.sum():,}", f"{va.sum():,}", f"{te.sum():,}"


def table_bench():
    B = json.load(open(os.path.join(RES, "bench.json")))
    algs = ["PSO", "DE", "GWO", "WOA", "HHO", "LCGW2O"]
    rows = ["Table: **Table 5.** Mean ± standard deviation of the best error on ten shifted benchmark functions (D = 30, NFE = 15,000, 20 runs). Signs give the Wilcoxon signed-rank outcome of LC-GW2O against each competitor at α = 0.05 (+ better, = no significant difference, − worse). Best mean in bold.", "",
            "| Function | " + " | ".join(a if a != "LCGW2O" else "LC-GW2O" for a in algs) + " |",
            "|:--|" + "--:|" * len(algs)]
    wins = {a: [0, 0, 0] for a in algs if a != "LCGW2O"}
    for fn, R in B["main"].items():
        means = [R[a]["mean"] for a in algs]
        best = int(np.argmin(means))
        cells = []
        for i, a in enumerate(algs):
            s = f"{sci(R[a]['mean'])} ± {sci(R[a]['std'])}"
            if i == best:
                s = f"**{s}**"
            if a != "LCGW2O":
                sg = R[a]["sign"]
                s += f" ({sg})"
                wins[a][{"+": 0, "=": 1, "-": 2}[sg]] += 1
            cells.append(s)
        rows.append(f"| {fn} | " + " | ".join(cells) + " |")
    fr = B["friedman"]
    rows.append("| **Friedman mean rank** | " + " | ".join(f"**{r:.2f}**" if a == "LCGW2O" else f"{r:.2f}" for a, r in zip(fr["algos"], fr["avg_rank"])) + " |")
    rows.append("| W/T/L of LC-GW2O | " + " | ".join(f"{wins[a][0]}/{wins[a][1]}/{wins[a][2]}" if a != "LCGW2O" else "–" for a in algs) + " |")
    G["TABLE5"] = "\n".join(rows)
    mant, ex = f"{fr['p']:.2e}".split("e")
    G["FRIEDMAN_P"] = f"{mant}\\times10^{{{int(ex)}}}"
    G["FRIEDMAN_STAT"] = f"{fr['stat']:.2f}"
    G["BENCH_RANK"] = f"mean rank {fr['avg_rank'][-1]:.2f} vs. {min(fr['avg_rank'][:-1]):.2f} for the runner-up"
    G["BENCH_WTL"] = "; ".join(f"{a}: {wins[a][0]}/{wins[a][1]}/{wins[a][2]}" for a in wins)
    G["_bench"] = B
    # ablation
    A = B["ablation"]
    fns = list(B["main"].keys())
    V = list(A.keys())
    SHORTV = {"LC-GW2O (full)": "Full", "w/o Levy flight": "−Lévy", "w/o chaotic maps": "−Chaos",
              "w/o opposition learning": "−OBL", "w/o whale spiral": "−Spiral", "w/o inheritance crossover": "−Crossover",
              "w/o elite local search": "−Elite LS"}
    rows = ["Table: **Table 6.** Component ablation of LC-GW2O on the ten benchmark functions (mean best error over 10 runs) and Friedman mean rank (lower is better). −Lévy: without Lévy flights; −Chaos: uniform instead of logistic-chaotic sequences; −OBL: without opposition-based initialisation and quasi-opposition restarts; −Spiral: without the whale spiral (pure encircling); −Crossover: without inheritance crossover; −Elite LS: without elite local search.", "",
            "| Variant | " + " | ".join(f.split(" ")[0] for f in fns) + " | Mean rank |", "|:--|" + "--:|" * (len(fns) + 1)]
    for v in V:
        r = B["ablation_rank"][v]
        rows.append(f"| {SHORTV.get(v, v)} | " + " | ".join(sci(A[v][f]["mean"]) for f in fns) + f" | {'**' if v.startswith('LC') else ''}{r:.2f}{'**' if v.startswith('LC') else ''} |")
    G["TABLE6"] = "\n".join(rows)


def table_fs(fd):
    S = json.load(open(os.path.join(RES, "fs.json")))
    algs = ["PSO", "DE", "GWO", "WOA", "HHO", "LCGW2O"]
    ref = np.array([r["fitness"] for r in S["algos"]["LCGW2O"]])
    rows = ["Table: **Table 7.** Federated wrapper feature selection (10 runs per optimiser, NFE = 300, N = 12). Fitness is Eq. (2) (lower is better); proxy macro-F1 is the client-weighted validation macro-F1 of the decision-tree proxy; stability is the mean pairwise Jaccard index of the selected subsets; p is the two-sided Mann–Whitney test of fitness against LC-GW2O.", "",
            "| Optimiser | Fitness | Best fitness | Selected features | Proxy macro-F1 | Stability (Jaccard) | Time per run (s) | p vs. LC-GW2O |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for a in algs:
        R = S["algos"][a]
        f = np.array([r["fitness"] for r in R])
        pv = stats.mannwhitneyu(ref, f).pvalue
        p = "–" if a == "LCGW2O" else ("< 0.001" if pv < 0.001 else f"{pv:.3f}")
        name = "**LC-GW2O**" if a == "LCGW2O" else a
        rows.append(f"| {name} | {pm(f)} | {f.min():.4f} | {pm([r['nfeat'] for r in R], 1)} | {pm([r['f1'] for r in R])} | "
                    f"{S['stability_jaccard'][a]:.3f} | {np.mean([r['time'] for r in R]):.1f} | {p} |")
    rows.append(f"| All 92 features (no selection) | {S['all_features']['fitness']:.4f} | – | 92 | {S['all_features']['f1']:.4f} | – | – | – |")
    G["TABLE7"] = "\n".join(rows)
    G["NSEL"] = str(len(S["selected"]["idx"]))
    G["SEL_NAMES"] = ", ".join(f"`{n}`" for n in S["selected"]["names"])
    G["FS_FIT"] = f"{S['selected']['fitness']:.4f}"
    G["FS_F1"] = f"{S['selected']['f1']:.4f}"
    G["FS_ALLF1"] = f"{S['all_features']['f1']:.4f}"
    G["FS_LC_MEAN"] = pm(ref)
    G["_fs"] = S


def table4():
    H = json.load(open(os.path.join(RES, "hpo_LCGW2O.json")))
    RSH = json.load(open(os.path.join(RES, "hpo_RS.json"))) if os.path.exists(os.path.join(RES, "hpo_RS.json")) else None
    b = H["best_hp"]
    rows = ["Table: **Table 4.** Hyper-parameters. The FENRIR column is selected by federated LC-GW2O tuning over a six-dimensional space (NFE = 40); baselines use the shared default configuration.", "",
            "| Hyper-parameter | Search space (LC-GW2O) | FENRIR (selected) | Baselines (default) |", "|:--|:--|:--|:--|",
            f"| Learning rate η (Adam) | $10^{{[-3.3,\\,-2.0]}}$ (log) | {b['lr']:.2e} | 3.00e-03 |",
            f"| Hidden width $D_h$ | {{16, 24}} | {b['D']} | 24 |",
            f"| Attention heads $H$ | {{2, 4}} | {b['heads']} | 4 |",
            f"| Dropout | [0, 0.3] | {b['drop']:.3f} | 0.1 |",
            f"| Proximal coefficient μ | [0, 0.05] | {b['mu']:.4f} | 0 (FedAvg) |",
            f"| Prior sparsity κ (top-k) | {{3, …, 10}} | {b['topk']} | 6 |",
            "| Class-weight exponent | fixed (not tuned) | 0.5 | 0.5 |",
            "| TAG-GNN blocks $L$, frequencies $n_f$, adaptive rank | fixed | 2, 4, 8 | – |",
            "| MATA temperature $\\tau_m$ | fixed | 3 | – |",
            "| Rounds / local epochs / client fraction per round | fixed | 30 / 1 / 50 % of data | same |",
            "| Batch size, label smoothing, gradient clip | fixed | 128, 0.05, 5 | same |"]
    G["TABLE4"] = "\n".join(rows)
    G["HPO_BEST"] = f"{H['best_val_f1']:.4f}"
    if RSH:
        G["HPO_RS_BEST"] = f"{RSH['best_val_f1']:.4f}"
        G["HPO_RS_MEAN"] = f"{np.mean([x['val_f1'] for x in RSH['log']]):.4f}"
        G["HPO_LC_MEAN"] = f"{np.mean([x['val_f1'] for x in H['log']]):.4f}"


KEYS = [("acc", "Accuracy"), ("prec", "Macro-P"), ("rec", "Macro-R"), ("f1", "Macro-F1"), ("f1w", "Weighted-F1"),
        ("mcc", "MCC"), ("auc", "Macro-AUC")]


def table_main(fd):
    R = load_group("main")
    order = ["MLP", "CNN1D", "GCN", "GAT", "Transformer", "FENRIR"]
    nice = {"MLP": "FedAvg-MLP", "CNN1D": "FedAvg-CNN1D", "GCN": "FedAvg-GCN", "GAT": "FedAvg-GAT",
            "Transformer": "FedAvg-Transformer", "FENRIR": "**FENRIR (TAG-GNN + MATA)**"}
    rows = ["Table: **Table 8.** Federated detection on the RT-IoT2022 global test set (10 non-IID clients, α = 0.5, 30 rounds; mean ± std over three seeds, metrics in %). Comm. = total client↔server traffic over training.", "",
            "| Model | " + " | ".join(n for _, n in KEYS) + " | Params | Comm. (MB) |", "|:--|" + "--:|" * (len(KEYS) + 2)]
    best = {k: max(np.mean([R[f"{m}_s{s}"]["final"][k] for s in range(3) if f"{m}_s{s}" in R] or [0]) for m in order) for k, _ in KEYS}
    stats_ = {}
    for m in order:
        runs = [R[f"{m}_s{s}"] for s in range(3) if f"{m}_s{s}" in R]
        if not runs:
            continue
        cells = []
        for k, _ in KEYS:
            v = np.array([r["final"][k] for r in runs])
            s = pm(v, pct=True)
            if abs(v.mean() - best[k]) < 1e-12:
                s = f"**{s}**"
            cells.append(s)
        stats_[m] = {k: np.array([r["final"][k] for r in runs]) for k, _ in KEYS}
        rows.append(f"| {nice[m]} | " + " | ".join(cells) + f" | {runs[0]['n_params']:,} | {runs[0]['comm_MB']:.1f} |")
    G["TABLE8"] = "\n".join(rows)
    F = stats_["FENRIR"]
    G["F_ACC"] = f"{100*F['acc'].mean():.2f} %"
    G["F_F1"] = f"{100*F['f1'].mean():.2f} %"
    G["F_MCC"] = f"{100*F['mcc'].mean():.2f} %"
    G["F_AUC"] = f"{100*F['auc'].mean():.2f} %"
    G["F_F1W"] = f"{100*F['f1w'].mean():.2f} %"
    bl = {m: stats_[m]["f1"].mean() for m in stats_ if m != "FENRIR"}
    bm = max(bl, key=bl.get)
    G["BEST_BASE"] = nice[bm]
    G["BEST_BASE_F1"] = f"{100*bl[bm]:.2f} %"
    G["GAIN_F1"] = f"{100*(F['f1'].mean()-bl[bm]):.2f}"
    G["BEST_BASE_ACC"] = f"{100*stats_[bm]['acc'].mean():.2f} %"
    G["BEST_BASE_MCC"] = f"{100*stats_[bm]['mcc'].mean():.2f} %"
    G["_main"] = (R, stats_)
    # per-class + significance
    y = fd.y[fd.test]
    C = len(fd.classes)
    pc = {}
    for m in order:
        runs = [R[f"{m}_s{s}"] for s in range(3) if f"{m}_s{s}" in R]
        pc[m] = np.mean([f1_score(y, np.array(r["pred"]), average=None, labels=range(C), zero_division=0) for r in runs], 0)
    G["_perclass"] = pc
    P_, R_, F_, S_ = [], [], [], []
    for s in range(3):
        p, r, f, sup = precision_recall_fscore_support(y, np.array(R[f"FENRIR_s{s}"]["pred"]), labels=range(C), zero_division=0)
        P_.append(p); R_.append(r); F_.append(f); S_ = sup
    P_, R_, F_ = map(np.array, (P_, R_, F_))
    rows = ["Table: **Table 10.** Per-class precision, recall and F1 of FENRIR on the global test set (%, mean ± std over three seeds) and the best baseline's F1 for reference.", "",
            f"| Class | Support | Precision | Recall | F1 | F1 of {nice[bm]} |", "|:--|--:|--:|--:|--:|--:|"]
    for k, c in enumerate(fd.classes):
        rows.append(f"| {SHORT[c]} | {S_[k]} | {pm(P_[:, k], pct=True)} | {pm(R_[:, k], pct=True)} | {pm(F_[:, k], pct=True)} | {100*pc[bm][k]:.2f} |")
    G["TABLE10"] = "\n".join(rows)
    worst = int(np.argmin(F_.mean(0)))
    G["WORST_CLASS"] = SHORT[fd.classes[worst]]
    G["WORST_CLASS_F1"] = f"{100*F_.mean(0)[worst]:.2f} %"
    G["WORST_CLASS_SUP"] = str(S_[worst])
    n_perfect = int(np.sum(F_.mean(0) >= 0.995))
    G["N_PERFECT"] = str(n_perfect)
    # McNemar + bootstrap vs each baseline (seed-0 predictions; FENRIR seed 0)
    rng = np.random.default_rng(0)
    pf = np.array(R["FENRIR_s0"]["pred"])
    B = 2000
    idx = rng.integers(0, len(y), (B, len(y)))
    rows = ["Table: **Table 15.** Statistical comparison of FENRIR with each federated baseline on the test set (seed 0): McNemar's test on paired predictions and a 2,000-sample bootstrap of the macro-F1 difference.", "",
            "| Comparison | n₀₁ (only FENRIR correct) | n₁₀ (only baseline correct) | McNemar χ² | p-value | ΔMacro-F1 (pp) | 95 % bootstrap CI (pp) |",
            "|:--|--:|--:|--:|--:|--:|--:|"]
    for m in order[:-1]:
        pb = np.array(R[f"{m}_s0"]["pred"])
        a = np.sum((pf == y) & (pb != y)); b = np.sum((pf != y) & (pb == y))
        chi = (abs(a - b) - 1) ** 2 / max(1, a + b)
        p = stats.chi2.sf(chi, 1)
        d0 = f1_score(y, pf, average="macro") - f1_score(y, pb, average="macro")
        ds = np.array([f1_score(y[i], pf[i], average="macro") - f1_score(y[i], pb[i], average="macro") for i in idx])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        rows.append(f"| FENRIR vs. {nice[m]} | {a} | {b} | {chi:.2f} | {p:.2e} | {100*d0:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
        if m == "MLP":
            G["MLP_CI"] = f"[{100*lo:+.2f}, {100*hi:+.2f}]"
    G["TABLE15"] = "\n".join(rows)


def table_refs():
    R = load_group("central")
    M = G["_main"][1]
    rows = ["Table: **Table 9.** Reference settings (%, mean ± std over seeds where applicable). Centralised training pools all client data and is therefore not privacy-preserving; local-only models are trained on a single client and averaged over the ten clients.", "",
            "| Setting | Model | Features | Accuracy | Macro-F1 | MCC | Privacy-preserving |", "|:--|:--|--:|--:|--:|--:|:--:|"]
    F = M["FENRIR"]
    rows.append(f"| Federated (ours) | FENRIR | {G['NSEL']} | {pm(F['acc'], pct=True)} | {pm(F['f1'], pct=True)} | {pm(F['mcc'], pct=True)} | Yes |")
    for m in ["TAGGNN", "MLP"]:
        runs = [R[f"central_{m}_s{s}"] for s in range(3) if f"central_{m}_s{s}" in R]
        if runs:
            rows.append(f"| Centralised | {'TAG-GNN' if m == 'TAGGNN' else 'MLP'} | {G['NSEL']} | {pm([r['final']['acc'] for r in runs], pct=True)} | "
                        f"{pm([r['final']['f1'] for r in runs], pct=True)} | {pm([r['final']['mcc'] for r in runs], pct=True)} | No |")
    if "local_TAGGNN_s0" in R:
        L = R["local_TAGGNN_s0"]
        rows.append(f"| Local-only (per client) | TAG-GNN | {G['NSEL']} | {100*L['final']['acc']:.2f} ± {100*L['final_std']['acc']:.2f} | "
                    f"{100*L['final']['f1']:.2f} ± {100*L['final_std']['f1']:.2f} | {100*L['final']['mcc']:.2f} ± {100*L['final_std']['mcc']:.2f} | Yes |")
        G["LOCAL_F1"] = f"{100*L['final']['f1']:.2f} %"
    if "classic_ml" in R:
        Cm = R["classic_ml"]
        for k in ["LogReg", "RandomForest", "ExtraTrees", "HistGB"]:
            for fs in ["selected", "all"]:
                r = Cm.get(f"{k}_{fs}")
                if r:
                    rows.append(f"| Centralised | {k} | {G['NSEL'] if fs == 'selected' else 92} | {100*r['acc']:.2f} | {100*r['f1']:.2f} | {100*r['mcc']:.2f} | No |")
    G["TABLE9"] = "\n".join(rows)
    runs = [R[f"central_TAGGNN_s{s}"] for s in range(3) if f"central_TAGGNN_s{s}" in R]
    if runs:
        G["CENT_F1"] = f"{100*np.mean([r['final']['f1'] for r in runs]):.2f} %"


ABL = [("full", "FENRIR (full)"), ("noFS", "w/o LC-GW2O feature selection (all 92 features)†"),
       ("noHPO", "w/o LC-GW2O tuning (default hyper-parameters)"), ("noG", "w/o graph-propagation branch G"),
       ("noS", "w/o structure-biased attention branch S"), ("noGate", "w/o gated fusion (average of G and S)"),
       ("noChan", "w/o channel attention C"), ("priorOnly", "w/o adaptive adjacency (prior only)"),
       ("adaptOnly", "w/o correlation prior (adaptive only)"), ("noPeriodic", "w/o periodic node tokenizer (linear)"),
       ("noDual", "w/o dual-view readout"), ("FedAvgAgg", "MATA → FedAvg aggregation (μ = 0)")]


def table_ablation():
    R = load_group("ablation")
    rows = ["Table: **Table 11.** Ablation of FENRIR (20 rounds; %, mean ± std over three seeds; Δ = change in mean macro-F1 w.r.t. the full model in percentage points). † single seed because of its cost.", "",
            "| Variant | Accuracy | Macro-F1 | MCC | ΔMacro-F1 (pp) | Params |", "|:--|--:|--:|--:|--:|--:|"]
    full = np.mean([R[f"full_s{s}"]["final"]["f1"] for s in range(3) if f"full_s{s}" in R])
    out = {}
    for key, name in ABL:
        runs = [R[f"{key}_s{s}"] for s in range(3) if f"{key}_s{s}" in R]
        if not runs:
            continue
        f1 = np.array([r["final"]["f1"] for r in runs])
        d = 100 * (f1.mean() - full)
        out[key] = d
        nm = f"**{name}**" if key == "full" else name
        rows.append(f"| {nm} | {pm([r['final']['acc'] for r in runs], pct=True)} | {pm(f1, pct=True)} | "
                    f"{pm([r['final']['mcc'] for r in runs], pct=True)} | {'–' if key == 'full' else f'{d:+.2f}'} | {runs[0]['n_params']:,} |")
    G["TABLE11"] = "\n".join(rows)
    G["_abl"] = out
    nf = R.get("noFS_s0")
    if nf:
        G["NOFS_TEXT"] = (f"Using all 92 features instead of the 24 selected ones (single seed because of its cost) gives "
                          f"{100*nf['final']['f1']:.2f} % macro-F1 and {100*nf['final']['acc']:.2f} % accuracy after "
                          f"{nf['wall']/60:.0f} min of training, versus {100*R['full_s0']['final']['f1']:.2f} % and "
                          f"{100*R['full_s0']['final']['acc']:.2f} % in {R['full_s0']['wall']/60:.0f} min for the full model with the same seed")
    G["ABL_FULL_F1"] = f"{100*full:.2f} %"


def table_robust():
    R = load_group("robust")
    aggs = ["FedAvg", "Median", "TrimmedMean", "Krum", "MATA"]
    atks = [("none", "No attack"), ("signflip", "Sign-flip ×3"), ("gaussian", "Gaussian noise"), ("labelflip", "Label flipping")]
    rows = ["Table: **Table 12.** Byzantine robustness with 3 of 10 clients malicious (TAG-GNN detector, 20 rounds, seed 0): test macro-F1 / accuracy in %. Trimmed mean (β = 0.3) and multi-Krum (f = 3) are given the true number of attackers; MATA is not.", "",
            "| Aggregator | " + " | ".join(n for _, n in atks) + " | Worst case |", "|:--|" + "--:|" * (len(atks) + 1)]
    worst = {}
    for a in aggs:
        cells, w = [], 1.0
        for t, _ in atks:
            r = R.get(f"{a}_{t}")
            if r:
                cells.append(f"{100*r['final']['f1']:.2f} / {100*r['final']['acc']:.2f}")
                w = min(w, r["final"]["f1"])
            else:
                cells.append("–")
        worst[a] = w
        nm = "**MATA (proposed)**" if a == "MATA" else a
        rows.append(f"| {nm} | " + " | ".join(cells) + f" | {100*w:.2f} |")
    G["TABLE12"] = "\n".join(rows)
    G["ROB_MIN"] = f"{100*worst.get('MATA', np.nan):.2f} %"
    fa = [R[f"FedAvg_{t}"]["final"]["f1"] for t, _ in atks[1:] if f"FedAvg_{t}" in R]
    G["ROB_FEDAVG"] = f"{100*min(fa):.2f} %" if fa else "n/a"
    G["_rob"] = (R, worst)


def table_noniid():
    R = load_group("noniid")
    M = G["_main"][0]
    al = ["0.1", "0.3", "0.5", "1.0", "iid"]
    rows = ["Table: **Table 13.** Effect of client heterogeneity (Dirichlet α; seed 0; macro-F1 / accuracy in %). The FS subset and tuned hyper-parameters obtained at α = 0.5 are reused unchanged.", "",
            "| Model | " + " | ".join(("α = " + a) if a != "iid" else "IID" for a in al) + " |", "|:--|" + "--:|" * len(al)]
    for m, nm in [("MLP", "FedAvg-MLP"), ("Transformer", "FedAvg-Transformer"), ("FENRIR", "**FENRIR**")]:
        cells = []
        for a in al:
            r = M.get(f"{m}_s0") if a == "0.5" else R.get(f"{m}_a{a}")
            cells.append(f"{100*r['final']['f1']:.2f} / {100*r['final']['acc']:.2f}" if r else "–")
        rows.append(f"| {nm} | " + " | ".join(cells) + " |")
    fa = R.get("FENRIRfedavg_a0.1")
    if fa:
        rows.append(f"| TAG-GNN + FedAvg (no MATA) | {100*fa['final']['f1']:.2f} / {100*fa['final']['acc']:.2f} | – | – | – | – |")
        G["NONIID_FEDAVG01"] = f"{100*fa['final']['f1']:.2f} % macro-F1 and {100*fa['final']['acc']:.2f} % accuracy"
    G["TABLE13"] = "\n".join(rows)


def table_complexity():
    p = os.path.join(RES, "complexity.json")
    if not os.path.exists(p):
        return
    Cx = json.load(open(p))
    M = G["_main"][1]
    key = {"FENRIR (TAG-GNN)": "FENRIR", "MLP": "MLP", "CNN1D": "CNN1D", "GCN": "GCN", "GAT": "GAT", "Transformer": "Transformer"}
    rows = ["Table: **Table 14.** Model cost on the two-core CPU used as an edge-gateway proxy: parameters, multiply–accumulate operations (MACs) per flow, model size, per-flow latency at batch 256 and batch 1, and traffic per federated round (10 clients, upload + download).", "",
            "| Model | Input features | Params | MACs / flow | Size (KB) | Latency b=256 (ms/flow) | Latency b=1 (ms) | Traffic / round (MB) | Macro-F1 (%) |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for n, r in Cx.items():
        f1 = f"{100*M[key[n]]['f1'].mean():.2f}" if n in key and key[n] in M else "–"
        rows.append(f"| {n} | {r['d']} | {r['params']:,} | {r['macs']:,} | {r['size_kb']:.1f} | {r['lat_b256_ms']:.4f} | {r['lat_b1_ms']:.2f} | {r['comm_round_MB']:.2f} | {f1} |")
    G["TABLE14"] = "\n".join(rows)
    f, a = Cx["FENRIR (TAG-GNN)"], Cx["TAG-GNN, all 92 features"]
    G["MAC_RED"] = f"{a['macs']/f['macs']:.1f}"
    G["LAT_RED"] = f"{a['lat_b256_ms']/f['lat_b256_ms']:.1f}"
    G["F_LAT"] = f"{f['lat_b256_ms']:.3f}"
    G["F_LAT1"] = f"{f['lat_b1_ms']:.2f}"
    G["F_PARAMS"] = f"{f['params']:,}"
    G["F_SIZE"] = f"{f['size_kb']:.1f}"
    G["_cx"] = Cx


def main():
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    G["TABLE1"], G["TABLE2"], G["ALGO1"], G["ALGO2"] = SB.TABLE1, SB.TABLE2, SB.ALGO1, SB.ALGO2
    table3(fd)
    table_bench()
    table_fs(fd)
    for fn in [table4, lambda: table_main(fd), table_refs, table_ablation, table_robust, table_noniid, table_complexity]:
        try:
            fn()
        except Exception as e:  # partial results are allowed while experiments run
            print("skip", getattr(fn, "__name__", "lambda"), repr(e))
    out = {k: v for k, v in G.items() if not k.startswith("_")}
    json.dump(out, open(os.path.join(PAPER, "generated.json"), "w"), indent=1)
    return G


if __name__ == "__main__":
    g = main()
    for k, v in g.items():
        if not k.startswith("_") and not k.startswith("TABLE") and not k.startswith("ALGO"):
            print(k, "=", v)
