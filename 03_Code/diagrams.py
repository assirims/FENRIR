"""Architecture and methodology diagrams (Figures 1-3)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def tint(hex_, a=0.13):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(255 - (255 - c) * a) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def box(ax, x, y, w, h, text, color, fs=8.5, bold=False, ec=None, lw=1.2, txtc=INK, r=0.012, ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=tint(color),
                       ec=ec or color, lw=lw)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 0.01
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=txtc,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return p


def panel(ax, x, y, w, h, title, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc="#fbfbfa",
                       ec=color, lw=1.6, ls="-")
    ax.add_patch(p)
    ax.text(x + 0.012, y + h - 0.022, title, ha="left", va="top", fontsize=10, fontweight="bold", color=color)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.3, style="-|>", ls="-", rad=0.0, label=None, lx=0, ly=0, fs=7.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs, color=color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6))


# ============================================================================ Figure 1
def fig_architecture():
    fig = plt.figure(figsize=(14, 7.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, "FENRIR: Federated Edge-graph Network with Resilient Intrusion Recognition for Edge IIoT",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
    # ---- Layer 1
    panel(ax, 0.01, 0.10, 0.18, 0.84, "1  IIoT perception layer", GREEN)
    devs = ["PLC / RTU controllers", "MQTT sensor nodes", "Smart actuators / bulbs", "IP cameras / HMI",
            "ThingSpeak telemetry", "Exposed services\n(SSH, HTTP, DNS, NTP)"]
    for i, d in enumerate(devs):
        box(ax, 0.022, 0.79 - i * 0.105, 0.156, 0.075, d, GREEN, fs=8)
    ax.text(0.10, 0.14, "Bidirectional flow features\nper session (83 attributes)", ha="center", fontsize=7.8,
            color=INK2, style="italic")
    # ---- Layer 2
    panel(ax, 0.215, 0.10, 0.325, 0.84, "2  Edge gateways = FL clients (k = 1..K)", BLUE)
    X0, W2 = 0.228, 0.30
    rows = {}
    for k, top in enumerate([0.875, 0.575]):
        lbl = "Edge client k" if k == 0 else "Edge client K"
        ax.text(X0, top + 0.005, lbl, fontsize=9, fontweight="bold", color=BLUE, va="bottom")
        y1 = top - 0.07
        box(ax, X0, y1, 0.14, 0.065, "Local store D_k\n(never leaves site)", BLUE, fs=8)
        box(ax, X0 + 0.16, y1, 0.14, 0.065, "Signed-log transform +\nsufficient statistics", BLUE, fs=8)
        arrow(ax, X0 + 0.14, y1 + 0.032, X0 + 0.16, y1 + 0.032)
        y2 = y1 - 0.08
        box(ax, X0, y2, W2, 0.065, "Local proxy evaluator: candidate mask / config  ->  local macro-F1", ORANGE, fs=8)
        y3 = y2 - 0.085
        box(ax, X0, y3, W2, 0.07, "TAG-GNN local training (proximal objective)\nmin_w  F_k(w) + (mu/2) ||w - w^t||^2   ->   Delta_k",
            VIOLET, fs=8)
        rows[k] = (y1 + 0.032, y2 + 0.032, y3 + 0.035, y3)
    ax.text(X0 + W2 / 2, 0.595, "...", fontsize=14, ha="center", color=BLUE)
    box(ax, X0, 0.13, W2, 0.075, "Real-time inference + explanation at the edge\n(feature-node importance, learned attack graph)",
        AQUA, fs=8)
    arrow(ax, X0 + W2 / 2, rows[1][3], X0 + W2 / 2, 0.205, AQUA)
    # ---- Layer 3
    S0, SW = 0.60, 0.385
    panel(ax, 0.585, 0.10, 0.405, 0.84, "3  Aggregation server (fog / cloud)", VIOLET)
    box(ax, S0, 0.765, SW, 0.105, "Phase I  -  Federated statistics (secure sums)\n"
        "mu = (1/n) sum_k S1_k ,   sigma^2 = (1/n) sum_k S2_k - mu^2 ,   R = (1/n) sum_k X_k' X_k\n"
        "->  correlation prior graph A_rho (top-k |R|), class weights", BLUE, fs=8)
    box(ax, S0, 0.565, SW, 0.17, "Phase II  -  LC-GW2O hybrid optimiser\n"
        "chaotic + opposition initialisation, fitness-weighted alpha/beta/delta encircling,\n"
        "whale logarithmic spiral, Levy-chaotic flights, inheritance crossover,\n"
        "elite local search, quasi-opposition restarts\n"
        "(a) federated feature selection      (b) federated hyper-parameter optimisation", ORANGE, fs=8)
    box(ax, S0, 0.345, SW, 0.19, "Phase III  -  MATA robust aggregation\n"
        "1. median anchor:  m = med_k(Delta_k)\n"
        "2. norm clipping:  Delta~_k = Delta_k * min(1, nu_med / ||Delta_k||)\n"
        "3. directional trust:  t_k = exp(-tau (1 - cos(Delta~_k, m))) * 1[cos > 0]\n"
        "4. update:  w^(t+1) = w^t + sum_k omega_k Delta~_k ,   omega_k ~ n_k t_k", RED, fs=8)
    box(ax, S0, 0.13, SW, 0.18, "Global TAG-GNN detector  w^(t+1)\n"
        "periodic node tokenizer, prior + adaptive adjacency,\n"
        "tri-attention gated blocks (graph | structural MHSA | channel),\n"
        "attention pooling + dual-view readout  ->  12-class softmax", VIOLET, fs=8)
    arrow(ax, S0 + SW / 2, 0.345, S0 + SW / 2, 0.31, RED)
    # ---- flows
    arrow(ax, 0.178, 0.83, X0, rows[0][0], GREEN)
    arrow(ax, 0.178, 0.51, X0, rows[1][0], GREEN)
    ax.text(0.2, 0.86, "flows", fontsize=7.5, color=GREEN)
    xe = X0 + W2
    arrow(ax, xe, rows[0][0], S0, 0.83, BLUE, label="sums", lx=0.0, ly=0.018, fs=7.3)
    arrow(ax, S0, 0.68, xe, rows[0][1] + 0.012, ORANGE, label="masks / configs", lx=0.0, ly=0.03, fs=7.0)
    arrow(ax, xe, rows[0][1] - 0.012, S0, 0.60, ORANGE, ls="--", label="scalar fitness", lx=0.0, ly=-0.035, fs=7.0)
    arrow(ax, xe, rows[0][2], S0, 0.47, RED, label="Delta_k", lx=0.0, ly=0.03, fs=7.3)
    arrow(ax, xe, rows[1][2], S0, 0.40, RED)
    arrow(ax, S0, 0.20, xe, rows[1][2] - 0.02, VIOLET, rad=-0.15, label="broadcast w^(t+1)", lx=0.0, ly=-0.045, fs=7.0)
    ax.text(0.5, 0.05, "Only secure sums, scalar fitness values and model updates traverse the network; "
            "raw IIoT traffic never leaves the edge gateway.", ha="center", fontsize=9, color=INK2, style="italic")
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ Figure 2
def fig_flow():
    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=(8.8, 12.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, X = 0.56, 0.22
    cx = X + W / 2
    steps = [
        ("Start: K edge gateways collect labelled IIoT flows D_k", GREEN, 1),
        ("Local cleaning: drop identifiers / ephemeral ports, one-hot proto & service,\nsigned-log transform x -> sgn(x) log(1+|x|)", BLUE, 2),
        ("Secure aggregation of {n_k, sum x, sum x^2, X'X, class counts}\n-> global mean / std, correlation matrix R, class weights", BLUE, 2),
        ("Federated LC-GW2O feature selection (binary, NFE = 300)", ORANGE, 1),
        ("Build correlation prior graph A_rho = top-k |R| over selected features", BLUE, 1),
        ("Federated LC-GW2O hyper-parameter optimisation (NFE = 40)\nfitness = 1 - macro-F1 on client-held validation sets", ORANGE, 2),
        ("Initialise global TAG-GNN w^0;  t = 0", VIOLET, 1),
        ("Broadcast w^t; each client runs proximal local training -> Delta_k", VIOLET, 1),
        ("MATA aggregation: median anchor -> norm clipping -> directional trust -> w^(t+1)", RED, 1),
        ("DIAMOND", None, 0),
        ("Deploy w^T at the edge: real-time multi-class detection +\nnode-importance / attack-graph explanations", AQUA, 2),
    ]
    y = 0.965
    pos = []
    for t, c, nl in steps:
        h = 0.06 if t == "DIAMOND" else (0.048 if nl == 1 else 0.062)
        if t == "DIAMOND":
            yc = y - h / 2
            dia = Polygon([[cx, y], [cx + 0.13, yc], [cx, y - h], [cx - 0.13, yc]], closed=True,
                          fc=tint(YELLOW, 0.25), ec=YELLOW, lw=1.3)
            ax.add_patch(dia)
            ax.text(cx, yc, "t + 1 < T ?", ha="center", va="center", fontsize=9, fontweight="bold")
        else:
            box(ax, X, y - h, W, h, t, c, fs=8.4, bold=(c in (GREEN, AQUA)))
        pos.append((y - h, y))
        y = y - h - 0.032
    for i in range(len(steps) - 1):
        lab = "no" if steps[i][0] == "DIAMOND" else None
        arrow(ax, cx, pos[i][0], cx, pos[i + 1][1], INK2, label=lab, lx=0.025)
    yd = (pos[9][0] + pos[9][1]) / 2
    yb = (pos[7][0] + pos[7][1]) / 2
    ax.plot([cx + 0.13, X + W + 0.06, X + W + 0.06], [yd, yd, yb], color=INK2, lw=1.3)
    arrow(ax, X + W + 0.06, yb, X + W, yb, INK2)
    ax.text(X + W + 0.07, (yd + yb) / 2, "yes:\nt <- t+1", fontsize=8, color=INK2, va="center")
    for idx, txt in [(3, "server proposes masks ->\nclients return local macro-F1\n(scalars only) -> update wolves"),
                     (5, "server proposes configs ->\nclients run short FL proxy ->\nreturn val. confusion counts")]:
        y0, y1 = pos[idx]
        ax.plot([X, X - 0.04, X - 0.04], [y0 + 0.01, y0 + 0.01, y1 - 0.01], color=ORANGE, lw=1.1)
        arrow(ax, X - 0.04, y1 - 0.01, X, y1 - 0.01, ORANGE)
        ax.text(X - 0.05, (y0 + y1) / 2, txt, ha="right", va="center", fontsize=7.4, color=ORANGE)
    for (i0, i1), lab, col in [((1, 2), "Phase I", BLUE), ((3, 5), "Phase II", ORANGE), ((6, 9), "Phase III", RED)]:
        ya, yb2 = pos[i1][0], pos[i0][1]
        xx = 0.97
        ax.plot([xx, xx], [ya, yb2], color=col, lw=2.5)
        ax.text(xx - 0.012, (ya + yb2) / 2, lab, rotation=90, fontsize=9.5, color=col, fontweight="bold",
                ha="right", va="center")
    fig.savefig(os.path.join(FIG, "fig2_flowchart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ Figure 3
def fig_taggnn():
    fig = plt.figure(figsize=(13.5, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "TAG-GNN: Tri-Attention Gated Graph Neural Network (feature-as-node view of a flow record)",
            ha="center", fontsize=12, fontweight="bold")
    # input + tokenizer
    box(ax, 0.01, 0.62, 0.10, 0.16, "Flow record\n$x\\in\\mathbb{R}^{d}$\n(d LC-GW2O\nselected features)", GREEN, fs=8)
    box(ax, 0.135, 0.58, 0.13, 0.24, "Periodic node\ntokenizer\n$h_i^0=\\mathrm{ReLU}(W_i[\\sin 2\\pi f_i x_i,$\n$\\cos 2\\pi f_i x_i]+b_i)$\n$+\\,x_i w_i+p_i$", BLUE, fs=7.8)
    arrow(ax, 0.11, 0.70, 0.135, 0.70)
    # adjacency
    box(ax, 0.135, 0.17, 0.13, 0.30, "Adjacency module\n$A_\\rho$: federated top-k |ρ|\n$A_\\theta=\\mathrm{softmax}$\n$(\\mathrm{ReLU}(E_1E_2^\\top))$\n"
        "$\\tilde A=\\lambda A_\\rho+(1-\\lambda)A_\\theta$\n$\\lambda=\\sigma(\\theta)$", ORANGE, fs=7.8)
    # block
    panel(ax, 0.29, 0.08, 0.47, 0.80, "Tri-attention gated block  (× L)", VIOLET)
    box(ax, 0.305, 0.58, 0.14, 0.17, "Branch G\ngraph propagation\n$M_G=\\mathrm{GELU}(\\tilde A H W_g)$", ORANGE, fs=7.8)
    box(ax, 0.305, 0.33, 0.14, 0.20, "Branch S\nstructure-biased MHSA\n\n$\\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_h}}+\\gamma_h\\log\\tilde A\\right)V$",
        MAGENTA, fs=7.8)
    box(ax, 0.47, 0.44, 0.12, 0.22, "Gated fusion\n$z=\\sigma([M_G\\|M_S]W_z)$\n$F=z\\odot M_G$\n$+(1-z)\\odot M_S$", RED, fs=7.8)
    box(ax, 0.47, 0.17, 0.12, 0.20, "Residual + LN\n$H'=\\mathrm{LN}(H+F)$\n$H''=\\mathrm{LN}(H'$\n$+\\mathrm{FFN}(H'))$", BLUE, fs=7.8)
    box(ax, 0.61, 0.33, 0.135, 0.33, "Branch C\nchannel (node)\nattention\n$s=\\sigma(W_2\\mathrm{ReLU}$\n$(W_1 H''u))$\n$H\\leftarrow H''\\odot s$\n(node importance)",
        AQUA, fs=7.8)
    arrow(ax, 0.265, 0.70, 0.305, 0.665)
    arrow(ax, 0.265, 0.68, 0.305, 0.45)
    arrow(ax, 0.265, 0.36, 0.305, 0.60, ORANGE, rad=0.15)
    arrow(ax, 0.265, 0.30, 0.305, 0.38, ORANGE, label="$\\log\\tilde A$", lx=-0.004, ly=-0.02, fs=7.3)
    arrow(ax, 0.445, 0.665, 0.47, 0.60)
    arrow(ax, 0.445, 0.43, 0.47, 0.50)
    arrow(ax, 0.53, 0.44, 0.53, 0.37)
    arrow(ax, 0.59, 0.27, 0.61, 0.40)
    arrow(ax, 0.745, 0.50, 0.775, 0.50)
    # readout
    box(ax, 0.775, 0.55, 0.10, 0.21, "Attention\npooling\n$a=\\mathrm{softmax}_i$\n$(u^\\top\\tanh(W_rh_i))$\n+ mean pool", VIOLET, fs=7.8)
    box(ax, 0.775, 0.27, 0.10, 0.17, "Dual-view\nreadout\n$[\\,r_{att}\\|r_{mean}\\|x\\,]$", GREEN, fs=7.8)
    box(ax, 0.895, 0.40, 0.095, 0.22, "MLP head\nGELU → drop\n→ softmax\n12 classes\n(CE + label\nsmoothing)", RED, fs=7.8)
    arrow(ax, 0.825, 0.55, 0.825, 0.44)
    arrow(ax, 0.875, 0.36, 0.895, 0.46)
    ax.plot([0.06, 0.06, 0.80], [0.62, 0.05, 0.05], color=GREEN, lw=1.2, ls="--")
    arrow(ax, 0.80, 0.05, 0.80, 0.27, GREEN, ls="--", label="raw-feature view", lx=-0.06, ly=-0.08, fs=7.3)
    fig.savefig(os.path.join(FIG, "fig3_taggnn.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_flow()
    fig_taggnn()
    print("ok")
