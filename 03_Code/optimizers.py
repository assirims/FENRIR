"""
Metaheuristic optimisers (minimisation, box-constrained, NFE-budgeted).
  * LCGW2O : proposed Levy-Chaotic Grey-Wolf-Whale hybrid optimiser
  * GWO, WOA, PSO, DE, HHO : baselines
All optimisers share an `Objective` wrapper that counts function evaluations (NFE)
and records the best-so-far trace so that comparisons are made at equal budget.
"""
import math
import numpy as np


class Objective:
    def __init__(self, f, budget):
        self.f, self.budget = f, budget
        self.nfe, self.best, self.bx, self.trace = 0, np.inf, None, []

    def __call__(self, X):
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            if self.nfe >= self.budget:
                out[i:] = np.inf
                break
            v = self.f(x)
            out[i] = v
            self.nfe += 1
            if v < self.best:
                self.best, self.bx = v, x.copy()
            self.trace.append(self.best)
        return out

    @property
    def done(self):
        return self.nfe >= self.budget


def levy(rng, shape, beta=1.5):
    s = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
         (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, s, shape)
    v = rng.normal(0, 1, shape)
    return u / np.abs(v) ** (1 / beta)


def _clip(X, lb, ub):
    return np.clip(X, lb, ub)


# ----------------------------------------------------------------------------- proposed
def lcgw2o(obj, D, lb, ub, N, rng, use_levy=True, use_chaos=True, use_obl=True, use_spiral=True,
           stag_lim=4, cr=0.5, elite_ls=3):
    """Levy-Chaotic Grey-Wolf-Whale Optimiser (LC-GW2O).
    (i) logistic-chaos + opposition-based initialisation, (ii) fitness-weighted alpha/beta/delta
    encircling (GWO) hybridised with logarithmic-spiral bubble-net exploitation (WOA) through a
    time-varying switching probability, (iii) Levy-flight perturbation scaled by a decaying
    chaotic step, (iv) elitist greedy selection and (v) quasi-opposition restarts on stagnation."""
    lb, ub = np.broadcast_to(lb, D).astype(float), np.broadcast_to(ub, D).astype(float)
    if use_chaos:
        z = rng.uniform(0.01, 0.99, (N, D))
        for _ in range(20):
            z = 4.0 * z * (1 - z)
        X = lb + np.clip(z, 0, 1) * (ub - lb)
    else:
        X = lb + rng.random((N, D)) * (ub - lb)
    F = obj(X)
    if use_obl:
        Xo = lb + ub - X
        Fo = obj(Xo)
        XX, FF = np.vstack([X, Xo]), np.concatenate([F, Fo])
        o = np.argsort(FF)[:N]
        X, F = XX[o], FF[o]
    zc, stag, last_best = 0.7, 0, np.min(F)
    T_est = max(1, (obj.budget - obj.nfe) // N)
    t = 0
    while not obj.done:
        tau = min(1.0, t / T_est)
        a = 2.0 * (1 - tau ** 2)                       # non-linear convergence factor
        p_gwo = 0.8 - 0.5 * tau if use_spiral else 1.0  # GWO <-> WOA switching probability
        zc = 4.0 * zc * (1 - zc) if use_chaos else rng.random()
        o = np.argsort(F)
        L = X[o[:3]]
        fl = F[o[:3]]
        w = 1.0 / (fl - fl[0] + 1e-12 + np.abs(fl[0]) * 1e-6 + 1e-300)
        w = w / w.sum() if np.all(np.isfinite(w)) else np.array([1 / 3] * 3)
        w = 0.5 * w + 0.5 / 3                            # fitness-weighted leaders
        Xn = np.empty_like(X)
        mode = rng.random(N) < p_gwo
        # --- grey-wolf encircling
        acc = np.zeros((N, D))
        Aabs = np.zeros(N)
        for j in range(3):
            A = 2 * a * rng.random((N, D)) - a
            C = 2 * rng.random((N, D))
            Dl = np.abs(C * L[j] - X)
            acc += w[j] * (L[j] - A * Dl)
            Aabs += np.abs(A).mean(1) / 3
        Xn[mode] = acc[mode]
        # --- whale logarithmic spiral around alpha
        if (~mode).any():
            l = rng.uniform(-1, 1, ((~mode).sum(), 1))
            Dp = np.abs(L[0] - X[~mode])
            Xn[~mode] = Dp * np.exp(l) * np.cos(2 * np.pi * l) + L[0]
        # --- Levy flight (exploration), strength decays with time and is chaos-modulated
        if use_levy:
            pl = 0.5 * (1 - tau)
            m = rng.random(N) < pl
            if m.any():
                step = 0.05 * zc * (1 - tau) * levy(rng, (m.sum(), D))
                Xn[m] = Xn[m] + step * (ub - lb) * np.sign(rng.random((m.sum(), D)) - 0.5)
        if cr > 0:  # dimension-wise crossover with the parent (keeps good coordinates)
            keep = rng.random((N, D)) > cr
            keep[np.arange(N), rng.integers(0, D, N)] = False
            Xn = np.where(keep, X, Xn)
        Xn = _clip(Xn, lb, ub)
        Fn = obj(Xn)
        better = Fn < F
        X[better], F[better] = Xn[better], Fn[better]
        if elite_ls and not obj.done:  # chaotic Gaussian local search around alpha
            ia = np.argmin(F)
            sig = (0.1 * (1 - tau) ** 2 + 1e-4) * (ub - lb) * zc
            Xe = _clip(X[ia] + sig * rng.normal(size=(elite_ls, D)), lb, ub)
            Fe = obj(Xe)
            j = np.argmin(Fe)
            if Fe[j] < F[ia]:
                X[ia], F[ia] = Xe[j], Fe[j]
        # --- stagnation-triggered quasi-opposition on the worst half
        b = np.min(F)
        stag = stag + 1 if b >= last_best - 1e-12 else 0
        last_best = min(last_best, b)
        if use_obl and stag >= stag_lim and not obj.done:
            worst = np.argsort(F)[N // 2:]
            lo, hi = X.min(0), X.max(0)
            c = (lo + hi) / 2
            opp = lo + hi - X[worst]
            Xq = c + (opp - c) * rng.random((len(worst), D))
            Xq = _clip(Xq, lb, ub)
            Fq = obj(Xq)
            bt = Fq < F[worst]
            X[worst[bt]], F[worst[bt]] = Xq[bt], Fq[bt]
            stag = 0
        t += 1
    return obj


# ----------------------------------------------------------------------------- baselines
def gwo(obj, D, lb, ub, N, rng):
    X = lb + rng.random((N, D)) * (ub - lb)
    F = obj(X)
    A_arch = np.argsort(F)[:3]
    Lx, Lf = X[A_arch].copy(), F[A_arch].copy()
    T = max(1, (obj.budget - obj.nfe) // N)
    t = 0
    while not obj.done:
        a = 2 - 2 * t / T
        acc = 0
        for j in range(3):
            A = 2 * a * rng.random((N, D)) - a
            C = 2 * rng.random((N, D))
            acc = acc + (Lx[j] - A * np.abs(C * Lx[j] - X))
        X = _clip(acc / 3, lb, ub)
        F = obj(X)
        allx, allf = np.vstack([Lx, X]), np.concatenate([Lf, F])
        o = np.argsort(allf)[:3]
        Lx, Lf = allx[o].copy(), allf[o].copy()
        t += 1
    return obj



def woa(obj, D, lb, ub, N, rng, b=1.0):
    X = lb + rng.random((N, D)) * (ub - lb)
    F = obj(X)
    bi = np.argmin(F)
    Bx, Bf = X[bi].copy(), F[bi]
    T = max(1, (obj.budget - obj.nfe) // N)
    t = 0
    while not obj.done:
        a = 2 - 2 * t / T
        Xn = np.empty_like(X)
        for i in range(N):
            r = rng.random()
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            if r < 0.5:
                if abs(A) < 1:
                    Xn[i] = Bx - A * np.abs(C * Bx - X[i])
                else:
                    Xr = X[rng.integers(N)]
                    Xn[i] = Xr - A * np.abs(C * Xr - X[i])
            else:
                l = rng.uniform(-1, 1)
                Xn[i] = np.abs(Bx - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + Bx
        X = _clip(Xn, lb, ub)
        F = obj(X)
        bi = np.argmin(F)
        if F[bi] < Bf:
            Bx, Bf = X[bi].copy(), F[bi]
        t += 1
    return obj


def pso(obj, D, lb, ub, N, rng, c1=1.49445, c2=1.49445):
    X = lb + rng.random((N, D)) * (ub - lb)
    V = np.zeros((N, D))
    F = obj(X)
    P, PF = X.copy(), F.copy()
    g = P[np.argmin(PF)].copy()
    vmax = 0.2 * (ub - lb)
    T = max(1, (obj.budget - obj.nfe) // N)
    t = 0
    while not obj.done:
        w = 0.9 - 0.5 * t / T
        V = w * V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = _clip(X + V, lb, ub)
        F = obj(X)
        imp = F < PF
        P[imp], PF[imp] = X[imp], F[imp]
        g = P[np.argmin(PF)].copy()
        t += 1
    return obj


def de(obj, D, lb, ub, N, rng, Fw=0.5, CR=0.9):
    X = lb + rng.random((N, D)) * (ub - lb)
    F = obj(X)
    while not obj.done:
        idx = np.array([rng.choice(np.delete(np.arange(N), i), 3, replace=False) for i in range(N)])
        M = X[idx[:, 0]] + Fw * (X[idx[:, 1]] - X[idx[:, 2]])
        cr = rng.random((N, D)) < CR
        cr[np.arange(N), rng.integers(0, D, N)] = True
        U = _clip(np.where(cr, M, X), lb, ub)
        FU = obj(U)
        b = FU < F
        X[b], F[b] = U[b], FU[b]
    return obj


def hho(obj, D, lb, ub, N, rng):
    X = lb + rng.random((N, D)) * (ub - lb)
    F = obj(X)
    bi = np.argmin(F)
    R, RF = X[bi].copy(), F[bi]
    T = max(1, (obj.budget - obj.nfe) // (1.5 * N))
    t = 0
    while not obj.done:
        E1 = 2 * (1 - t / T)
        Xm = X.mean(0)
        for i in range(N):
            if obj.done:
                break
            E = E1 * (2 * rng.random() - 1)
            if abs(E) >= 1:
                q = rng.random()
                Xr = X[rng.integers(N)]
                if q >= 0.5:
                    Xn = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
                else:
                    Xn = (R - Xm) - rng.random() * (lb + rng.random() * (ub - lb))
                Xn = _clip(Xn, lb, ub)
                fn = obj(Xn)[0]
                X[i], F[i] = Xn, fn
            else:
                r = rng.random()
                J = 2 * (1 - rng.random())
                if r >= 0.5 and abs(E) >= 0.5:
                    Xn = (R - X[i]) - E * np.abs(J * R - X[i])
                elif r >= 0.5:
                    Xn = R - E * np.abs(R - X[i])
                else:
                    base = R if abs(E) < 0.5 else R
                    ref = Xm if abs(E) < 0.5 else X[i]
                    Y = _clip(base - E * np.abs(J * R - ref), lb, ub)
                    fy = obj(Y)[0]
                    if fy < F[i]:
                        X[i], F[i] = Y, fy
                        continue
                    Z = _clip(Y + rng.random(D) * levy(rng, D) * 0.01 * (ub - lb), lb, ub)
                    fz = obj(Z)[0]
                    if fz < F[i]:
                        X[i], F[i] = Z, fz
                    continue
                Xn = _clip(Xn, lb, ub)
                fn = obj(Xn)[0]
                X[i], F[i] = Xn, fn
            if F[i] < RF:
                R, RF = X[i].copy(), F[i]
        t += 1
    return obj


ALGOS = {"PSO": pso, "DE": de, "GWO": gwo, "WOA": woa, "HHO": hho, "LCGW2O": lcgw2o}


# ----------------------------------------------------------------------------- benchmarks
def _shift(D, lo, hi, seed):
    return np.random.default_rng(seed).uniform(0.6 * lo, 0.6 * hi, D)


def benchmark_suite(D=30):
    """Shifted classical functions (shift removes the centre bias of GWO-type search)."""
    S = {}

    def mk(name, f, lo, hi, opt=0.0, shift=True, seed=0):
        o = _shift(D, lo, hi, seed) if shift else np.zeros(D)
        S[name] = dict(f=(lambda x, f=f, o=o: f(x - o)), lb=lo, ub=hi, opt=opt)
    mk("F1 Sphere", lambda z: np.sum(z ** 2), -100, 100, seed=1)
    mk("F2 Schwefel 2.22", lambda z: np.sum(np.abs(z)) + np.prod(np.abs(z)), -10, 10, seed=2)
    mk("F3 Schwefel 1.2", lambda z: np.sum(np.cumsum(z) ** 2), -100, 100, seed=3)
    mk("F4 Rosenbrock", lambda z: np.sum(100 * ((z[1:] + 1) - (z[:-1] + 1) ** 2) ** 2 + z[:-1] ** 2), -30, 30, seed=4)
    mk("F5 Zakharov", lambda z: np.sum(z ** 2) + np.sum(0.5 * np.arange(1, D + 1) * z) ** 2 +
       np.sum(0.5 * np.arange(1, D + 1) * z) ** 4, -5, 10, seed=5)
    mk("F6 Rastrigin", lambda z: np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10), -5.12, 5.12, seed=6)
    mk("F7 Ackley", lambda z: -20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e,
       -32, 32, seed=7)
    mk("F8 Griewank", lambda z: np.sum(z ** 2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1, -600, 600, seed=8)

    def levyf(z):
        w = 1 + z / 4
        return (np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
                + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2))
    mk("F9 Levy", levyf, -10, 10, seed=9)
    mk("F10 Schwefel 2.26", lambda z: 418.9829 * D - np.sum(z * np.sin(np.sqrt(np.abs(z)))), -500, 500, shift=False)
    return S
