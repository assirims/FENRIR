"""
Neural models used in the study.
  * TAGGNN  : proposed Tri-Attention Gated Graph Neural Network (FENRIR detector)
  * MLP, CNN1D, GCN, GATNet, FTTransformer : federated deep baselines
All models treat a flow record x in R^d either as a vector (MLP/CNN) or as a
graph whose nodes are the d (selected) traffic features.
"""
import numpy as np
import autograd as ag
from autograd import Tensor, Param


class Module:
    def __init__(self, seed=0):
        self.params = []
        self.rng = np.random.default_rng(seed)
        self.train = True
        self.cache = {}

    def P(self, *shape, init="xavier"):
        if init == "zeros":
            a = np.zeros(shape)
        elif init == "ones":
            a = np.ones(shape)
        elif init == "normal":
            a = self.rng.normal(0, 0.1, shape)
        else:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            fan_out = shape[-1]
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            a = self.rng.uniform(-lim, lim, shape)
        p = Param(a)
        self.params.append(p)
        return p

    def n_params(self):
        return int(sum(p.data.size for p in self.params))

    def get_flat(self):
        return np.concatenate([p.data.ravel() for p in self.params]).astype(np.float32)

    def set_flat(self, v):
        i = 0
        for p in self.params:
            n = p.data.size
            p.data = v[i:i + n].reshape(p.data.shape).astype(ag.DT).copy()
            i += n

    def predict_proba(self, X, bs=1024):
        self.train = False
        out = []
        for i in range(0, len(X), bs):
            lg = self.forward(X[i:i + bs]).data
            lg = lg - lg.max(1, keepdims=True)
            e = np.exp(lg)
            out.append(e / e.sum(1, keepdims=True))
        self.train = True
        return np.concatenate(out)


def _lin(m, x, W, b=None):
    y = ag.matmul(x, W)
    return y if b is None else ag.add(y, b)


class Tokenizer:
    """Feature-as-node embedding: h_i = x_i * w_i + p_i  (value + identity embedding)."""

    def __init__(self, m, d, D):
        self.W = m.P(d, D, init="normal")
        self.Pb = m.P(d, D, init="normal")

    def __call__(self, x):
        xt = Tensor(x[:, :, None].astype(ag.DT))
        return ag.add(ag.mul(xt, self.W), self.Pb)


class PeriodicTokenizer:
    """Learnable periodic (Fourier) node embedding for numerical traffic features:
    h_i = ReLU(W_i [sin(2*pi*f_i x_i), cos(2*pi*f_i x_i)] + b_i) + x_i w_i + p_i."""

    def __init__(self, m, d, D, nf=4, sigma=0.5):
        self.d, self.D, self.nf = d, D, nf
        self.f = Param(m.rng.normal(0, sigma, (1, d, nf)))
        m.params.append(self.f)
        self.Wp = m.P(d, 2 * nf, D)
        self.bp = m.P(d, 1, D, init="zeros")
        self.lin = Tokenizer(m, d, D)

    def __call__(self, x):
        B = x.shape[0]
        xt = Tensor((2 * np.pi * x[:, :, None]).astype(ag.DT))
        ph = ag.mul(xt, self.f)                                    # (B,d,nf)
        P = ag.concat([ag.sin(ph), ag.cos(ph)], -1)                # (B,d,2nf)
        P = ag.transpose(P, (1, 0, 2))                             # (d,B,2nf)
        E = ag.relu(ag.add(ag.matmul(P, self.Wp), self.bp))       # (d,B,D)
        return ag.add(ag.transpose(E, (1, 0, 2)), self.lin(x))


class MHSA:
    """Multi-head self-attention over feature nodes with optional additive structural bias."""

    def __init__(self, m, D, h):
        self.m, self.h, self.dh = m, h, D // h
        self.Wq, self.Wk, self.Wv, self.Wo = (m.P(D, D) for _ in range(4))
        self.bo = m.P(D, init="zeros")

    def __call__(self, H, bias=None):
        B, d, D = H.shape
        h, dh = self.h, self.dh

        def heads(W):
            return ag.transpose(ag.reshape(ag.matmul(H, W), (B, d, h, dh)), (0, 2, 1, 3))
        Q, K, V = heads(self.Wq), heads(self.Wk), heads(self.Wv)
        S = ag.scale(ag.matmul(Q, ag.transpose(K, (0, 1, 3, 2))), 1.0 / np.sqrt(dh))
        if bias is not None:
            S = ag.add(S, bias)
        A = ag.softmax(S, -1)
        self.m.cache.setdefault("attn", []).append(A.data)
        O = ag.reshape(ag.transpose(ag.matmul(A, V), (0, 2, 1, 3)), (B, d, D))
        return _lin(self.m, O, self.Wo, self.bo)


class FFN:
    def __init__(self, m, D, mult=2):
        self.W1, self.b1 = m.P(D, mult * D), m.P(mult * D, init="zeros")
        self.W2, self.b2 = m.P(mult * D, D), m.P(D, init="zeros")

    def __call__(self, H):
        return _lin(None, ag.gelu(_lin(None, H, self.W1, self.b1)), self.W2, self.b2)


class LN:
    def __init__(self, m, D):
        self.g, self.b = m.P(D, init="ones"), m.P(D, init="zeros")

    def __call__(self, x):
        return ag.layernorm(x, self.g, self.b)


class Head:
    def __init__(self, m, Din, D, C):
        self.W1, self.b1 = m.P(Din, D), m.P(D, init="zeros")
        self.W2, self.b2 = m.P(D, C), m.P(C, init="zeros")
        self.m = m

    def __call__(self, z, p):
        z = ag.gelu(_lin(None, z, self.W1, self.b1))
        z = ag.dropout(z, p, self.m.train, self.m.rng)
        return _lin(None, z, self.W2, self.b2)


# =============================================================================
# Proposed: Tri-Attention Gated Graph Neural Network (TAG-GNN)
# =============================================================================
class TAGGNN(Module):
    def __init__(self, d, C, A_prior, D=32, heads=4, L=2, drop=0.1, emb=8,
                 use_G=True, use_S=True, gate=True, chan_att=True, adj="both",
                 att_pool=True, periodic=True, dual_view=True, seed=0):
        super().__init__(seed)
        self.d, self.C, self.D, self.L, self.drop = d, C, D, L, drop
        self.use_G, self.use_S, self.gate, self.chan_att, self.adj, self.att_pool = \
            use_G, use_S, gate, chan_att, adj, att_pool
        self.A_prior = A_prior.astype(ag.DT)
        self.dual_view = dual_view
        self.tok = PeriodicTokenizer(self, d, D) if periodic else Tokenizer(self, d, D)
        if adj in ("both", "adaptive"):
            self.E1, self.E2 = self.P(d, emb, init="normal"), self.P(d, emb, init="normal")
        if adj == "both":
            self.theta = self.P(1, init="zeros")  # lambda = sigmoid(theta) = 0.5 at init
        self.layers = []
        for _ in range(L):
            lay = {}
            if use_G:
                lay["Wg"], lay["bg"] = self.P(D, D), self.P(D, init="zeros")
            if use_S:
                lay["att"] = MHSA(self, D, heads)
                lay["gam"] = self.P(1, heads, 1, 1, init="ones")
            if use_G and use_S and gate:
                lay["Wz"], lay["bz"] = self.P(2 * D, D), self.P(D, init="zeros")
            lay["ln1"], lay["ffn"], lay["ln2"] = LN(self, D), FFN(self, D), LN(self, D)
            if chan_att:
                r = max(4, d // 4)
                lay["uc"] = self.P(D, 1)
                lay["Wc1"], lay["bc1"] = self.P(d, r), self.P(r, init="zeros")
                lay["Wc2"], lay["bc2"] = self.P(r, d), self.P(d, init="ones")
            self.layers.append(lay)
        if att_pool:
            self.Wr, self.br, self.ur = self.P(D, D), self.P(D, init="zeros"), self.P(D, 1)
        self.head = Head(self, (2 * D if att_pool else D) + (d if dual_view else 0), 2 * D, C)

    def adjacency(self):
        Ap = Tensor(self.A_prior)
        if self.adj == "prior":
            return Ap
        Aad = ag.softmax(ag.relu(ag.matmul(self.E1, ag.transpose(self.E2, (1, 0)))), -1)
        if self.adj == "adaptive":
            return Aad
        lam = ag.sigmoid(self.theta)
        return ag.add(ag.mul(lam, Ap), ag.mul(ag.add(ag.neg(lam), 1.0), Aad))

    def forward(self, x):
        self.cache = {}
        H = self.tok(x)
        A = self.adjacency()
        self.cache["A"] = A.data
        logA = ag.reshape(ag.log(A), (1, 1, self.d, self.d))
        for lay in self.layers:
            br = []
            if self.use_G:
                MG = ag.gelu(_lin(None, ag.matmul(A, H), lay["Wg"], lay["bg"]))
                br.append(MG)
            if self.use_S:
                MS = lay["att"](H, ag.mul(lay["gam"], logA))
                br.append(MS)
            if len(br) == 2:
                if self.gate:
                    z = ag.sigmoid(_lin(None, ag.concat(br, -1), lay["Wz"], lay["bz"]))
                    self.cache.setdefault("gate", []).append(z.data.mean())
                    F = ag.add(ag.mul(z, br[0]), ag.mul(ag.add(ag.neg(z), 1.0), br[1]))
                else:
                    F = ag.scale(ag.add(br[0], br[1]), 0.5)
            else:
                F = br[0]
            H = lay["ln1"](ag.add(H, ag.dropout(F, self.drop, self.train, self.rng)))
            H = lay["ln2"](ag.add(H, ag.dropout(lay["ffn"](H), self.drop, self.train, self.rng)))
            if self.chan_att:
                c = ag.reshape(ag.matmul(H, lay["uc"]), (H.shape[0], self.d))  # learned squeeze
                s = ag.sigmoid(_lin(None, ag.relu(_lin(None, c, lay["Wc1"], lay["bc1"])), lay["Wc2"], lay["bc2"]))
                self.cache.setdefault("chan", []).append(s.data)
                H = ag.mul(H, ag.expand_last(s))
        if self.att_pool:
            e = ag.matmul(ag.tanh(_lin(None, H, self.Wr, self.br)), self.ur)  # (B,d,1)
            a = ag.softmax(e, 1)
            self.cache["pool"] = a.data[..., 0]
            z = ag.concat([ag.sum_(ag.mul(a, H), 1), ag.mean(H, 1)], -1)
        else:
            z = ag.mean(H, 1)
        if self.dual_view:  # relational (graph) view + raw-feature view
            z = ag.concat([z, Tensor(x.astype(ag.DT))], -1)
        return self.head(z, self.drop)


# =============================================================================
# Baselines
# =============================================================================
class MLP(Module):
    def __init__(self, d, C, hid=(128, 64), drop=0.1, seed=0):
        super().__init__(seed)
        dims = (d,) + tuple(hid)
        self.Ws = [(self.P(a, b), self.P(b, init="zeros")) for a, b in zip(dims[:-1], dims[1:])]
        self.Wout, self.bout = self.P(dims[-1], C), self.P(C, init="zeros")
        self.drop = drop

    def forward(self, x):
        h = Tensor(x.astype(ag.DT))
        for W, b in self.Ws:
            h = ag.dropout(ag.gelu(_lin(None, h, W, b)), self.drop, self.train, self.rng)
        return _lin(None, h, self.Wout, self.bout)


class CNN1D(Module):
    def __init__(self, d, C, ch=(32, 64), k=3, drop=0.1, seed=0):
        super().__init__(seed)
        self.k, self.drop = k, drop
        cin = 1
        self.convs = []
        for c in ch:
            self.convs.append((self.P(k * cin, c), self.P(c, init="zeros")))
            cin = c
        self.head = Head(self, 2 * cin, 64, C)

    def forward(self, x):
        h = Tensor(x[:, :, None].astype(ag.DT))
        for W, b in self.convs:
            h = ag.gelu(_lin(None, ag.unfold1d(h, self.k), W, b))
        z = ag.concat([ag.mean(h, 1), ag.max_(h, 1)], -1)
        return self.head(z, self.drop)


class GCN(Module):
    def __init__(self, d, C, A_prior, D=32, L=2, drop=0.1, seed=0):
        super().__init__(seed)
        self.A = A_prior.astype(ag.DT)
        self.tok = Tokenizer(self, d, D)
        self.layers = [(self.P(D, D), self.P(D, init="zeros"), LN(self, D)) for _ in range(L)]
        self.head = Head(self, D, D, C)
        self.drop = drop

    def forward(self, x):
        H = self.tok(x)
        A = Tensor(self.A)
        for W, b, ln in self.layers:
            M = ag.gelu(_lin(None, ag.matmul(A, H), W, b))
            H = ln(ag.add(H, ag.dropout(M, self.drop, self.train, self.rng)))
        return self.head(ag.mean(H, 1), self.drop)


class _AttnNet(Module):
    def __init__(self, d, C, D=32, heads=4, L=2, drop=0.1, mask=None, seed=0):
        super().__init__(seed)
        self.tok = Tokenizer(self, d, D)
        self.blocks = [(MHSA(self, D, heads), LN(self, D), FFN(self, D), LN(self, D)) for _ in range(L)]
        self.head = Head(self, D, D, C)
        self.drop, self.mask = drop, mask

    def forward(self, x):
        self.cache = {}
        H = self.tok(x)
        bias = None if self.mask is None else Tensor(self.mask[None, None].astype(ag.DT))
        for att, l1, ff, l2 in self.blocks:
            H = l1(ag.add(H, ag.dropout(att(H, bias), self.drop, self.train, self.rng)))
            H = l2(ag.add(H, ag.dropout(ff(H), self.drop, self.train, self.rng)))
        return self.head(ag.mean(H, 1), self.drop)


class GATNet(_AttnNet):
    """Graph attention restricted to the (federated) correlation-graph support."""

    def __init__(self, d, C, A_prior, **kw):
        mask = np.where(A_prior > 0, 0.0, -1e9)
        super().__init__(d, C, mask=mask, **kw)


class FTTransformer(_AttnNet):
    """Feature-tokenizer Transformer (dense self-attention, no structural prior)."""
    pass


def build(name, d, C, A_prior, hp, seed=0):
    D, H, drop = hp.get("D", 32), hp.get("heads", 4), hp.get("drop", 0.1)
    if name == "MLP":
        return MLP(d, C, drop=drop, seed=seed)
    if name == "CNN1D":
        return CNN1D(d, C, drop=drop, seed=seed)
    if name == "GCN":
        return GCN(d, C, A_prior, D=D, drop=drop, seed=seed)
    if name == "GAT":
        return GATNet(d, C, A_prior, D=D, heads=H, drop=drop, seed=seed)
    if name == "Transformer":
        return FTTransformer(d, C, D=D, heads=H, drop=drop, seed=seed)
    if name.startswith("TAGGNN"):
        flags = dict(use_G=True, use_S=True, gate=True, chan_att=True, adj="both", att_pool=True,
                     periodic=True, dual_view=True)
        flags.update(hp.get("flags", {}))
        return TAGGNN(d, C, A_prior, D=D, heads=H, drop=drop, seed=seed, **flags)
    raise ValueError(name)
