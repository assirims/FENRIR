"""
Minimal reverse-mode automatic differentiation engine in NumPy.
Used to implement and train all neural models of the FENRIR framework
(no external deep-learning framework is required).
"""
import numpy as np

DT = np.float32


def set_dtype(dt):
    global DT
    DT = dt


def _unb(g, shape):
    """Sum a broadcast gradient back to `shape`."""
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Tensor:
    __slots__ = ("data", "grad", "_prev", "_back", "req", "name")

    def __init__(self, data, prev=(), back=None, req=False, name=None):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=DT)
        self.data = data
        self.grad = None
        self._prev = prev
        self._back = back
        self.req = req
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        if not self.req:
            return
        self.grad = g if self.grad is None else self.grad + g

    def backward(self, g=None):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for p in v._prev:
                if p.req and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data) if g is None else g
        for v in reversed(topo):
            if v._back is not None and v.grad is not None:
                v._back(v.grad)

    # operator sugar
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(self, o)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(self, o)
    def __sub__(self, o): return add(self, neg(_t(o)))
    def __neg__(self): return neg(self)
    def __matmul__(self, o): return matmul(self, o)


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=DT))


def Param(arr):
    return Tensor(arr.astype(DT), req=True)


# ---------------------------------------------------------------- elementwise
def add(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data + b.data, (a, b), None, a.req or b.req)

    def back(g):
        a._acc(_unb(g, a.shape))
        b._acc(_unb(g, b.shape))
    out._back = back
    return out


def mul(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data * b.data, (a, b), None, a.req or b.req)

    def back(g):
        if a.req:
            a._acc(_unb(g * b.data, a.shape))
        if b.req:
            b._acc(_unb(g * a.data, b.shape))
    out._back = back
    return out


def neg(a):
    out = Tensor(-a.data, (a,), None, a.req)
    out._back = lambda g: a._acc(-g)
    return out


def scale(a, s):
    out = Tensor(a.data * s, (a,), None, a.req)
    out._back = lambda g: a._acc(g * s)
    return out


def matmul(a, b):
    a, b = _t(a), _t(b)
    A, Bm = a.data, b.data
    if A.ndim >= 3 and Bm.ndim == 2:  # (..., K) @ (K, E): fold batch dims into one GEMM
        K, E = Bm.shape
        A2 = A.reshape(-1, K)
        out = Tensor((A2 @ Bm).reshape(A.shape[:-1] + (E,)), (a, b), None, a.req or b.req)

        def back(g):
            g2 = g.reshape(-1, E)
            if a.req:
                a._acc((g2 @ Bm.T).reshape(A.shape))
            if b.req:
                b._acc(A2.T @ g2)
        out._back = back
        return out
    if A.ndim == 2 and Bm.ndim == 3:  # (d,d) @ (B,d,D): graph propagation
        out = Tensor(np.matmul(A, Bm), (a, b), None, a.req or b.req)

        def back(g):
            if a.req:
                a._acc(np.tensordot(g, Bm, axes=([0, 2], [0, 2])))
            if b.req:
                b._acc(np.matmul(A.T, g))
        out._back = back
        return out
    out = Tensor(np.matmul(A, Bm), (a, b), None, a.req or b.req)

    def back(g):
        if a.req:
            ga = np.matmul(g, np.swapaxes(Bm, -1, -2)) if Bm.ndim > 1 else np.outer(g, Bm)
            a._acc(_unb(ga, a.shape))
        if b.req:
            gb = np.matmul(np.swapaxes(A, -1, -2), g)
            b._acc(_unb(gb, b.shape))
    out._back = back
    return out


def relu(a):
    m = a.data > 0
    out = Tensor(a.data * m, (a,), None, a.req)
    out._back = lambda g: a._acc(g * m)
    return out


def leaky_relu(a, s=0.2):
    m = np.where(a.data > 0, 1.0, s).astype(a.data.dtype)
    out = Tensor(a.data * m, (a,), None, a.req)
    out._back = lambda g: a._acc(g * m)
    return out


def sigmoid(a):
    y = 1.0 / (1.0 + np.exp(-np.clip(a.data, -30, 30)))
    out = Tensor(y, (a,), None, a.req)
    out._back = lambda g: a._acc(g * y * (1 - y))
    return out


def tanh(a):
    y = np.tanh(a.data)
    out = Tensor(y, (a,), None, a.req)
    out._back = lambda g: a._acc(g * (1 - y * y))
    return out


def gelu(a):
    x = a.data
    c = float(np.sqrt(2.0 / np.pi))
    x2 = x * x
    u = c * x * (1 + 0.044715 * x2)
    t = np.tanh(u)
    y = 0.5 * x * (1 + t)
    out = Tensor(y.astype(x.dtype), (a,), None, a.req)

    def back(g):
        du = c * (1 + 3 * 0.044715 * x2)
        d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du
        a._acc((g * d).astype(x.dtype))
    out._back = back
    return out


def sin(a):
    out = Tensor(np.sin(a.data), (a,), None, a.req)
    out._back = lambda g: a._acc(g * np.cos(a.data))
    return out


def cos(a):
    out = Tensor(np.cos(a.data), (a,), None, a.req)
    out._back = lambda g: a._acc(-g * np.sin(a.data))
    return out


def log(a, eps=1e-6):
    y = np.log(a.data + eps)
    out = Tensor(y, (a,), None, a.req)
    out._back = lambda g: a._acc(g / (a.data + eps))
    return out


# ---------------------------------------------------------------- reductions / shape
def sum_(a, axis=None, keepdims=False):
    y = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(np.asarray(y, dtype=a.data.dtype), (a,), None, a.req)

    def back(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._acc(np.broadcast_to(g, a.shape).astype(a.data.dtype))
    out._back = back
    return out


def mean(a, axis=None, keepdims=False):
    n = a.data.size if axis is None else a.data.shape[axis]
    return scale(sum_(a, axis, keepdims), 1.0 / n)


def max_(a, axis):
    y = a.data.max(axis=axis, keepdims=True)
    m = (a.data == y).astype(a.data.dtype)
    m /= m.sum(axis=axis, keepdims=True)
    out = Tensor(np.squeeze(y, axis), (a,), None, a.req)
    out._back = lambda g: a._acc(np.expand_dims(g, axis) * m)
    return out


def reshape(a, shape):
    s0 = a.shape
    out = Tensor(a.data.reshape(shape), (a,), None, a.req)
    out._back = lambda g: a._acc(g.reshape(s0))
    return out


def transpose(a, axes):
    inv = np.argsort(axes)
    out = Tensor(np.transpose(a.data, axes), (a,), None, a.req)
    out._back = lambda g: a._acc(np.transpose(g, inv))
    return out


def concat(ts, axis=-1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts), None, any(t.req for t in ts))
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]

    def back(g):
        for t, gg in zip(ts, np.split(g, sizes, axis=axis)):
            t._acc(gg)
    out._back = back
    return out


def expand_last(a):
    return reshape(a, a.shape + (1,))


# ---------------------------------------------------------------- NN primitives
def softmax(a, axis=-1):
    x = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(x)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(y, (a,), None, a.req)

    def back(g):
        a._acc(y * (g - (g * y).sum(axis=axis, keepdims=True)))
    out._back = back
    return out


def layernorm(a, gamma, beta, eps=1e-5):
    x = a.data
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    y = xh * gamma.data + beta.data
    out = Tensor(y.astype(x.dtype), (a, gamma, beta), None, True)

    def back(g):
        if gamma.req:
            gamma._acc(_unb(g * xh, gamma.shape))
        if beta.req:
            beta._acc(_unb(g, beta.shape))
        if a.req:
            gx = g * gamma.data
            n = x.shape[-1]
            dx = inv / n * (n * gx - gx.sum(-1, keepdims=True) - xh * (gx * xh).sum(-1, keepdims=True))
            a._acc(dx.astype(x.dtype))
    out._back = back
    return out


def dropout(a, p, train, rng):
    if not train or p <= 0:
        return a
    m = (rng.random(a.shape) >= p).astype(a.data.dtype) / (1 - p)
    out = Tensor(a.data * m, (a,), None, a.req)
    out._back = lambda g: a._acc(g * m)
    return out


def unfold1d(a, k):
    """(B, L, C) -> (B, L, k*C) with zero 'same' padding."""
    B, L, C = a.shape
    p = k // 2
    xp = np.pad(a.data, ((0, 0), (p, p), (0, 0)))
    idx = np.arange(L)[:, None] + np.arange(k)[None, :]  # (L,k)
    y = xp[:, idx, :].reshape(B, L, k * C)
    out = Tensor(y, (a,), None, a.req)

    def back(g):
        g = g.reshape(B, L, k, C)
        gp = np.zeros_like(xp)
        for j in range(k):
            gp[:, j:j + L, :] += g[:, :, j, :]
        a._acc(gp[:, p:p + L, :])
    out._back = back
    return out


def cross_entropy(logits, y, class_w=None, smoothing=0.0):
    x = logits.data
    x = x - x.max(-1, keepdims=True)
    lse = np.log(np.exp(x).sum(-1, keepdims=True))
    logp = x - lse
    B, C = x.shape
    tgt = np.full((B, C), smoothing / C, dtype=x.dtype)
    tgt[np.arange(B), y] += 1.0 - smoothing
    w = np.ones(B, dtype=x.dtype) if class_w is None else class_w[y].astype(x.dtype)
    wn = w / w.sum()
    loss = -(wn[:, None] * tgt * logp).sum()
    out = Tensor(np.asarray(loss, dtype=x.dtype), (logits,), None, logits.req)

    def back(g):
        p = np.exp(logp)
        logits._acc((g * wn[:, None] * (p - tgt)).astype(x.dtype))
    out._back = back
    return out


# ---------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(q.data) for q in params]
        self.v = [np.zeros_like(q.data) for q in params]
        self.t = 0

    def zero(self):
        for q in self.p:
            q.grad = None

    def step(self, extra_grads=None, clip=5.0):
        self.t += 1
        # global-norm clipping
        gs = [q.grad if q.grad is not None else np.zeros_like(q.data) for q in self.p]
        if extra_grads is not None:
            gs = [g + e for g, e in zip(gs, extra_grads)]
        nrm = np.sqrt(sum(float((g * g).sum()) for g in gs))
        c = min(1.0, clip / (nrm + 1e-12))
        b1t, b2t = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, (q, g) in enumerate(zip(self.p, gs)):
            g = g * c
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            upd = (self.m[i] / b1t) / (np.sqrt(self.v[i] / b2t) + self.eps)
            if self.wd > 0:
                upd = upd + self.wd * q.data
            q.data -= (self.lr * upd).astype(q.data.dtype)
