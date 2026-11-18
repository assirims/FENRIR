"""
Data loading, federated-consistent preprocessing and non-IID client partitioning.
Primary dataset: RT-IoT2022 (UCI; real IoT/IIoT edge testbed traffic, 12 classes).
A loader for Edge-IIoTset (ML-EdgeIIoT-dataset.csv) is also provided.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def load_rt_iot2022(path=None, cap_majority=20000, seed=0):
    if path is None:
        for cand in [os.path.join(DATA_DIR, "RT_IOT2022"), os.path.join(HERE, "..", "04_Data", "RT_IOT2022_raw_UCI.zip")]:
            if os.path.exists(cand):
                path = cand
                break
    df = pd.read_csv(path, compression="zip" if str(path).endswith(".zip") else "infer")
    df = df.drop(columns=[c for c in ["Unnamed: 0", "id.orig_p"] if c in df.columns])
    lab = df.pop("Attack_type").astype(str)
    # cap the dominant DoS class (94,659 flows) to limit extreme imbalance
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), bool)
    for c, n in lab.value_counts().items():
        if n > cap_majority:
            idx = np.where(lab.values == c)[0]
            drop = rng.choice(idx, n - cap_majority, replace=False)
            keep[drop] = False
    df, lab = df[keep].reset_index(drop=True), lab[keep].reset_index(drop=True)
    cat = ["proto", "service"]
    df = pd.get_dummies(df, columns=cat, prefix=cat, dtype=float)
    return df, lab


def load_edge_iiotset(path, seed=0):
    """Edge-IIoTset 'ML-EdgeIIoT-dataset.csv' (Ferrag et al., 2022). Standard cleaning."""
    df = pd.read_csv(path, low_memory=False)
    drop = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
            "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
            "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg", "Attack_label"]
    df = df.drop(columns=[c for c in drop if c in df.columns]).dropna().drop_duplicates()
    lab = df.pop("Attack_type").astype(str)
    cat = [c for c in ["http.request.method", "http.referer", "http.request.version", "dns.qry.name.len",
                       "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat, prefix=cat, dtype=float)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.reset_index(drop=True), lab.reset_index(drop=True)


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def stratified_split(y, frac, rng):
    a, b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = int(round(len(idx) * frac))
        if len(idx) >= 2:
            k = min(max(k, 1), len(idx) - 1)
        a.extend(idx[:k]); b.extend(idx[k:])
    return np.array(sorted(a)), np.array(sorted(b))


def dirichlet_partition(y, K, alpha, rng, min_size=50):
    C = int(y.max()) + 1
    for _ in range(200):
        parts = [[] for _ in range(K)]
        for c in range(C):
            idx = rng.permutation(np.where(y == c)[0])
            if alpha is None:  # IID
                p = np.ones(K) / K
            else:
                p = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    return [np.array(sorted(p)) for p in parts]


class FedData:
    """Container produced by `prepare`. Only sufficient statistics leave clients."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def prepare(dataset="rt-iot2022", K=10, alpha=0.5, seed=42, test_frac=0.2, val_frac=0.15, path=None):
    rng = np.random.default_rng(seed)
    if dataset == "rt-iot2022":
        df, lab = load_rt_iot2022(path, seed=0)
    else:
        df, lab = load_edge_iiotset(path, seed=0)
    classes = sorted(lab.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = lab.map(cmap).values.astype(int)
    X = signed_log(df.values.astype(np.float64))
    feat = list(df.columns)
    te, tr = stratified_split(y, test_frac, np.random.default_rng(0))  # fixed global test split
    parts = dirichlet_partition(y[tr], K, alpha, rng)
    clients = []
    for p in parts:
        gidx = tr[p]
        yl = y[gidx]
        # local stratified train/validation split (classes with 1 sample stay in train)
        v, t = [], []
        lr = np.random.default_rng(seed + len(clients))
        for c in np.unique(yl):
            ii = lr.permutation(np.where(yl == c)[0])
            k = int(np.floor(len(ii) * val_frac)) if len(ii) >= 4 else 0
            v.extend(ii[:k]); t.extend(ii[k:])
        clients.append({"tr": gidx[np.array(sorted(t))], "va": gidx[np.array(sorted(v), dtype=int)]})
    # ---- federated sufficient statistics (secure-aggregatable sums) -------------
    n = sum(len(c["tr"]) for c in clients)
    s1 = sum(X[c["tr"]].sum(0) for c in clients)
    s2 = sum((X[c["tr"]] ** 2).sum(0) for c in clients)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    keep = sd > 1e-8
    Xs = np.clip((X[:, keep] - mu[keep]) / sd[keep], -10, 10).astype(np.float32)
    feat = [f for f, k in zip(feat, keep) if k]
    sxx = sum(Xs[c["tr"]].T.astype(np.float64) @ Xs[c["tr"]] for c in clients)
    cov = sxx / n  # data are standardised -> covariance == correlation
    dsd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(dsd, dsd)
    counts = sum(np.bincount(y[c["tr"]], minlength=len(classes)) for c in clients)
    return FedData(X=Xs, y=y, classes=classes, features=feat, clients=clients, test=te,
                   corr=corr, class_counts=counts, K=K, alpha=alpha, seed=seed)


def prior_adjacency(corr, sel, topk):
    """Row-normalised top-k |rho| graph with self-loops over the selected features."""
    C = np.abs(corr[np.ix_(sel, sel)]).copy()
    np.fill_diagonal(C, 0)
    d = len(sel)
    k = int(min(max(topk, 1), d - 1))
    A = np.zeros_like(C)
    for i in range(d):
        nb = np.argsort(-C[i])[:k]
        A[i, nb] = C[i, nb] + 1e-3
    A = np.maximum(A, A.T) + np.eye(d)
    return A / A.sum(1, keepdims=True)


def class_weights(counts, power=0.5):
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return (w / w.mean()).astype(np.float32)
