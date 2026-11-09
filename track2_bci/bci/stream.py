"""Streaming Track 2 model, self-contained (numpy, scipy, scikit-learn, torch).

This file is copied verbatim into the Codabench ``submission.py`` by
``bci.export``, so it must not import anything from this project. Training
(``bci.ensemble``) uses the same step functions, so a model is scored in
cross-validation exactly as it runs on the platform.

One window at a time, per session (``reset_state()`` starts a session):

1. Riemannian experts: band-pass, OAS covariance, a channel subset, causal
   recentering at a running log-Euclidean mean that starts from the expert's
   training mean, tangent space, linear softmax head. Global experts are
   trained on everyone; subject experts on one training participant each.
2. Optional deep member: ShallowConvNet on Euclidean-aligned windows (running
   mean covariance of the session), mixed into the global experts.
3. Optional routing: the session's raw running mean covariance is matched to
   the training participants' session means. If the nearest participant is
   clearly nearer than the second one (ratio test), that participant's experts
   take over in proportion to the confidence; a new person keeps the global ones.
4. Optional balance: the trials of a run are balanced over the classes, so the
   probabilities are divided by the session's running mean prediction (to the
   power ``alpha``), which undoes a drift towards one class.
5. Optional transduction: the classes of a session are balanced and form
   clusters in the session's own feature space even when a cross-subject
   model misplaces them. Full-channel filter-bank tangent vectors of the
   session so far are projected on their whitened principal axes; a
   class-balanced (Sinkhorn) assignment, started from the ensemble's
   probabilities, and an LDA fitted on it are iterated; the current window
   gets its row of the final assignment.
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


def _matrix_function(C, f):
    """f(C) for a symmetric matrix or a stack of them, through the eigendecomposition (as pyriemann)."""
    eigvals, eigvecs = np.linalg.eigh(C)
    return (eigvecs * f(eigvals)[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)


def logm(C):
    return _matrix_function(C, np.log)


def expm(C):
    return _matrix_function(C, np.exp)


def invsqrtm(C):
    return _matrix_function(C, lambda v: 1.0 / np.sqrt(v))


_UPPER = {}


def upper(C):
    """Upper triangle as a vector (of each matrix of a stack), off-diagonal entries times sqrt(2) (as pyriemann)."""
    n = C.shape[-1]
    if n not in _UPPER:
        idx = np.triu_indices(n)
        _UPPER[n] = idx, (np.sqrt(2) * np.triu(np.ones((n, n)), 1) + np.eye(n))[idx]
    idx, coeffs = _UPPER[n]
    return coeffs * C[..., idx[0], idx[1]]


def log_softmax(z):
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def oas_covariance(x):
    """OAS covariance of one window x (C, T), the formula of scikit-learn 1.9 ``oas`` (samples = time points)."""
    x = x - x.mean(1, keepdims=True)
    n_samples, n_features = x.shape[1], x.shape[0]
    S = x @ x.T / n_samples
    alpha = np.mean(S ** 2)
    mu = np.trace(S) / n_features
    num = alpha + mu ** 2
    den = (n_samples + 1) * (alpha - mu ** 2 / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    S = (1.0 - shrinkage) * S
    S.flat[:: n_features + 1] += shrinkage * mu
    return S


def bandpass(band, sfreq):
    return butter(4, band, btype="bandpass", fs=sfreq, output="sos")


def covariances(X, sos):
    """(N, C, T) -> (N, C, C): band-pass, then the OAS estimator of pyriemann's Covariances("oas")."""
    X = sosfiltfilt(sos, np.asarray(X, np.float64), axis=-1)
    return np.stack([oas_covariance(x) for x in X])


def tangent_step(state, cov, log_cov=None):
    """Tangent vector of ``cov`` at the session's running log-Euclidean mean; updates ``state`` = [log_sum, n]."""
    state[0] = state[0] + (logm(cov) if log_cov is None else log_cov)
    state[1] = state[1] + 1
    n = np.asarray(state[1], np.float64)[..., None, None]
    W = expm(-state[0] / (2 * n))  # = invsqrtm(expm(mean)), one eigendecomposition
    return upper(logm(W @ cov @ W))


def balanced_assignment(logits, n_iter=30):
    """Rows of exp(logits) rescaled (Sinkhorn) so that every class receives the same total mass."""
    K = np.exp(logits - logits.max(1, keepdims=True))
    u, v = np.ones(len(K)), np.ones(K.shape[1])
    mass = np.full(K.shape[1], len(K) / K.shape[1])
    for _ in range(n_iter):
        u = 1 / (K @ v)
        v = mass / (K.T @ u)
    q = u[:, None] * K * v[None]
    return q / q.sum(1, keepdims=True)


def self_label(gram, log_prior, beta=1.0, k=20, iters=5, shrink=0.5):
    """Balanced self-labelling of a session: (n, n) Gram matrix of its features, (n, K) prior log-probabilities.

    Whitened principal coordinates come from the centred Gram matrix (eigenvectors times sqrt(n)), so the
    feature dimension never enters. Alternates an LDA on the soft labels with a balanced assignment.
    """
    n = len(gram)
    k = max(1, min(k, n // 3))
    centred = gram - gram.mean(0)[None] - gram.mean(1)[:, None] + gram.mean()
    _, vecs = eigh(centred, subset_by_index=[n - k, n - 1])  # only the k leading axes
    P = vecs[:, ::-1] * np.sqrt(n)
    q = balanced_assignment(log_prior)
    for _ in range(iters):
        mu = (q.T @ P) / q.sum(0)[:, None]
        R = P - q @ mu
        S = R.T @ R / n
        S = (1 - shrink) * S + shrink * np.trace(S) / k * np.eye(k)
        Si = np.linalg.inv(S)
        loglik = P @ Si @ mu.T - 0.5 * np.einsum("cd,de,ce->c", mu, Si, mu)[None]
        q = balanced_assignment(log_prior + beta * loglik)
    return q


def align_step(state, x):
    """Euclidean alignment of one filtered window x (C, T) by the session's running mean covariance."""
    state[0] = state[0] + x @ x.T / x.shape[-1]
    state[1] += 1
    return invsqrtm(state[0] / state[1]) @ x


class ShallowNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017) at 120 Hz: temporal conv, spatial conv, square, mean, log."""

    def __init__(self, n_chans, n_times, n_classes=3, n_filters=40, kernel=13, pool=35, stride=7, dropout=0.5):
        super().__init__()
        self.config = dict(n_chans=n_chans, n_times=n_times, n_classes=n_classes, n_filters=n_filters,
                           kernel=kernel, pool=pool, stride=stride, dropout=dropout)
        self.temporal = nn.Conv2d(1, n_filters, (1, kernel))
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False)
        self.norm = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * ((n_times - kernel + 1 - pool) // stride + 1), n_classes)

    def forward(self, x):
        h = self.norm(self.spatial(self.temporal(x[:, None])))
        h = torch.log(torch.clamp(self.pool(h * h), min=1e-6))
        return self.out(self.drop(h).flatten(1))


def save_spec(spec, path):
    """A nested dict/list spec -> .npz: arrays stored as such, the rest as JSON (nothing pickled)."""
    arrays = {}

    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [strip(x) for x in v]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            key = f"a{len(arrays)}"
            arrays[key] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            return "@" + key
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    skeleton = json.dumps(strip(spec))
    np.savez(path, _skeleton=np.array(skeleton), **arrays)


def load_spec(path):
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}

    def fill(v):
        if isinstance(v, dict):
            return {k: fill(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fill(x) for x in v]
        if isinstance(v, str) and v.startswith("@a"):
            return arrays[v[1:]]
        return v

    return fill(json.loads(str(arrays.pop("_skeleton"))))


class StreamModel:
    """Track 2 contract: ``reset_state()`` per session, ``predict(X)`` for X (B, C, T) in recording order."""

    def __init__(self, spec, channel_index=None):
        self.spec, self.channel_index = spec, channel_index
        self.sos = [bandpass(b, spec["sfreq"]) for b in spec["bands"]]
        self.experts = spec["experts"]
        self.groups = np.array([e["group"] for e in self.experts])
        self.blocks = self._stack_experts(spec)
        self.nets = []
        if spec.get("deep"):
            deep = spec["deep"]
            self.deep_sos = bandpass(deep["band"], spec["sfreq"])
            for member in deep["nets"]:
                net = ShallowNet(**member["config"])
                net.load_state_dict({k: torch.as_tensor(v) for k, v in member["state"].items()})
                self.nets.append(net.eval())
        self.reset_state()

    @staticmethod
    def _stack_experts(spec):
        """Experts with the same number of channels, stacked so that one window is one batched computation."""
        n_chans = len(spec["ch_names"])
        channels = [np.arange(n_chans) if e["channels"] is None else np.asarray(e["channels"]) for e in spec["experts"]]
        blocks = []
        for size in sorted({len(c) for c in channels}):
            idx = [i for i, c in enumerate(channels) if len(c) == size]
            ex = [spec["experts"][i] for i in idx]
            blocks.append({
                "idx": np.array(idx), "band": np.array([e["band"] for e in ex]),
                "channels": np.stack([channels[i] for i in idx]),
                "prior_weight": np.array([e["prior_weight"] for e in ex], np.float64),
                "log_prior": np.stack([np.asarray(e["log_prior"]) for e in ex]),
                "x_mean": np.stack([e["x_mean"] for e in ex]), "x_scale": np.stack([e["x_scale"] for e in ex]),
                "coef": np.stack([e["coef"] for e in ex]), "intercept": np.stack([e["intercept"] for e in ex])})
        return blocks

    def reset_state(self):
        self.block_states = [[b["prior_weight"][:, None, None] * b["log_prior"], b["prior_weight"].copy()]
                             for b in self.blocks]
        if self.nets:
            deep = self.spec["deep"]
            self.align_state = [deep["prior_weight"] * np.asarray(deep["cov_prior"]), deep["prior_weight"]]
        self.route_state = [0.0, 0]
        self.balance_state = [0.0, 0]
        self.known, self.subject = 0.0, -1  # routing outcome after the last window
        t = self.spec.get("transduction")
        if t:
            self.trans_states = [[t["prior_weight"] * np.asarray(lp), t["prior_weight"]] for lp in t["log_priors"]]
            self.trans_feats, self.trans_logp = [], []
            self.gram = np.zeros((0, 0))

    def predict(self, X):
        X = np.asarray(X.detach().cpu() if hasattr(X, "detach") else X, np.float64)
        if self.channel_index is not None:
            X = X[:, self.channel_index]
        return np.array([int(np.argmax(self.proba(x))) for x in X], dtype=np.int64)

    def proba(self, x):
        """Class probabilities of one window x (C, T); updates the session state."""
        covs = {}

        def cov(band):
            if band not in covs:
                covs[band] = covariances(x[None], self.sos[band])[0]
            return covs[band]

        probas = np.empty((len(self.experts), self.spec.get("n_classes", 3)))
        for block, state in zip(self.blocks, self.block_states):
            bands = np.unique(block["band"])
            full = np.stack([cov(b) for b in bands])
            ch = block["channels"]
            subs = full[np.searchsorted(bands, block["band"])[:, None, None], ch[:, :, None], ch[:, None, :]]
            feats = (tangent_step(state, subs) - block["x_mean"]) / block["x_scale"]
            z = np.einsum("ekd,ed->ek", block["coef"], feats) + block["intercept"]
            probas[block["idx"]] = np.exp(log_softmax(z))
        p = probas[self.groups < 0].mean(0)
        if self.nets:
            w, deep = self.spec["deep"]["weight"], self._deep(x)
            if self.spec["deep"].get("pool") == "log":  # geometric mean: either member can veto a class
                z = (1 - w) * np.log(p + 1e-9) + w * np.log(deep + 1e-9)
                p = np.exp(z - z.max())
                p = p / p.sum()
            else:
                p = (1 - w) * p + w * deep
        if self.spec.get("router"):
            known, weights = self._route(cov(self.spec["router"]["band"]))
            subject = np.stack([probas[self.groups == k].mean(0) for k in range(len(weights))])
            p = (1 - known) * p + known * weights @ subject
        prior = p
        if self.spec.get("balance"):
            p = self._balance(p)
        if self.spec.get("transduction"):
            p = self._transduce(prior, p, cov)
        return p

    def _transduce(self, prior, p, cov):
        t = self.spec["transduction"]
        z = np.concatenate([tangent_step(s, cov(b)) for s, b in zip(self.trans_states, t["bands"])])
        cross = np.array([f @ z for f in self.trans_feats] + [z @ z])
        n = len(cross)
        gram = np.empty((n, n))
        gram[:-1, :-1], gram[-1], gram[:, -1] = self.gram, cross, cross
        self.gram = gram
        self.trans_feats.append(z)
        self.trans_logp.append(np.log(prior + 1e-9))
        if n < t["warm"]:
            return p
        q = self_label(gram, np.array(self.trans_logp), t["beta"], t["k"], t["iters"], t["shrink"])
        return q[-1]

    @torch.no_grad()
    def _deep(self, x):
        xa = align_step(self.align_state, sosfiltfilt(self.deep_sos, x, axis=-1))
        crop = self.spec["deep"].get("crop")
        if crop:  # evenly spaced crops, averaged
            starts = np.linspace(0, xa.shape[-1] - crop, self.spec["deep"]["n_crops"]).astype(int)
            xa = np.stack([xa[:, s:s + crop] for s in starts])
        else:
            xa = xa[None]
        xa = torch.as_tensor(xa, dtype=torch.float32)
        return np.mean([torch.softmax(net(xa), -1).mean(0).numpy() for net in self.nets], axis=0)

    def _route(self, cov):
        """(confidence that the session is a known participant, weights over the training participants)."""
        r, s = self.spec["router"], self.route_state
        s[0] = s[0] + logm(cov)
        s[1] += 1
        d_t = np.linalg.norm(np.asarray(r["templates"]) - s[0] / s[1], axis=(1, 2))
        owner = np.asarray(r["template_subject"])
        d = np.array([d_t[owner == k].min() for k in range(r["n_subjects"])])
        first, second = np.maximum(np.sort(d)[:2], 1e-9)  # 0 only for a stream equal to a template
        known = 0.0
        if s[1] >= r["min_windows"]:
            known = 1 / (1 + math.exp(-(r["ratio"] - first / second) / r["width"]))
        logits = -(d / first - 1) / r["sharpness"]
        weights = np.exp(log_softmax(logits))
        self.known, self.subject = known, int(np.argmin(d))
        return known, weights

    def _balance(self, p):
        b, s = self.spec["balance"], self.balance_state
        s[0] = s[0] + p
        s[1] += 1
        prior = (s[0] + b["prior_windows"] / len(p)) / (s[1] + b["prior_windows"])
        q = p / prior ** b["alpha"]
        return q / q.sum()
