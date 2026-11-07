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
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp
from sklearn.covariance import oas


def _matrix_function(C, f):
    """f(C) for a symmetric matrix, through its eigendecomposition (as pyriemann)."""
    eigvals, eigvecs = np.linalg.eigh(C)
    return (eigvecs * f(eigvals)) @ eigvecs.T


def logm(C):
    return _matrix_function(C, np.log)


def expm(C):
    return _matrix_function(C, np.exp)


def invsqrtm(C):
    return _matrix_function(C, lambda v: 1.0 / np.sqrt(v))


def upper(C):
    """Upper triangle as a vector, off-diagonal entries times sqrt(2) (as pyriemann)."""
    n = C.shape[-1]
    idx = np.triu_indices(n)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n, n)), 1) + np.eye(n))[idx]
    return coeffs * C[idx]


def bandpass(band, sfreq):
    return butter(4, band, btype="bandpass", fs=sfreq, output="sos")


def covariances(X, sos):
    """(N, C, T) -> (N, C, C): band-pass, then the OAS estimator of pyriemann's Covariances("oas")."""
    X = sosfiltfilt(sos, np.asarray(X, np.float64), axis=-1)
    return np.stack([oas(x.T)[0] for x in X])


def tangent_step(state, cov, log_cov=None):
    """Tangent vector of ``cov`` at the session's running log-Euclidean mean; updates ``state`` = [log_sum, n]."""
    state[0] = state[0] + (logm(cov) if log_cov is None else log_cov)
    state[1] += 1
    W = invsqrtm(expm(state[0] / state[1]))
    return upper(logm(W @ cov @ W))


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
        self.nets = []
        if spec.get("deep"):
            deep = spec["deep"]
            self.deep_sos = bandpass(deep["band"], spec["sfreq"])
            for member in deep["nets"]:
                net = ShallowNet(**member["config"])
                net.load_state_dict({k: torch.as_tensor(v) for k, v in member["state"].items()})
                self.nets.append(net.eval())
        self.reset_state()

    def reset_state(self):
        self.expert_states = [[e["prior_weight"] * np.asarray(e["log_prior"]), e["prior_weight"]]
                              for e in self.experts]
        if self.nets:
            deep = self.spec["deep"]
            self.align_state = [deep["prior_weight"] * np.asarray(deep["cov_prior"]), deep["prior_weight"]]
        self.route_state = [0.0, 0]
        self.balance_state = [0.0, 0]
        self.known, self.subject = 0.0, -1  # routing outcome after the last window

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

        probas = np.stack([np.exp(self._expert(e, s, cov(e["band"]))) for e, s in zip(self.experts, self.expert_states)])
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
        if self.spec.get("balance"):
            p = self._balance(p)
        return p

    def _expert(self, e, state, cov):
        channels = e["channels"]
        if channels is not None:
            cov = cov[np.ix_(channels, channels)]
        z = e["coef"] @ ((tangent_step(state, cov) - e["x_mean"]) / e["x_scale"]) + e["intercept"]
        return z - logsumexp(z)

    @torch.no_grad()
    def _deep(self, x):
        xa = align_step(self.align_state, sosfiltfilt(self.deep_sos, x, axis=-1))
        xa = torch.as_tensor(xa[None], dtype=torch.float32)
        return np.mean([torch.softmax(net(xa), -1)[0].numpy() for net in self.nets], axis=0)

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
        weights = np.exp(logits - logsumexp(logits))
        self.known, self.subject = known, int(np.argmin(d))
        return known, weights

    def _balance(self, p):
        b, s = self.spec["balance"], self.balance_state
        s[0] = s[0] + p
        s[1] += 1
        prior = (s[0] + b["prior_windows"] / len(p)) / (s[1] + b["prior_windows"])
        q = p / prior ** b["alpha"]
        return q / q.sum()
