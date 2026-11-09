"""Causal class-balanced transduction within one session (numpy + scipy only).

Two independent pieces:

- ``TangentFeatures``: one EEG window -> filter-bank tangent vector, recentred
  on the session's running log-Euclidean mean (causal, no labels);
- ``Transducer``: feature vector + any model's class probabilities for that
  window -> refined probabilities, using every window of the session so far.

    feats, trans = TangentFeatures(sfreq=120), Transducer()
    for x, p in session:            # x: (channels, times), p: (n_classes,)
        q = trans.step(feats.step(x), p)
    feats.reset(); trans.reset()    # before the next session

``bci/stream.py`` holds the same computation inlined into the Codabench
submission; ``check_parity.py`` checks that both agree.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (4, 40))


# ---------------------------------------------------------------- balanced self-labelling

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
    """Balanced labels of n windows from their (n, n) Gram matrix and (n, K) prior log-probabilities.

    Coordinates: the k leading whitened principal axes of the centred Gram matrix. Then an LDA on
    the soft labels and a balanced assignment of prior + LDA log-likelihood, alternated ``iters`` times.
    """
    n = len(gram)
    k = max(1, min(k, n // 3))
    centred = gram - gram.mean(0)[None] - gram.mean(1)[:, None] + gram.mean()
    _, vecs = eigh(centred, subset_by_index=[n - k, n - 1])
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


class Transducer:
    """Online wrapper of ``self_label``: keeps the session's features and priors, grows the Gram matrix."""

    def __init__(self, beta=1.0, k=20, iters=5, shrink=0.5, warm=20):
        self.beta, self.k, self.iters, self.shrink, self.warm = beta, k, iters, shrink, warm
        self.reset()

    def reset(self):
        self.feats, self.log_prior, self.gram = [], [], np.zeros((0, 0))

    def step(self, z, proba):
        """Probabilities for the newest window, given its features ``z`` and a model's ``proba``."""
        z = np.asarray(z, np.float64)
        cross = np.array([f @ z for f in self.feats] + [z @ z])
        n = len(cross)
        gram = np.empty((n, n))
        gram[:-1, :-1], gram[-1], gram[:, -1] = self.gram, cross, cross
        self.gram = gram
        self.feats.append(z)
        self.log_prior.append(np.log(np.asarray(proba, np.float64) + 1e-9))
        if n < self.warm:
            return np.asarray(proba, np.float64)
        return self_label(gram, np.array(self.log_prior), self.beta, self.k, self.iters, self.shrink)[-1]


# ---------------------------------------------------------------- causal tangent features

def _eig_function(C, f):
    w, V = np.linalg.eigh(C)
    return (V * f(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def logm(C):
    return _eig_function(C, np.log)


def expm(C):
    return _eig_function(C, np.exp)


def upper(C):
    """Upper triangle, off-diagonal entries times sqrt(2) (as pyriemann)."""
    n = C.shape[-1]
    i, j = np.triu_indices(n)
    return np.where(i == j, 1.0, np.sqrt(2)) * C[..., i, j]


def oas_covariance(x):
    """OAS-shrunk covariance of one window x (channels, times)."""
    x = x - x.mean(1, keepdims=True)
    n_samples, n_features = x.shape[1], x.shape[0]
    S = x @ x.T / n_samples
    alpha, mu = np.mean(S ** 2), np.trace(S) / n_features
    den = (n_samples + 1) * (alpha - mu ** 2 / n_features)
    shrinkage = 1.0 if den == 0 else min((alpha + mu ** 2) / den, 1.0)
    S = (1.0 - shrinkage) * S
    S.flat[:: n_features + 1] += shrinkage * mu
    return S


class TangentFeatures:
    """Filter-bank tangent vectors of single windows, recentred on the session's running log-Euclidean mean.

    ``log_priors``: one (channels, channels) matrix per band, the mean log-covariance of training windows,
    counted as ``prior_weight`` windows at the start of every session (identity, i.e. zeros, if absent).
    """

    def __init__(self, sfreq, bands=BANDS, log_priors=None, prior_weight=5):
        self.sos = [butter(4, b, btype="bandpass", fs=sfreq, output="sos") for b in bands]
        self.log_priors, self.prior_weight = log_priors, prior_weight
        self.reset()

    def reset(self):
        self.state = None

    def step(self, x):
        x = np.asarray(x, np.float64)
        covs = np.stack([oas_covariance(sosfiltfilt(sos, x, axis=-1)) for sos in self.sos])
        if self.state is None:
            prior = np.zeros_like(covs) if self.log_priors is None else np.asarray(self.log_priors, np.float64)
            self.state = [self.prior_weight * prior, float(self.prior_weight)]
        self.state[0] = self.state[0] + logm(covs)
        self.state[1] += 1
        W = expm(-self.state[0] / (2 * self.state[1]))  # inverse square root of the running mean
        return upper(logm(W @ covs @ W)).ravel()

    @staticmethod
    def fit_log_priors(windows, sfreq, bands=BANDS):
        """Mean log-covariance per band over training windows (N, channels, times)."""
        out = []
        for b in bands:
            sos = butter(4, b, btype="bandpass", fs=sfreq, output="sos")
            Xf = sosfiltfilt(sos, np.asarray(windows, np.float64), axis=-1)
            out.append(np.mean([logm(oas_covariance(x)) for x in Xf], axis=0))
        return out
