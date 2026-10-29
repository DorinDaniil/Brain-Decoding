"""Online Bayesian estimate of the time to sleep onset, self-contained (numpy and scipy only).

This file is copied verbatim into the Codabench ``submission.py`` by
``sleep_onset.export``, so it must not import anything from this project.

The recording runs from its start to the first N2 epoch: its length L (in
windows) is one change point, unknown in advance. After window i the remaining
time is r = L - 1 - i windows, so the posterior over L is updated recursively
(Bayesian online change-point detection with a known countdown structure):

    log p(L | x_0..x_i) = log p(L) + sum_j  gamma * log l_j(L - 1 - j),   L > i,

where p(L) is the empirical distribution of recording lengths and
l_j(r) = p(time-to-onset bin of r | features_j) / p(bin) is the likelihood
ratio of a multinomial logistic regression on the window's features (a hybrid
HMM; gamma < 1 discounts the correlation between overlapping feature histories).
The answer is the Bayes decision for the competition's W-bMAE: a weighted
median of the posterior time to onset, weights = bin weight / bin width.
"""

import numpy as np
from scipy.signal import welch
from scipy.special import logsumexp

BANDS = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0), (30.0, 45.0))
HALF_LIVES = (6.0, 24.0, 72.0)  # in windows: 30 s, 2 min, 6 min
BASELINE_WINDOWS = 6            # the first 30 s: the sleeper's awake reference
METRIC_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
METRIC_WEIGHTS = (10.0, 5.0, 3.0, 1.0)
CAP = 600.0


def window_features(x, sfreq, eye_pair=(1, 2)):
    """Spectral features of one window x (C, T) in microvolts.

    Per channel: log and relative power in 7 bands, log total power, log line
    length, log peak amplitude. Plus slow (0.3-2 Hz) power of the frontal
    difference AF7 - AF8, which follows slow eye movements at sleep onset.
    """
    f, p = welch(x, fs=sfreq, nperseg=min(256, x.shape[-1]))
    total = p[:, (f >= 0.5) & (f < 45)].sum(-1) + 1e-12
    feats = []
    for lo, hi in BANDS:
        band = p[:, (f >= lo) & (f < hi)].sum(-1) + 1e-12
        feats += [np.log(band), band / total]
    feats += [np.log(total), np.log(np.abs(np.diff(x, axis=-1)).mean(-1) + 1e-6),
              np.log(np.abs(x).max(-1) + 1e-6)]
    eye = x[eye_pair[0]] - x[eye_pair[1]]
    fe, pe = welch(eye, fs=sfreq, nperseg=min(256, x.shape[-1]))
    slow = pe[(fe >= 0.3) & (fe < 2.0)].sum() + 1e-12
    feats.append(np.array([np.log(slow), slow / (pe[(fe >= 0.3) & (fe < 45)].sum() + 1e-12)]))
    return np.concatenate(feats)


class FeatureHistory:
    """Causal context of the current window: now, three running averages, change since the start."""

    def __init__(self):
        self.n, self.emas, self.baseline = 0, None, None

    def update(self, feats):
        rates = [1 - 0.5 ** (1 / h) for h in HALF_LIVES]
        if self.n == 0:
            self.emas = [feats.copy() for _ in rates]
            self.baseline = feats.copy()
        else:
            self.emas = [e + a * (feats - e) for e, a in zip(self.emas, rates)]
            if self.n < BASELINE_WINDOWS:
                self.baseline = self.baseline + (feats - self.baseline) / (self.n + 1)
        self.n += 1
        return np.concatenate([feats, *self.emas, self.emas[1] - self.baseline])


def metric_weight(tau):
    """Weight per second of true time to onset in W-bMAE: bin weight / bin width."""
    tau = np.minimum(tau, CAP)
    idx = np.clip(np.digitize(tau, METRIC_EDGES[1:-1]), 0, len(METRIC_WEIGHTS) - 1)
    widths = np.diff(METRIC_EDGES)
    return np.asarray(METRIC_WEIGHTS)[idx] / widths[idx]


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


class OnsetModel:
    """Streaming contract: ``reset_state()`` per recording, ``predict(X)`` -> seconds to onset (B,)."""

    def __init__(self, spec, channel_index=None):
        self.spec = spec
        self.channel_index = channel_index
        self.lengths = np.arange(1, len(spec["log_length_prior"]) + 1)  # L = 1 .. Lmax windows
        self.reset_state()

    def reset_state(self):
        self.history = FeatureHistory()
        self.log_post = self.spec["log_length_prior"].astype(np.float64).copy()
        self.i = 0

    def bin_log_likelihood(self, context):
        s = self.spec
        z = s["coef"] @ ((context - s["mean"]) / s["scale"]) + s["intercept"]
        return s["gamma"] * (z - logsumexp(z) - s["log_class_prior"])

    def step(self, x):
        """Update with one window x (C, T); returns the estimated seconds to onset."""
        return self.update(self.history.update(window_features(x, self.spec["sfreq"])))

    def update(self, context):
        """Update with the context features of the next window; returns the estimated seconds to onset."""
        s = self.spec
        tau = s["hop"] * (self.lengths - 1 - self.i) + s["offset"]  # seconds to onset for each L
        alive = self.lengths > self.i
        bins = np.clip(np.digitize(tau, s["bin_edges"][1:-1]), 0, len(s["bin_edges"]) - 2)
        self.log_post = np.where(alive, self.log_post + self.bin_log_likelihood(context)[bins], -np.inf)
        self.i += 1
        if not alive.any():  # longer than every length the prior allows
            return CAP
        if not np.isfinite(self.log_post).any():
            self.log_post = np.where(alive, 0.0, -np.inf)
        post = np.exp(self.log_post - logsumexp(self.log_post))
        tau = np.clip(tau, 0.0, CAP)
        return min(weighted_median(tau[alive], (post * metric_weight(tau))[alive]), CAP)

    def predict(self, X):
        X = np.asarray(X.detach().cpu() if hasattr(X, "detach") else X, np.float64)
        if self.channel_index is not None:
            X = X[:, self.channel_index]
        return np.array([self.step(x) for x in X], dtype=np.float64)
