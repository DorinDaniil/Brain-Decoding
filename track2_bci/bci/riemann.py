"""Riemannian features: covariances + causal session recentering.

For each window we get two vectors:
- ``x``:  the window's covariance in the tangent space at its session's
          reference point (the person is "removed");
- ``fp``: the session's fingerprint, where its reference point lies on the
          manifold relative to the training mean (the person "itself").

The reference point is a running log-Euclidean mean of the session's
covariances, so it only uses windows seen so far, exactly as on the platform.
"""

import numpy as np
import torch
from pyriemann.estimation import Covariances
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler

try:  # pyriemann moved these to pyriemann.geometry
    from pyriemann.geometry.base import expm, invsqrtm, logm
    from pyriemann.geometry.mean import mean_riemann
except ImportError:
    from pyriemann.utils.base import expm, invsqrtm, logm
    from pyriemann.utils.mean import mean_riemann
try:
    from pyriemann.geometry.tangentspace import upper
except ImportError:
    from pyriemann.utils.tangentspace import upper

from bci.data import SFREQ


class SessionRecentering:
    """Running log-Euclidean mean of one session's covariances.

    Starts from the training mean (prior) counted as ``prior_weight`` windows.
    """

    def __init__(self, log_prior, prior_weight):
        self.log_prior = log_prior
        self.log_sum, self.n = prior_weight * log_prior, prior_weight

    def __call__(self, cov):
        """-> (x, fp) for one covariance, updating the session reference."""
        self.log_sum = self.log_sum + logm(cov)
        self.n += 1
        log_ref = self.log_sum / self.n
        W = invsqrtm(expm(log_ref))
        return upper(logm(W @ cov @ W)), upper(log_ref - self.log_prior)


class RiemannFeatures:
    """Fit on training windows, then turn any windows into scaled (x, fp).

    ``x`` is standardized per feature. ``fp`` is divided by one global number,
    so distances between fingerprints stay log-Euclidean distances.

    Parameters
    ----------
    band : (low, high) band-pass in Hz before the covariance.
    prior_weight : how many windows the training mean counts for at the start
        of a session (larger = the reference adapts more slowly).
    """

    def __init__(self, band=(4, 40), prior_weight=5):
        self.band = tuple(band)
        self.sos = butter(4, band, btype="bandpass", fs=SFREQ, output="sos")
        self.prior_weight = prior_weight

    def covariances(self, X):
        """(N, C, T) -> (N, C, C)"""
        X = sosfiltfilt(self.sos, np.asarray(X, np.float64), axis=-1)
        return Covariances("oas").transform(X)

    def fit_transform(self, windows, covs=None):
        """``covs``: precomputed covariances of ``windows`` (else computed here)."""
        covs = self.covariances(windows.X) if covs is None else covs
        self.log_prior = np.mean([logm(c) for c in covs], axis=0)
        self.x_scaler, self.fp_scale = None, 1.0
        x, fp = self._session_features(windows, covs)
        self.x_scaler, self.fp_scale = StandardScaler().fit(x), fp.std()
        return self.x_scaler.transform(x), fp / self.fp_scale

    def transform(self, windows, covs=None):
        covs = self.covariances(windows.X) if covs is None else covs
        return self._session_features(windows, covs)

    def new_session(self):
        return SessionRecentering(self.log_prior, self.prior_weight)

    def step(self, session, cov):
        """Scaled (x, fp) of one window of a session."""
        x, fp = session(cov)
        if self.x_scaler is not None:
            x = self.x_scaler.transform(x[None])[0]
        return x, fp / self.fp_scale

    def _session_features(self, windows, covs):
        x, fp = [None] * len(windows), [None] * len(windows)
        for idx in windows.streams():
            session = self.new_session()
            for i in idx:
                x[i], fp[i] = self.step(session, covs[i])
        return np.array(x), np.array(fp)


class StreamingDecoder:
    """Track 2 contract: ``reset_state()`` + ``predict(X)`` for X of shape (1, C, T).

    ``classify(x, fp) -> class`` is any model working on RiemannFeatures.
    """

    def __init__(self, features, classify):
        self.features, self.classify = features, classify
        self.reset_state()

    def reset_state(self):
        self.session = self.features.new_session()

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        x, fp = self.features.step(self.session, self.features.covariances(X)[0])
        return np.array([self.classify(x, fp)])
