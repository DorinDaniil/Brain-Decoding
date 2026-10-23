"""Riemannian Track 2 submission (exported by bci/export.py).

``model.pkl`` holds plain numpy arrays: a list of experts, each with a band, a
channel subset, its session-recentering prior and a linear softmax head on
tangent-space features. Experts are combined by a weighted geometric mean
(``combine="geometric"``) or by averaging probabilities (``combine="mean"``).
Covers the logreg, fbstack and forest models.
"""

import pickle

import numpy as np
from pyriemann.estimation import Covariances
from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp

from benchmark_utils.base_solver import CompetSolver

try:  # pyriemann moved these to pyriemann.geometry
    from pyriemann.geometry.base import expm, invsqrtm, logm
except ImportError:
    from pyriemann.utils.base import expm, invsqrtm, logm
try:
    from pyriemann.geometry.tangentspace import upper
except ImportError:
    from pyriemann.utils.tangentspace import upper


class RiemannModel:
    """``reset_state()`` + ``predict(X)``; X of shape (B, C, T), windows in recording order."""

    def __init__(self, spec, channel_index):
        self.experts, self.combine, self.weights = spec["experts"], spec["combine"], spec["weights"]
        self.channel_index = channel_index  # input channels -> training order
        bands = {tuple(e["band"]) for e in self.experts}
        self.sos = {b: butter(4, b, btype="bandpass", fs=spec["sfreq"], output="sos") for b in bands}
        self.reset_state()

    def reset_state(self):
        """New session: every expert's reference starts again from its training prior."""
        self.sessions = [{"log_sum": e["prior_weight"] * e["log_prior"], "n": e["prior_weight"]}
                         for e in self.experts]

    def predict(self, X):
        X = np.asarray(X.detach().cpu(), np.float64)[:, self.channel_index]
        covs = {b: Covariances("oas").transform(sosfiltfilt(sos, X, axis=-1)) for b, sos in self.sos.items()}
        predictions = []
        for i in range(len(X)):
            log_p = np.stack([self._expert_log_proba(e, s, covs[tuple(e["band"])][i])
                              for e, s in zip(self.experts, self.sessions)])
            if self.combine == "geometric":
                predictions.append(np.argmax(self.weights @ log_p))
            else:
                predictions.append(np.argmax(np.exp(log_p).mean(axis=0)))
        return np.array(predictions, dtype=np.int64)

    @staticmethod
    def _expert_log_proba(expert, session, cov):
        if expert["channels"] is not None:
            cov = cov[np.ix_(expert["channels"], expert["channels"])]
        session["log_sum"] = session["log_sum"] + logm(cov)
        session["n"] += 1
        W = invsqrtm(expm(session["log_sum"] / session["n"]))
        x = (upper(logm(W @ cov @ W)) - expert["x_mean"]) / expert["x_scale"]
        z = expert["coef"] @ x + expert["intercept"]
        return z - logsumexp(z)


class Solver(CompetSolver):

    name = "RiemannTrack2"

    def load_model(self, meta):
        with open(meta["submission_dir"] / "model.pkl", "rb") as f:
            spec = pickle.load(f)
        if abs(float(meta.get("sfreq", spec["sfreq"])) - spec["sfreq"]) > 1e-6:
            raise ValueError(f"model trained at {spec['sfreq']} Hz, data at {meta['sfreq']} Hz")
        names = list(meta.get("ch_names") or spec["ch_names"])
        return RiemannModel(spec, [names.index(n) for n in spec["ch_names"]])
