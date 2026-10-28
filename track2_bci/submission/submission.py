"""Riemannian Track 2 submission (exported by bci/export.py).

``model.npz`` holds plain numpy arrays: a list of experts, each with a band, a
channel subset, its session-recentering prior and a linear softmax head on
tangent-space features. The prediction averages the experts' probabilities.
Covers the logreg (one expert) and forest (one expert per tree) models.

Only numpy, scipy and scikit-learn are imported: the Codabench worker does not
list pyriemann, so its functions are reproduced here exactly (same formulas).
The weights are an .npz without pickled objects, readable by any numpy version.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp
from sklearn.covariance import oas

from benchmark_utils.base_solver import CompetSolver


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


def read_spec(path):
    """model.npz -> {"experts": [...], "ch_names": [...], "sfreq": float} (inverse of bci/export.py)."""
    with np.load(path, allow_pickle=False) as f:
        experts = []
        for i in range(int(f["n_experts"])):
            e = {k: f[f"e{i}_{k}"] for k in ("band", "log_prior", "x_mean", "x_scale", "coef", "intercept")}
            e["band"] = tuple(float(b) for b in e["band"])
            e["prior_weight"] = float(f[f"e{i}_prior_weight"])
            channels = f[f"e{i}_channels"]
            e["channels"] = channels if channels.size else None
            experts.append(e)
        return {"experts": experts, "ch_names": [str(c) for c in f["ch_names"]], "sfreq": float(f["sfreq"])}


def covariances_oas(X):
    """(N, C, T) -> (N, C, C), the OAS estimator of pyriemann's Covariances("oas")."""
    return np.stack([oas(x.T)[0] for x in X])


class RiemannModel:
    """``reset_state()`` + ``predict(X)``; X of shape (B, C, T), windows in recording order."""

    def __init__(self, spec, channel_index):
        self.experts = spec["experts"]
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
        covs = {b: covariances_oas(sosfiltfilt(sos, X, axis=-1)) for b, sos in self.sos.items()}
        predictions = []
        for i in range(len(X)):
            log_p = np.stack([self._expert_log_proba(e, s, covs[tuple(e["band"])][i])
                              for e, s in zip(self.experts, self.sessions)])
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
        spec = read_spec(meta["submission_dir"] / "model.npz")
        if abs(float(meta.get("sfreq", spec["sfreq"])) - spec["sfreq"]) > 1e-6:
            raise ValueError(f"model trained at {spec['sfreq']} Hz, data at {meta['sfreq']} Hz")
        names = list(meta.get("ch_names") or spec["ch_names"])
        return RiemannModel(spec, [names.index(n) for n in spec["ch_names"]])
