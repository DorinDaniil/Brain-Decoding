"""Fit the Bayesian onset model and score it as the leaderboard does.

Score: W-bMAE within each recording (true-target ranges [0, 40), [40, 90),
[90, 300), [300, 600] s, weights 10, 5, 3, 1, non-empty ranges only), averaged
over recordings.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from sleep_onset.data import HOP, SFREQ
from sleep_onset.model import CAP, METRIC_EDGES, METRIC_WEIGHTS, FeatureHistory, OnsetModel, window_features

# time-to-onset bins of the classifier, in seconds (finer near onset)
BIN_EDGES = (0.0, 20.0, 40.0, 65.0, 90.0, 150.0, 220.0, 300.0, 450.0, 600.0, 900.0, 1500.0, np.inf)


def wbmae(y_true, y_pred):
    idx = np.clip(np.digitize(y_true, METRIC_EDGES[1:-1]), 0, len(METRIC_WEIGHTS) - 1)
    bins = [b for b in range(len(METRIC_WEIGHTS)) if np.any(idx == b)]
    maes = np.array([np.abs(y_pred - y_true)[idx == b].mean() for b in bins])
    w = np.asarray(METRIC_WEIGHTS)[bins]
    return float((w * maes).sum() / w.sum())


def contexts(recording, eye_pair=(1, 2)):
    """(n, D) causal context features of every window, computed as at inference."""
    history = FeatureHistory()
    return np.stack([history.update(window_features(x.astype(np.float64), SFREQ, eye_pair)) for x in recording.X])


def predict_recording(spec, ctx):
    model = OnsetModel(spec)
    return np.array([model.update(c) for c in ctx])


def score(spec, recordings, ctxs):
    """Leaderboard score of ``spec`` and its per-recording values."""
    per = np.array([wbmae(r.y, predict_recording(spec, c)) for r, c in zip(recordings, ctxs)])
    return float(per.mean()), per


def length_prior(lengths, l_max, bandwidth=0.25):
    """log p(L) for L = 1..l_max: a kernel density of the training lengths in log scale, with a floor."""
    grid = np.log(np.arange(1, l_max + 1))
    dens = np.exp(-0.5 * ((grid[:, None] - np.log(lengths)[None]) / bandwidth) ** 2).mean(1) / np.arange(1, l_max + 1)
    dens = dens / dens.sum() + 1e-6 / l_max
    return np.log(dens / dens.sum())


def fit(train, val, C=0.05, gammas=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), log=print):
    """Spec of a fitted OnsetModel. ``gamma`` is chosen on ``val`` by the leaderboard score."""
    train_ctx = [contexts(r) for r in train]
    X = np.concatenate(train_ctx)
    tau = np.concatenate([r.tau for r in train])
    labels = np.clip(np.digitize(tau, BIN_EDGES[1:-1]), 0, len(BIN_EDGES) - 2)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000).fit((X - mean) / scale, labels)
    n_bins = len(BIN_EDGES) - 1
    coef = np.zeros((n_bins, X.shape[1]))
    intercept = np.full(n_bins, -30.0)  # bins absent from training get ~zero probability
    coef[clf.classes_], intercept[clf.classes_] = clf.coef_, clf.intercept_
    class_prior = np.bincount(labels, minlength=n_bins) / len(labels)

    lengths = np.array([len(r.y) for r in train])
    spec = {
        "coef": coef, "intercept": intercept, "mean": mean, "scale": scale,
        "log_class_prior": np.log(class_prior + 1e-12), "bin_edges": np.asarray(BIN_EDGES),
        "log_length_prior": length_prior(lengths, int(lengths.max() * 3)),
        "gamma": 1.0, "hop": HOP, "offset": float(np.mean([r.y[-1] for r in train])), "sfreq": SFREQ,
    }
    val_ctx = [contexts(r) for r in val]
    results = {}
    for g in gammas:
        spec["gamma"] = g
        results[g] = score(spec, val, val_ctx)[0]
        log(f"gamma {g}: val W-bMAE {results[g]:.1f} s")
    spec["gamma"] = min(results, key=results.get)
    log(f"gamma = {spec['gamma']} (gamma 0 = elapsed-time prior only, no EEG)")
    return spec, results
