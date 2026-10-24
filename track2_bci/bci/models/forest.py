"""Riemannian random forest: bagging + random subspaces on the SPD manifold.

Each "tree" is a small Riemannian model (session recentering + tangent space +
LogReg) that sees only
- a random frequency band from the bank,
- a random subset of channels: the covariance sub-matrix of a channel subset
  is again SPD, so the whole Riemannian pipeline applies unchanged,
- a bootstrap of the training *subjects* (sample weights = how many times a
  subject was drawn), which trains robustness to a change of person.

The forest averages the trees' probabilities: p(y | X) = mean_k p_k(y | X).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from bci.riemann import RiemannFeatures

# broadband + theta, alpha/mu, low beta, high beta, low gamma
BANDS = ((4, 40), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40))


@dataclass
class Tree:
    band: int                  # index into the band bank
    channels: np.ndarray       # channel subset
    features: RiemannFeatures  # recentering state of this tree
    clf: LogisticRegression


class RiemannForest:
    """Track 2 contract: ``reset_state()`` + ``predict(X)`` for X of shape (1, C, T)."""

    def __init__(self, bank, trees):
        self.bank, self.trees = bank, trees
        self.reset_state()

    def reset_state(self):
        self.sessions = [t.features.new_session() for t in self.trees]

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        covs = [f.covariances(X)[0] for f in self.bank]  # one full covariance per band
        probas = []
        for tree, session in zip(self.trees, self.sessions):
            x, _ = tree.features.step(session, _sub(covs[tree.band], tree.channels))
            probas.append(tree.clf.predict_proba(x[None])[0])
        return np.array([np.mean(probas, axis=0).argmax()])

    def tree_probas(self, windows):
        """(n_trees, N, 3) probabilities of every tree on ``windows`` (same as streaming)."""
        covs = [f.covariances(windows.X) for f in self.bank]
        probas = []
        for tree in self.trees:
            x, _ = tree.features.transform(windows, _sub(covs[tree.band], tree.channels))
            probas.append(tree.clf.predict_proba(x))
        return np.stack(probas)


def _sub(cov, channels):
    """Covariance sub-matrix of a channel subset (works for (C, C) and (N, C, C))."""
    return cov[..., channels, :][..., channels]


def fit(train, n_trees=50, n_channels=20, C=0.01, bands=BANDS, prior_weight=5, seed=0):
    rng = np.random.default_rng(seed)
    bank = [RiemannFeatures(band, prior_weight) for band in bands]
    covs = [f.covariances(train.X) for f in bank]  # full covariances, computed once per band
    subjects = sorted(set(train.subject))

    trees = []
    for _ in range(n_trees):
        band = int(rng.integers(len(bands)))
        channels = np.sort(rng.choice(train.X.shape[1], n_channels, replace=False))
        drawn = Counter(rng.choice(subjects, len(subjects)))  # bootstrap of subjects
        weight = np.array([drawn[s] for s in train.subject], float)

        features = RiemannFeatures(bands[band], prior_weight)
        x, _ = features.fit_transform(train, _sub(covs[band], channels))
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(x[weight > 0], train.y[weight > 0], sample_weight=weight[weight > 0])
        trees.append(Tree(band, channels, features, clf))
    return RiemannForest(bank, trees)
