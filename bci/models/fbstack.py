"""Filter-bank stacking ensemble.

One Riemannian expert per frequency band (session recentering + tangent space +
LogReg). They are combined as a weighted geometric mean with non-negative
band weights:

    p(y | X) ∝ exp( sum_b w_b log p_b(y | x_b) ),   w_b >= 0, sum_b w_b = 1

The weights are fitted on out-of-subject predictions: each training subject is
predicted by experts fitted without it, as a new person would be. ``w_b >= 0``
keeps the combination from flipping an expert, so it stays sane on known
subjects too, where the experts are more accurate than out of subject.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from bci.riemann import FilterBankFeatures, StreamingDecoder

# broadband + theta, alpha/mu, low beta, high beta, low gamma
BANDS = ((4, 40), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40))


def band_weights(log_p, y):
    """Band weights on the simplex, w = softmax(v), fitted by class-balanced log loss.

    A free temperature t takes the overall scale (z = t * sum_b w_b log_p_b), so
    uninformative experts leave w near uniform instead of shrinking it to 0.
    """
    sample_weight = (len(y) / (3 * np.bincount(y, minlength=3)))[y]
    B = log_p.shape[1]

    def loss(params):
        w, t = np.exp(params[:B] - logsumexp(params[:B])), np.exp(params[B])
        z = t * np.einsum("nbk,b->nk", log_p, w)
        log_softmax = z - logsumexp(z, axis=1, keepdims=True)
        return -np.mean(sample_weight * log_softmax[np.arange(len(y)), y])

    v = minimize(loss, np.zeros(B + 1)).x[:B]
    return np.exp(v - logsumexp(v))


def fit(train, bands=BANDS, C=0.01, shrink=0.5, prior_weight=5, seed=0):
    """``shrink`` pulls the fitted band weights towards uniform: 0 = as fitted, 1 = plain average.
    Fitted weights come from few subjects and can be noisy."""
    features = FilterBankFeatures(bands, prior_weight)
    x, _ = features.fit_transform(train)  # (N, B, 861)
    y, B = train.y, len(bands)

    def expert(x_band, y_band):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
        return clf.fit(x_band, y_band)

    # out-of-subject probabilities of every band expert: (N, B, 3)
    oof = np.zeros((len(y), B, 3))
    for s in set(train.subject):
        out = train.subject == s
        for b in range(B):
            oof[out, b] = expert(x[~out, b], y[~out]).predict_proba(x[out, b])

    weights = shrink / B + (1 - shrink) * band_weights(np.log(oof + 1e-6), y)
    experts = [expert(x[:, b], y) for b in range(B)]

    def classify(x, fp):
        log_p = np.log([e.predict_proba(x[b][None])[0] for b, e in enumerate(experts)])
        return int(np.argmax(weights @ log_p))

    model = StreamingDecoder(features, classify)
    model.bands, model.experts, model.weights = bands, experts, weights
    return model
