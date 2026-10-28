"""Retrieval metrics exactly as Track 1 computes them.

Each prediction is ranked by cosine similarity against the candidate gallery,
the unique target embeddings of the evaluated split. Tied scores share their
average rank. Two variants:
- ``top5``: every window is a query (warm-up);
- ``top5_subject_agg``: predictions of repeated presentations of one image are
  first averaged within each subject (sealed evaluation).
"""

import numpy as np
import torch

# Reference development results on THINGS-EEG2 (top-5 %, competition repo README).
REFERENCE = {"chance": 2.22, "EEGNet": 28.13, "REVE frozen probe": 84.75}


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _topk(scores, true_idx, k):
    true = scores[np.arange(len(scores)), true_idx][:, None]
    rank = ((scores > true).sum(1) + (scores >= true).sum(1) - 1) / 2
    rank[np.isnan(true[:, 0])] = np.inf  # a NaN prediction is a miss
    return float(np.mean(rank < k))


def retrieval(pred, target, subject):
    """Top-1 / top-5 (in %) of predictions (N, D) for targets (N, D) of subjects (N,)."""
    candidates, idx = np.unique(target, axis=0, return_inverse=True)
    idx = idx.reshape(-1)
    gallery = _normalize(candidates).T
    scores = _normalize(pred) @ gallery

    keys, inverse = np.unique(np.stack([subject, idx], axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    pred_agg = np.zeros((len(keys), pred.shape[1]))
    np.add.at(pred_agg, inverse, pred)
    scores_agg = _normalize(pred_agg) @ gallery

    return {
        "top1": 100 * _topk(scores, idx, 1),
        "top5": 100 * _topk(scores, idx, 5),
        "top1_subject_agg": 100 * _topk(scores_agg, keys[:, 1], 1),
        "top5_subject_agg": 100 * _topk(scores_agg, keys[:, 1], 5),
        "n_candidates": len(candidates),
    }


@torch.no_grad()
def predict(model, loader, known_subject=False):
    """``model.predict(X) -> (B, D)`` over a loader -> (pred, target, subject) as numpy.

    ``known_subject``: also pass the subject ids (models with subject channels);
    the platform passes only X.
    """
    pred, target, subject = [], [], []
    for X, y, s in loader:
        out = model.predict(X, s) if known_subject else model.predict(X)
        pred.append(torch.as_tensor(out).detach().cpu().numpy().astype(np.float64))
        target.append(y.numpy())
        subject.append(np.asarray(s))
    return np.concatenate(pred), np.concatenate(target), np.concatenate(subject)


def evaluate(model, loader, known_subject=False):
    return retrieval(*predict(model, loader, known_subject))
