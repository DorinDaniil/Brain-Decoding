"""Retrieval metrics exactly as Track 1 computes them, in bounded memory.

Each prediction is ranked by cosine similarity against the candidate gallery,
the unique target embeddings of the evaluated split. Tied scores share their
average rank. Two variants:
- ``top5``: every window is a query (warm-up);
- ``top5_subject_agg``: predictions of repeated presentations of one image are
  first averaged within each subject (sealed evaluation).

Scores are computed in chunks of queries, so memory does not grow with
windows x gallery (hundreds of thousands x thousands on Alljoined-1.6M).
"""

import numpy as np
import torch

# Reference development results on THINGS-EEG2 (top-5 %, competition repo README).
REFERENCE = {"chance": 2.22, "EEGNet": 28.13, "REVE frozen probe": 84.75}


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def image_ids(target):
    """Image index of each row and the row of each image's first occurrence.

    Equal embeddings give equal ids. Rows are told apart by an exact integer
    hash of their bits (wrap-around arithmetic) instead of sorting all 1536
    columns, which would copy the array.
    """
    bits = np.ascontiguousarray(target, np.float32).view(np.uint32)
    weights = np.random.default_rng(0).integers(1, 2**63, size=bits.shape[1], dtype=np.uint64) | np.uint64(1)
    key = np.concatenate([(bits[i:i + 4096].astype(np.uint64) * weights).sum(1)
                          for i in range(0, len(bits), 4096)])
    _, first, idx = np.unique(key, return_index=True, return_inverse=True)
    return idx.reshape(-1), first


def _hits(queries, gallery, true_idx, ks=(1, 5), chunk=2048):
    """Number of queries whose true candidate has an average rank below each k."""
    hits = np.zeros(len(ks))
    for start in range(0, len(queries), chunk):
        scores = _normalize(queries[start:start + chunk].astype(np.float64)) @ gallery
        true = scores[np.arange(len(scores)), true_idx[start:start + chunk]][:, None]
        rank = ((scores > true).sum(1) + (scores >= true).sum(1) - 1) / 2
        rank[np.isnan(true[:, 0])] = np.inf  # a NaN prediction is a miss
        hits += [(rank < k).sum() for k in ks]
    return hits


def retrieval(pred, target, subject):
    """Top-1 / top-5 (in %) of predictions (N, D) for targets (N, D) of subjects (N,)."""
    idx, first = image_ids(target)
    gallery = _normalize(np.asarray(target[first], np.float64)).T
    top1, top5 = 100 * _hits(pred, gallery, idx) / len(pred)

    # average the predictions of each (subject, image)
    subject = np.unique(np.asarray(subject), return_inverse=True)[1].reshape(-1)
    keys, inverse = np.unique(subject * (idx.max() + 1) + idx, return_inverse=True)
    pred_agg = np.zeros((len(keys), pred.shape[1]))
    np.add.at(pred_agg, inverse.reshape(-1), pred)
    agg1, agg5 = 100 * _hits(pred_agg, gallery, keys % (idx.max() + 1)) / len(keys)

    return {"top1": float(top1), "top5": float(top5), "top1_subject_agg": float(agg1),
            "top5_subject_agg": float(agg5), "n_candidates": len(first)}


@torch.no_grad()
def predict(model, loader, known_subject=False):
    """``model.predict(X) -> (B, D)`` over a loader -> (pred, target, subject) as numpy float32.

    ``known_subject``: also pass the subject ids (models with subject channels);
    the platform passes only X.
    """
    pred, target, subject = [], [], []
    for X, y, s in loader:
        out = model.predict(X, s) if known_subject else model.predict(X)
        pred.append(torch.as_tensor(out).detach().float().cpu().numpy())
        target.append(y.numpy())
        subject.append(np.asarray(s))
    return np.concatenate(pred), np.concatenate(target), np.concatenate(subject)


def evaluate(model, loader, known_subject=False):
    return retrieval(*predict(model, loader, known_subject))
