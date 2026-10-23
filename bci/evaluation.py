"""Evaluation exactly as Track 2 scores it.

- Streamed and causal: each session of each participant is a stream; the model
  sees its windows one at a time (X of shape (1, C, T)), in recording order.
- Every stream starts from a fresh ``copy.deepcopy`` of the model, after
  ``reset_state()`` if the model has one. Adapting inside a stream is allowed.
- Balanced accuracy = mean recall over the classes present in y_true or y_pred.

Two headline numbers:
- ``cell_mean``: per subject x session x interface, averaged -> leaderboard;
- ``session_mean``: per subject x session, averaged -> NeuralBench's
  ``bal_acc_stream_mean``, the number the published baselines report.
"""

import copy

import numpy as np
import pandas as pd
import torch

# Published baselines: PROTEUS public cross-subject test split,
# test/bal_acc_stream_mean (= session_mean), mean over seeds 33-35.
BASELINES = {
    "Constant": 0.333,
    "EEGNet": 0.326,
    "CTNet": 0.350,
    "REVE fine-tuned": 0.360,
}


def balanced_accuracy(y_true, y_pred):
    classes = np.union1d(y_true, y_pred)
    return float(np.mean([np.mean(y_pred[y_true == c] == c) if np.any(y_true == c) else 0.0
                          for c in classes]))


def predict_streams(model, windows):
    """Predictions for every window, made the way the platform makes them."""
    y_pred = np.empty(len(windows), dtype=np.int64)
    for idx in windows.streams():
        stream_model = copy.deepcopy(model)
        if hasattr(stream_model, "reset_state"):
            stream_model.reset_state()
        for i in idx:
            X = torch.from_numpy(windows.X[i : i + 1])
            y_pred[i] = int(np.asarray(stream_model.predict(X)).reshape(-1)[0])
    return y_pred


def score(windows, y_pred):
    """Track 2 metrics for predictions ``y_pred`` on ``windows``."""
    df = pd.DataFrame({"y": windows.y, "pred": y_pred, "stream": windows.stream,
                       "context": windows.context})

    def group_mean(keys):
        return float(np.mean([balanced_accuracy(g["y"].values, g["pred"].values)
                              for _, g in df.groupby(keys)]))

    return {
        "cell_mean": group_mean(["stream", "context"]),
        "session_mean": group_mean("stream"),
        "pooled": balanced_accuracy(windows.y, y_pred),
        "accuracy": float(np.mean(windows.y == y_pred)),
    }


def evaluate(model, windows):
    return score(windows, predict_streams(model, windows))


def compare_with_baselines(results):
    """Table of our models (dict name -> score dict) next to the published ones."""
    rows = [{"model": k, "session_mean": v, "cell_mean": np.nan} for k, v in BASELINES.items()]
    rows += [{"model": k, "session_mean": r["session_mean"], "cell_mean": r["cell_mean"]}
             for k, r in results.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("session_mean").round(3)
