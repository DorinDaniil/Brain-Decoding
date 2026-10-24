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


def leave_one_subject_out(windows):
    """Folds (name, train, test): each person in turn is the new, unseen subject."""
    for s in sorted(set(windows.subject)):
        yield s, windows.select(windows.subject != s), windows.select(windows.subject == s)


def later_sessions(windows, n_test=1):
    """One fold: the last ``n_test`` sessions of every person are the test,
    their earlier sessions are in train (the known-subject half of the sealed test)."""
    is_test = np.zeros(len(windows), bool)
    for s in set(windows.subject):
        sessions = sorted(set(windows.session[windows.subject == s]), key=lambda v: (len(v), v))
        if len(sessions) > n_test:
            is_test |= (windows.subject == s) & np.isin(windows.session, sessions[-n_test:])
    yield "later sessions", windows.select(~is_test), windows.select(is_test)


def cross_validate(fit, folds):
    """``fit(train) -> model``; one row of Track 2 scores per fold."""
    rows = [{"fold": name, **evaluate(fit(train), test)} for name, train, test in folds]
    return pd.DataFrame(rows).set_index("fold")


def cross_validation_summary(fit, windows):
    """session_mean of ``fit`` on both protocols, run on ``windows`` (train + val)."""
    loso = cross_validate(fit, leave_one_subject_out(windows))["session_mean"]
    later = cross_validate(fit, later_sessions(windows))["session_mean"]
    return pd.DataFrame({
        "new subjects (LOSO)": [loso.mean(), loso.std()],
        "later sessions": [later.mean(), np.nan],
    }, index=["session_mean", "std over subjects"]).round(3)


def compare_with_baselines(results):
    """Table of our models (dict name -> score dict) next to the published ones."""
    rows = [{"model": k, "session_mean": v, "cell_mean": np.nan} for k, v in BASELINES.items()]
    rows += [{"model": k, "session_mean": r["session_mean"], "cell_mean": r["cell_mean"]}
             for k, r in results.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("session_mean").round(3)
