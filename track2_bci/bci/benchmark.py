"""Cross-validate every variant of the streaming model on the public data, in one run.

    python -m bci.benchmark                          # every variant of bci.ensemble.VARIANTS
    python -m bci.benchmark --variants forest +balance --jobs 8
    python -m bci.benchmark --synthetic --n-trees 6  # code check, no download

Two protocols, the two halves of the sealed test:
- new people: each participant in turn is left out of training;
- later sessions: the last session of every participant with several is the
  test, their earlier sessions are in training (only 2 such participants
  on the public data, so this half is noisy).

Each variant is the very model a submission would contain, scored stream by
stream as on the platform. Prints the leaderboard's ``cell_mean`` per protocol
and their mean, plus how the router behaved, and writes every fold to a CSV.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from bci import data
from bci.ensemble import VARIANTS, assemble, fit_components
from bci.evaluation import later_sessions, leave_one_subject_out, score


def run_streams(model, windows, subjects):
    """Predictions as on the platform, and per stream: router confidence and whether it named the right person."""
    y_pred = np.empty(len(windows), dtype=np.int64)
    routes = []
    for idx in windows.streams():
        stream = copy.deepcopy(model)
        stream.reset_state()
        for i in idx:
            y_pred[i] = int(stream.predict(torch.from_numpy(windows.X[i:i + 1]))[0])
        named = subjects[stream.subject] if stream.subject >= 0 else None
        routes.append({"known": stream.known, "named_right": named == windows.subject[idx[0]]})
    return y_pred, routes


def run_fold(protocol, name, train, test, variants, args):
    log = (lambda *a: None) if args.jobs != 1 else print
    parts = fit_components(train, [VARIANTS[v] for v in variants], n_trees=args.n_trees,
                           n_subject_trees=args.n_subject_trees, n_channels=args.n_channels, C=args.C,
                           deep_seeds=tuple(range(args.deep_seeds)), deep_epochs=args.deep_epochs, log=log)
    subjects = sorted(set(train.subject))
    rows = []
    for variant in variants:
        _, model = assemble(parts, VARIANTS[variant], train.ch_names, deep_weight=args.deep_weight,
                            alpha=args.alpha, prior_windows=args.prior_windows, deep_pool=args.deep_pool)
        y_pred, routes = run_streams(model, test, subjects)
        row = {"protocol": protocol, "fold": name, "variant": variant, **score(test, y_pred)}
        if VARIANTS[variant].get("routing"):
            row["router known"] = np.mean([r["known"] for r in routes])
            row["router named right"] = np.mean([r["named_right"] for r in routes])
        rows.append(row)
        log(f"{protocol} / {name} / {variant}: cell_mean {row['cell_mean']:.3f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="tiny fake data, to check the code")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--n-subject-trees", type=int, default=10)
    parser.add_argument("--n-channels", type=int, default=20)
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--deep-seeds", type=int, default=1)
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--deep-weight", type=float, default=0.3)
    parser.add_argument("--deep-pool", default="log", choices=["log", "lin"])
    parser.add_argument("--alpha", type=float, default=1.0, help="strength of the balance correction")
    parser.add_argument("--prior-windows", type=int, default=10, help="uniform windows the balance starts from")
    parser.add_argument("--jobs", type=int, default=1, help="folds in parallel")
    parser.add_argument("--out", default="benchmark.csv")
    args = parser.parse_args()

    if args.synthetic:
        windows = data.synthetic()
    else:
        windows = data.concat(*[data.load_split(s, args.data_dir) for s in ("train", "val", "test")])
    print(f"{len(windows)} windows, {len(set(windows.subject))} participants, {len(set(windows.stream))} sessions")

    folds = [("new people", *f) for f in leave_one_subject_out(windows)]
    folds += [("later sessions", *f) for f in later_sessions(windows)]
    results = Parallel(n_jobs=args.jobs)(delayed(run_fold)(*fold, args.variants, args) for fold in folds)
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(args.out, index=False)

    summary = table.pivot_table(index="variant", columns="protocol", values="cell_mean", aggfunc="mean")
    summary = summary.reindex(args.variants)
    summary["sealed-like mean"] = summary.mean(axis=1)
    print("\ncell_mean (the leaderboard metric), mean over folds:")
    print(summary.round(3).to_string())
    if "router known" in table:
        routed = table.dropna(subset=["router known"])
        print("\nrouter, per protocol (new people: known should be near 0; later sessions: near 1, named right):")
        print(routed.groupby(["variant", "protocol"])[["router known", "router named right"]].mean().round(2).to_string())
    print(f"\nevery fold: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
