"""Train the final model on all public data and build the Codabench ZIP in one command.

    python -m bci.make_submission                       # forest, defaults below
    python -m bci.make_submission --model logreg
    python -m bci.make_submission --n-trees 100 --n-channels 20 --name forest_k100

The sealed test holds later sessions of the ten public participants and ten new
ones, so the final model is fitted on every public window (train + val + test).
The ZIP is then checked: contract, imports available on the worker, and the
packaged model must give the same predictions as the trained one.

Warm-up scores of such a model are meaningless: the warm-up test participants
are part of its training data. Use the cross-validation of notebooks/forest.ipynb
to compare models, and the warm-up only to check that the upload works.
"""

import argparse
from pathlib import Path

import numpy as np

from bci.data import concat, load_split
from bci.evaluation import predict_streams
from bci.export import check_zip, load_submission, save_submission
from bci.models import forest, logreg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["forest", "logreg"], default="forest")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default=None, help="zip name, default: the model name")
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--n-channels", type=int, default=20)
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = [load_split(s, args.data_dir) for s in ("train", "val", "test")]
    full = concat(*splits)
    print(f"training on all public data: {len(full)} windows, {len(set(full.subject))} subjects")

    if args.model == "forest":
        model = forest.fit(full, n_trees=args.n_trees, n_channels=args.n_channels, C=args.C,
                           prior_weight=args.prior_weight, seed=args.seed)
    else:
        model = logreg.fit(full, C=args.C, prior_weight=args.prior_weight)

    folder = Path(args.out_dir) / (args.name or args.model)
    zip_path = save_submission(model, folder, full.ch_names)

    # the packaged model must reproduce the trained one on a few real streams
    check = splits[2]
    same = np.mean(predict_streams(model, check) == predict_streams(load_submission(folder, full.ch_names), check))
    print(f"packaged vs trained model, same predictions: {same:.1%}")
    if same < 1.0:
        raise SystemExit("the packaged model differs from the trained one, do not upload")

    check_zip(zip_path)
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
