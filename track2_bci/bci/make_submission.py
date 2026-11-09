"""Train a variant on all public data and build the Codabench ZIP in one command.

    python -m bci.make_submission                       # the best variant: all+transduction, 3 deep seeds
    python -m bci.make_submission --variant "all but deep"
    python -m bci.make_submission --model forest              # the older single models: forest, logreg
    python -m bci.make_submission --synthetic --n-trees 6          # code check, no download

Pick the variant (and its options) by ``python -m bci.benchmark``: the sealed
test holds later sessions of the ten public participants and ten new ones, so
the final model is fitted on every public window (train + val + test). The ZIP
is then checked: contract, imports available on the worker, and the packaged
model must give the same predictions as the trained one.
"""

import argparse
from pathlib import Path

import numpy as np

from bci import data
from bci.ensemble import VARIANTS, assemble, fit_components
from bci.evaluation import predict_streams
from bci.export import check_zip, load_submission, save_submission
from bci.models import forest, logreg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="all+transduction", choices=list(VARIANTS))
    parser.add_argument("--model", choices=["forest", "logreg"], default=None,
                        help="an older single model instead of a variant")
    parser.add_argument("--prior-weight", type=int, default=5)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="tiny fake data, to check the code")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default=None, help="zip name, default: from the variant")
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--n-subject-trees", type=int, default=10)
    parser.add_argument("--n-channels", type=int, default=20)
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--deep-seeds", type=int, default=3)
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--deep-weight", type=float, default=0.3)
    parser.add_argument("--deep-pool", default="log", choices=["log", "lin"])
    parser.add_argument("--deep-crop", type=int, default=None, help="train on random crops of this many samples")
    parser.add_argument("--deep-jitter", action="store_true", help="channel gain jitter and dropout")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior-windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synthetic:
        full = check = data.synthetic()
    else:
        splits = [data.load_split(s, args.data_dir) for s in ("train", "val", "test")]
        full, check = data.concat(*splits), splits[2]
    label = args.model or args.variant
    print(f"training {label!r} on all public data: {len(full)} windows, {len(set(full.subject))} subjects")

    if args.model == "forest":
        model = forest.fit(full, n_trees=args.n_trees, n_channels=args.n_channels, C=args.C,
                           prior_weight=args.prior_weight, seed=args.seed)
    elif args.model == "logreg":
        model = logreg.fit(full, C=args.C, prior_weight=args.prior_weight)
    else:
        switches = VARIANTS[args.variant]
        parts = fit_components(full, [switches], n_trees=args.n_trees, n_subject_trees=args.n_subject_trees,
                               n_channels=args.n_channels, C=args.C, prior_weight=args.prior_weight,
                               deep_seeds=tuple(range(args.deep_seeds)), deep_epochs=args.deep_epochs, deep_crop=args.deep_crop,
                               deep_jitter=args.deep_jitter, seed=args.seed)
        _, model = assemble(parts, switches, full.ch_names, deep_weight=args.deep_weight, alpha=args.alpha,
                            prior_windows=args.prior_windows, deep_pool=args.deep_pool)

    name = args.name or label.replace("+", "_").replace(" ", "_").strip("_")
    folder = Path(args.out_dir) / name
    zip_path = save_submission(model, folder, full.ch_names)

    # the packaged model must reproduce the trained one on real streams
    same = np.mean(predict_streams(model, check) == predict_streams(load_submission(folder, full.ch_names), check))
    print(f"packaged vs trained model, same predictions: {same:.1%}")
    if same < 1.0:
        raise SystemExit("the packaged model differs from the trained one, do not upload")
    check_zip(zip_path)
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
