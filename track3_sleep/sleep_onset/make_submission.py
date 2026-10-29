"""Fit the Bayesian onset model and build the Codabench ZIP in one command.

    python -m sleep_onset.make_submission
    python -m sleep_onset.make_submission --synthetic      # code check, no download

The classifier is fitted on ``train`` and the likelihood temperature gamma is
chosen on ``val``; ``test`` (the warm-up recordings) is scored for reference.
The final model is then refitted on all public recordings with that gamma: the
sealed phase scores new nights of the same sleepers too.
"""

import argparse
from pathlib import Path

import numpy as np

from sleep_onset import data
from sleep_onset.export import check_zip, load_submission, save_submission
from sleep_onset.train import contexts, fit, predict_recording, score, wbmae


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="fake recordings, to check the code")
    parser.add_argument("--C", type=float, default=0.05)
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default="bayes_onset")
    args = parser.parse_args()

    if args.synthetic:
        splits, ch_names = data.synthetic()
    else:
        loaded = {s: data.load_split(s, args.data_dir) for s in ("train", "val", "test")}
        splits, ch_names = {s: v[0] for s, v in loaded.items()}, loaded["train"][1]
    print({s: f"{len(r)} recordings" for s, r in splits.items()}, ch_names)

    spec, _ = fit(splits["train"], splits["val"], C=args.C)
    test_ctx = [contexts(r) for r in splits["test"]]
    print(f"test W-bMAE: {score(spec, splits['test'], test_ctx)[0]:.1f} s "
          f"(elapsed-time prior only: {score(dict(spec, gamma=0.0), splits['test'], test_ctx)[0]:.1f} s, "
          f"constant 300 s: {np.mean([wbmae(r.y, np.full(len(r.y), 300.0)) for r in splits['test']]):.1f} s)")

    gamma = spec["gamma"]
    everything = splits["train"] + splits["val"] + splits["test"]
    spec, _ = fit(everything, splits["val"], C=args.C, gammas=(gamma,), log=lambda *_: None)
    print(f"final model: refitted on {len(everything)} recordings, gamma = {gamma}")

    folder = Path(args.out_dir) / args.name
    zip_path = save_submission(spec, ch_names, folder)
    packaged = load_submission(folder, ch_names)
    rec = splits["test"][0]
    packaged.reset_state()
    a = packaged.predict(rec.X)
    b = predict_recording(spec, contexts(rec))
    gap = float(np.abs(a - b).max())
    print(f"packaged vs fitted model, max difference: {gap:.2e} s")
    if gap > 1e-6:
        raise SystemExit("the packaged model differs from the fitted one, do not upload")
    check_zip(zip_path)
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
