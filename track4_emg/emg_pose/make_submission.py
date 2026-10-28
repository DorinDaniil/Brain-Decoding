"""Train an ensemble of PoseNets and build the Codabench ZIP in one command.

    python -m emg_pose.make_submission --seeds 3 --epochs 40
    python -m emg_pose.make_submission --synthetic --seeds 2 --epochs 3   # code check, no download

Each seed trains on ``train`` and keeps its best epoch on ``val`` (unseen
users); the ensemble averages the seeds and, at test time, the electrode ring
rolled by -1, 0 and +1. ``test`` is the warm-up evaluation split (unseen users
x unseen stages); its labels are public, so its score is indicative only.
"""

import argparse
import json
from pathlib import Path

import torch

from emg_pose import data
from emg_pose.export import check_zip, load_submission, save_submission
from emg_pose.nets import PoseEnsemble
from emg_pose.train import evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="small fake data, to check the code")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default="posenet")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic:
        loaders, meta = data.synthetic()
    else:
        loaders, meta = data.loaders(args.data_dir, batch_size=args.batch_size)
    print({name: len(l.dataset) for name, l in loaders.items()}, meta)

    folder = Path(args.out_dir) / args.name
    folder.parent.mkdir(parents=True, exist_ok=True)
    nets, report = [], {}
    for seed in range(args.seeds):
        net, history = train(loaders, meta, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
        nets.append(net)
        report[f"seed {seed}"] = evaluate(PoseEnsemble([net]), loaders["test"])
        print(f"seed {seed}, test:", report[f"seed {seed}"])

    ensemble = PoseEnsemble(nets, tta_offsets=(-1, 0, 1)).to(device)
    report["ensemble"] = evaluate(ensemble, loaders["test"])
    print("ensemble + rotation TTA, test:", report["ensemble"])

    zip_path = save_submission(ensemble, folder)
    (folder.parent / f"{args.name}_scores.json").write_text(json.dumps(report, indent=1))
    X = next(iter(loaders["test"]))[0][:4]
    gap = (load_submission(folder, device).predict(X).cpu() - ensemble.predict(X).cpu()).abs().max()
    print(f"packaged vs trained model, max difference: {gap:.2e}")
    if gap > 1e-4:
        raise SystemExit("the packaged model differs from the trained one, do not upload")
    check_zip(zip_path, n_times=meta["n_times"])
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
