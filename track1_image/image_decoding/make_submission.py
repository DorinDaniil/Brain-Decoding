"""Train the ensemble (and the optional prior) and build the Codabench ZIP in one command.

    python -m image_decoding.make_submission --study xu2025alljoined
    python -m image_decoding.make_submission --synthetic          # code check, no download

A few subjects are held out of training entirely and scored on the study's
test images, subject-aggregated: a rehearsal of the sealed phase (new people,
new images). Every encoder keeps its best epoch by that score; ``--seeds``
encoders form the ensemble. The flow prior then learns on half of the
held-out subjects (training images) and ships only if some number of steps
beats none on the other half (test images). The ZIP is checked against the contract, and the packaged model must
predict like the trained one.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from image_decoding import data
from image_decoding.encoder import held_out_subjects, split, target_space, train_encoder
from image_decoding.evaluation import evaluate, predict
from image_decoding.export import check_zip, load_submission, save_submission
from image_decoding.nets import ImageDecoder
from image_decoding.prior import fit_prior, score_steps


def rounded(scores):
    return {k: round(v, 2) for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="xu2025alljoined", choices=sorted(data.STUDIES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="tiny fake data, to check the code")
    parser.add_argument("--epochs", type=int, default=30, help="encoder epochs")
    parser.add_argument("--seeds", type=int, default=3, help="encoders in the ensemble")
    parser.add_argument("--held-out", type=int, default=2, help="subjects kept out of training for validation")
    parser.add_argument("--soft", type=float, default=0.0, help="soft-target mass on similar images")
    parser.add_argument("--mse-weight", type=float, default=0.0, help="weight of the MSE term")
    parser.add_argument("--prior-epochs", type=int, default=100, help="0: no flow prior")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default="spatial_conv")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    packed = (data.synthetic() if args.synthetic else data.load(args.study, args.data_dir)).to(device)
    print(packed.summary())
    held = held_out_subjects(packed, args.held_out)
    _, val_idx = split(packed, held)

    small = dict(width=64, hidden=256, n_virtual=32, images_per_batch=64) if args.synthetic else {}  # quick on CPU
    encoders = []
    for seed in range(args.seeds):
        net, _ = train_encoder(packed, held, epochs=args.epochs, device=device, seed=seed,
                               soft=args.soft, mse_weight=args.mse_weight, **small)
        encoders.append(net)
    mu, s, radius = target_space(packed)
    model = ImageDecoder(encoders, mu, s, radius, packed.positions, packed.ch_names).to(device)
    for m, enc in enumerate(encoders):  # each member's typical norm, so none dominates the mean
        single = ImageDecoder([enc], mu, s, radius, packed.positions, packed.ch_names).to(device)
        model.scales[m] = float(np.linalg.norm(predict(single, packed, val_idx), axis=1).mean())
    print("ensemble, held-out subjects on test images:", rounded(evaluate(model, packed, val_idx)))

    if args.prior_epochs > 0:
        # the prior learns on half of the held-out subjects and is judged on the other half: new people both
        learn, judge = held[: len(held) // 2], held[len(held) // 2:]
        fit_idx = np.flatnonzero(np.isin(packed.subject, learn) & ~packed.test)
        judge_idx = np.flatnonzero(np.isin(packed.subject, judge) & packed.test)
        monitor = (predict(model, packed, judge_idx), packed.image[judge_idx], packed.subject[judge_idx], packed.images)
        fit_prior(model, predict(model, packed, fit_idx), packed.images[packed.image[fit_idx]],
                  epochs=args.prior_epochs, monitor=monitor)
        steps = {n: score_steps(model, *monitor, n_steps=n)["top5_subject_agg"] for n in (0, 1, 2, 5, 10, 20)}
        model.n_steps = max(steps, key=steps.get)
        print(f"subjects {judge.tolist()}, top-5 subject-agg by Euler steps:", steps, "-> n_steps =", model.n_steps)
        if model.n_steps == 0:
            model.prior = None

    print("final, held-out subjects on test images:", rounded(evaluate(model, packed, val_idx)))
    seen = np.flatnonzero(packed.test & ~np.isin(packed.subject, held))
    print("final, training subjects on test images:", rounded(evaluate(model, packed, seen)))

    folder = Path(args.out_dir) / args.name
    zip_path = save_submission(model, folder)
    packaged = load_submission(folder, packed.ch_names, device)
    X = packed.windows(val_idx[:32], device)
    gap = (torch.as_tensor(packaged.predict(X)).cpu() - torch.as_tensor(model.predict(X)).cpu()).abs().max()
    print(f"packaged vs trained model, max difference: {gap:.2e}")
    if gap > 1e-3:
        raise SystemExit("the packaged model differs from the trained one, do not upload")
    check_zip(zip_path)
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
