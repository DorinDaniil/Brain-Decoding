"""Train both stages and build the Codabench ZIP in one command.

    python -m image_decoding.make_submission --study xu2025alljoined
    python -m image_decoding.make_submission --synthetic          # code check, no download

Stage 1 (encoder) trains on ``train`` and keeps its best val epoch; stage 2
(flow prior) trains on the encoder's outputs on ``val``, with 10% of the val
images held out to pick its epoch and the number of Euler steps. The ZIP is
checked against the contract, and the packaged model must predict like the
trained one. Warm-up scores are indicative only: its test images are public.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from image_decoding import data
from image_decoding.encoder import train_encoder
from image_decoding.evaluation import evaluate, retrieval
from image_decoding.export import check_zip, load_submission, save_submission
from image_decoding.prior import embed_split, fit_prior, refined_prediction, split_by_image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="xu2025alljoined", choices=sorted(data.STUDIES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true", help="tiny fake data, to check the code")
    parser.add_argument("--epochs", type=int, default=30, help="encoder epochs")
    parser.add_argument("--prior-epochs", type=int, default=200, help="0: no flow prior")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--name", default="clip_flow")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders, meta = data.synthetic() if args.synthetic else data.loaders(args.study, args.data_dir)
    print({name: len(l.dataset) for name, l in loaders.items()}, {k: v for k, v in meta.items() if k != "ch_names"})

    model, _ = train_encoder(loaders, meta, epochs=args.epochs, device=device)
    print("stage 1, test:", {k: round(v, 2) for k, v in evaluate(model, loaders["test"]).items()})

    if args.prior_epochs > 0:
        e, y, s = embed_split(model, loaders["val"])
        held = split_by_image(y, fraction=0.1)
        fit_prior(model, e[~held], y[~held], epochs=args.prior_epochs, monitor=(e[held], y[held], s[held]))
        steps = {n: retrieval(refined_prediction(model, e[held], n), y[held], s[held])["top5"]
                 for n in (0, 1, 2, 5, 10, 20)}
        model.n_steps = max(steps, key=steps.get)
        print("held-out val top-5 by Euler steps:", steps, "-> n_steps =", model.n_steps)
        print("stages 1 + 2, test:", {k: round(v, 2) for k, v in evaluate(model, loaders["test"]).items()})

    folder = Path(args.out_dir) / args.name
    zip_path = save_submission(model, folder)
    packaged = load_submission(folder, meta["ch_names"], device)
    X = next(iter(loaders["test"]))[0][:32]
    gap = (torch.as_tensor(packaged.predict(X)).cpu() - torch.as_tensor(model.predict(X)).cpu()).abs().max()
    print(f"packaged vs trained model, max difference: {gap:.2e}")
    if gap > 1e-3:
        raise SystemExit("the packaged model differs from the trained one, do not upload")
    check_zip(zip_path)
    print(f"upload {zip_path.resolve()} in My Submissions")


if __name__ == "__main__":
    main()
