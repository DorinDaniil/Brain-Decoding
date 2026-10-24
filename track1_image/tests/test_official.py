"""Our Track 1 retrieval metrics must give the competition objective's numbers.

    cd track1_image && python -m pytest tests -q

Needs the competition repo (see image_decoding/official.py).
"""

import numpy as np
import pytest
import torch

from image_decoding import data
from image_decoding.evaluation import evaluate
from image_decoding.official import official_scores


class FixedModel:
    """predict(X) -> noisy copy of the true embedding, looked up from X's first value."""

    def __init__(self, table, noise, seed):
        self.table, self.noise = table, noise
        self.rng = np.random.default_rng(seed)

    def predict(self, X):
        idx = X[:, 0, 0].long().numpy()
        out = self.table[idx] + self.noise * self.rng.normal(size=(len(idx), self.table.shape[1]))
        return torch.as_tensor(out, dtype=torch.float32)


@pytest.mark.parametrize("noise", [0.5, 2.0, 5.0])
def test_retrieval_matches_their_objective(noise):
    rng = np.random.default_rng(0)
    n_images, reps, dim = 60, 3, 32
    table = rng.normal(size=(n_images, dim)) + 1.0
    image = np.tile(np.repeat(np.arange(n_images), reps), 4)
    subject = np.repeat(np.arange(4), n_images * reps)
    X = torch.zeros(len(image), 2, 4)
    X[:, 0, 0] = torch.as_tensor(image, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X, torch.as_tensor(table[image], dtype=torch.float32),
                                        torch.as_tensor(subject))
    loader = torch.utils.data.DataLoader(ds, batch_size=50)

    ours = evaluate(FixedModel(table, noise, seed=1), loader)
    theirs = official_scores(FixedModel(table, noise, seed=1), loader)
    their_keys = {"top1": "top1_acc", "top5": "top5_acc",
                  "top1_subject_agg": "top1_acc_subject_agg", "top5_subject_agg": "top5_acc_subject_agg"}
    for ours_key, their_key in their_keys.items():
        assert ours[ours_key] == pytest.approx(100 * theirs[their_key])


def test_synthetic_pipeline_matches_their_objective():
    """End to end on synthetic data: a briefly trained encoder, scored both ways."""
    from image_decoding.encoder import train_encoder

    loaders, meta = data.synthetic(n_train=100, n_test=30)
    encoder, _ = train_encoder(loaders, meta, epochs=1)
    ours = evaluate(encoder, loaders["test"])
    theirs = official_scores(encoder, loaders["test"])
    assert ours["top5"] == pytest.approx(100 * theirs["top5_acc"])
    assert ours["top5_subject_agg"] == pytest.approx(100 * theirs["top5_acc_subject_agg"])
