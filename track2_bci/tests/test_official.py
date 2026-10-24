"""Our Track 2 evaluation must give the competition's numbers.

    cd track2_bci && python -m pytest tests -q

Needs the competition repo (see bci/official.py); the real-data tests also
need ``python -m bci.data --data-dir data`` (or $BCI_DATA_DIR).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bci import evaluation
from bci.data import Windows
from bci.official import _competition_repo, official_test_loader, official_test_scores

DATA_DIR = Path(os.environ.get("BCI_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
needs_data = pytest.mark.skipif(not (DATA_DIR / "proteus_test.npz").exists(), reason="no prepared data")


@pytest.fixture(scope="module")
def metrics():
    _competition_repo()
    from benchmark_utils import metrics
    return metrics


def _random_windows(n, rng):
    sessions = rng.integers(0, 6, n).astype(str)
    return Windows(X=np.zeros((n, 1, 1), np.float32), y=rng.integers(0, 3, n),
                   subject=rng.integers(0, 4, n).astype(str), session=sessions,
                   context=rng.choice(["Graz", "BrainHero"], n), run=np.ones(n, int),
                   onset=np.arange(n, dtype=float), ch_names=np.array(["C"]))


@pytest.mark.parametrize("seed", range(20))
def test_scores_match_their_metrics(metrics, seed):
    rng = np.random.default_rng(seed)
    w = _random_windows(300, rng)
    pred = rng.integers(0, 3, len(w))
    if seed % 2:  # a class absent from some cells: their recall-0 rule must match
        pred[w.y == 0] = 2
    ours = evaluation.score(w, pred)
    stream = np.unique(w.stream, return_inverse=True)[1]
    context = np.unique(w.context, return_inverse=True)[1]
    cells = metrics.group_scores(metrics.balanced_accuracy, w.y, pred, np.stack([stream, context], 1))
    sessions = metrics.group_scores(metrics.balanced_accuracy, w.y, pred, stream)
    assert ours["cell_mean"] == pytest.approx(cells.mean())
    assert ours["session_mean"] == pytest.approx(sessions.mean())
    assert ours["pooled"] == pytest.approx(metrics.balanced_accuracy(w.y, pred))


@needs_data
def test_our_test_split_is_theirs():
    """Same windows, labels and stream order as their test loader."""
    from bci.data import load_split

    ours = load_split("test", DATA_DIR)
    loader, _ = official_test_loader(DATA_DIR)
    X, y, stream = [], [], []
    for xb, yb, info in loader:
        X.append(xb.numpy()); y.append(yb.numpy()); stream.append(np.asarray(info["stream_id"]))
    X, y, stream = np.concatenate(X), np.concatenate(y), np.concatenate(stream)
    assert X.shape == ours.X.shape
    np.testing.assert_allclose(X, ours.X, atol=1e-4)
    np.testing.assert_array_equal(y, ours.y)
    # streams: same partition into contiguous blocks, in the same order
    ours_blocks = np.unique(ours.stream, return_inverse=True)[1]
    assert (np.diff(stream) != 0).sum() == (np.diff(ours_blocks) != 0).sum()


@needs_data
def test_logreg_scores_match_official():
    from bci.data import load_split
    from bci.models import logreg

    train, test = load_split("train", DATA_DIR), load_split("test", DATA_DIR)
    model = logreg.fit(train)
    ours = evaluation.evaluate(model, test)
    theirs = official_test_scores(model, DATA_DIR)
    for key in ["cell_mean", "session_mean", "pooled", "accuracy"]:
        assert ours[key] == pytest.approx(theirs[key])
