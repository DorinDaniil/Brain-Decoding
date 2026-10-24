"""Track 2 test scores computed by the competition's own code.

Uses a clone of github.com/neural-interfaces26/2026-competition: their data
wrapper (``load_task`` with their context = interface and acquisition order),
their streamed prediction and their ``Objective.evaluate_result``, on the same
data folder as ``bci.data``. ``bci.evaluation`` is our copy of the same rules,
for validation and cross-validation splits, which their code does not cover.

    scores = official_test_scores(model)  # balanced_accuracy_cell_mean, ..._session_mean, ...

The clone is looked up in $COMPETITION_REPO, else next to this repository:
    git clone https://github.com/neural-interfaces26/2026-competition ../2026-competition
"""

import importlib.util
import os
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[3] / "2026-competition"


def _competition_repo(repo=None):
    repo = Path(repo or os.environ.get("COMPETITION_REPO", DEFAULT_REPO)).resolve()
    if not (repo / "benchmark_utils").is_dir():
        raise FileNotFoundError(f"no competition repo at {repo}; clone it there or set COMPETITION_REPO")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))  # their benchmark_utils
    return repo


def _load_module(name, path):
    # benchopt reconfigures sys.stdout on Windows, which a Jupyter stream does not support
    if not hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure = lambda **kwargs: None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_test_loader(data_dir="data", repo=None):
    """Their Track 2 test loader for PROTEUS (one window per batch, streams in order)."""
    track = _competition_repo(repo) / "tracks" / "bci_decoding"
    studies = _load_module("compet_bci_studies", track / "datasets" / "bci_studies.py")
    from benchmark_utils.nb_task import load_task

    loaders, meta = load_task(
        "eeg", studies.TASK, data_dir=Path(data_dir).resolve(), num_workers=0,
        target_transform=lambda y: y.argmax(-1),
        context_of=studies._CONTEXTS["dreyer2026proteus"],
        order_of=studies._ORDERS["dreyer2026proteus"],
    )
    return loaders["test"], meta


def official_test_scores(model, data_dir="data", repo=None):
    """Score ``model`` (``predict`` + optional ``reset_state``) exactly as the platform does.

    Returns their numbers under our names (as ``bci.evaluation.score``):
    cell_mean = balanced_accuracy_cell_mean (leaderboard),
    session_mean = balanced_accuracy_session_mean (published baselines),
    pooled = balanced_accuracy, accuracy = accuracy.
    """
    track = _competition_repo(repo) / "tracks" / "bci_decoding"
    objective = _load_module("compet_bci_objective", track / "objective.py").Objective()
    loader, meta = official_test_loader(data_dir, repo)
    objective.set_data(train_loader=None, test_loader=loader,
                       n_classes=int(meta.pop("raw_target_shape")[-1]),
                       **{k: v for k, v in meta.items() if k != "target_shape"})
    theirs = objective.evaluate_result(model)
    return {"cell_mean": theirs["balanced_accuracy_cell_mean"],
            "session_mean": theirs["balanced_accuracy_session_mean"],
            "pooled": theirs["balanced_accuracy"], "accuracy": theirs["accuracy"]}
