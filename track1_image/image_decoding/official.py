"""Track 1 scores computed by the competition's own objective.

Runs ``Objective.evaluate_result`` of github.com/neural-interfaces26/2026-competition
(tracks/image_decoding) on any of our loaders. ``image_decoding.evaluation`` is
our copy of the same rules; tests/test_official.py checks they agree.

The clone is looked up in $COMPETITION_REPO, else next to this repository:
    git clone https://github.com/neural-interfaces26/2026-competition ../2026-competition
"""

import importlib.util
import os
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[3] / "2026-competition"


def _objective(repo=None):
    repo = Path(repo or os.environ.get("COMPETITION_REPO", DEFAULT_REPO)).resolve()
    if not (repo / "benchmark_utils").is_dir():
        raise FileNotFoundError(f"no competition repo at {repo}; clone it there or set COMPETITION_REPO")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))  # their benchmark_utils
    # benchopt reconfigures sys.stdout on Windows, which a Jupyter stream does not support
    if not hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure = lambda **kwargs: None
    path = repo / "tracks" / "image_decoding" / "objective.py"
    spec = importlib.util.spec_from_file_location("compet_image_objective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Objective()


def official_scores(model, loader, repo=None):
    """Their top-1 / top-5 (fractions, not %) of ``model.predict`` on our loader."""
    objective = _objective(repo)
    their_loader = [(X, y, {"subject_id": s}) for X, y, s in loader]
    objective.set_data(train_loader=None, test_loader=their_loader, n_outputs=None)
    return objective.evaluate_result(model)
