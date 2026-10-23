"""Package a trained model for Codabench and check the package locally.

The platform runs one self-contained ``submission.py`` (no own modules), so a
model is exported as plain numpy arrays in ``model.pkl`` and read back by
``submission/submission.py``. Supported: logreg, forest.

    save_submission(model, "submissions/forest", ch_names)  # folder + .zip
    packaged = load_submission("submissions/forest", ch_names)
"""

import importlib.util
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np

from bci.data import SFREQ

TEMPLATE = Path(__file__).resolve().parent.parent / "submission" / "submission.py"


def _expert(features, clf, channels=None):
    """One Riemannian expert: recentering prior + scaler + softmax head."""
    return {
        "band": features.band,
        "channels": None if channels is None else np.asarray(channels),
        "prior_weight": features.prior_weight,
        "log_prior": features.log_prior,
        "x_mean": features.x_scaler.mean_,
        "x_scale": features.x_scaler.scale_,
        "coef": clf.coef_,
        "intercept": clf.intercept_,
    }


def model_spec(model, ch_names):
    """Plain-array description of a trained logreg / forest model (a list of experts)."""
    if hasattr(model, "trees"):  # forest
        experts = [_expert(t.features, t.clf, t.channels) for t in model.trees]
    elif hasattr(model, "clf"):  # logreg: a forest of one tree
        experts = [_expert(model.features, model.clf)]
    else:
        raise NotImplementedError(f"export of {type(model).__name__} is not supported")
    return {"experts": experts, "ch_names": [str(c) for c in ch_names], "sfreq": SFREQ}


def save_submission(model, folder, ch_names):
    """Write submission.py + model.pkl into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, folder / "submission.py")
    with open(folder / "model.pkl", "wb") as f:
        pickle.dump(model_spec(model, ch_names), f)
    zip_path = shutil.make_archive(str(folder), "zip", folder)  # files at the root of the zip
    print(f"{zip_path}: upload this file in My Submissions")
    return Path(zip_path)


def load_submission(folder, ch_names):
    """Load the packaged model the way the platform does (``Solver().load_model(meta)``)."""
    # submission.py imports CompetSolver from the competition repo; stand in for it.
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils"))
    sys.modules.setdefault("benchmark_utils.base_solver", stub)

    folder = Path(folder)
    spec = importlib.util.spec_from_file_location("submission", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {"submission_dir": folder, "ch_names": list(ch_names), "sfreq": SFREQ,
            "device": "cpu", "n_classes": 3}
    return module.Solver().load_model(meta)
