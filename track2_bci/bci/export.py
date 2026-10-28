"""Package a trained model for Codabench and check the package locally.

The platform runs one self-contained ``submission.py`` (no own modules), so a
model is exported as plain numpy arrays in ``model.npz`` (no pickled objects,
readable by any numpy version) and read back by
``submission/submission.py``. Supported: logreg, forest.

    save_submission(model, "submissions/forest", ch_names)  # folder + .zip
    packaged = load_submission("submissions/forest", ch_names)
    check_zip("submissions/forest.zip")                      # contract checks, no data needed
"""

import ast
import copy
import importlib.util
import shutil
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch

from bci.data import SFREQ

TEMPLATE = Path(__file__).resolve().parent.parent / "submission" / "submission.py"

# Top-level imports available on the Codabench worker (its requirements.txt + stdlib we use).
WORKER_MODULES = {"numpy", "scipy", "sklearn", "torch", "copy", "math", "pathlib",
                  "benchmark_utils"}


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


def write_spec(spec, path):
    """Flatten a model spec into ``model.npz``: one array per expert field."""
    arrays = {"n_experts": len(spec["experts"]), "ch_names": np.array(spec["ch_names"]),
              "sfreq": spec["sfreq"]}
    for i, e in enumerate(spec["experts"]):
        for k in ("band", "prior_weight", "log_prior", "x_mean", "x_scale", "coef", "intercept"):
            arrays[f"e{i}_{k}"] = np.asarray(e[k], dtype=np.float64)
        channels = e["channels"]
        arrays[f"e{i}_channels"] = np.zeros(0, np.int64) if channels is None else np.asarray(channels, np.int64)
    np.savez(path, **arrays)


def save_submission(model, folder, ch_names):
    """Write submission.py + model.npz into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, folder / "submission.py")
    write_spec(model_spec(model, ch_names), folder / "model.npz")
    zip_path = shutil.make_archive(str(folder), "zip", folder)  # files at the root of the zip
    return Path(zip_path)


def _stub_competition_repo():
    """submission.py imports CompetSolver from the competition repo; stand in for it."""
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils"))
    sys.modules.setdefault("benchmark_utils.base_solver", stub)


def load_submission(folder, ch_names=None):
    """Load the packaged model the way the platform does (``Solver().load_model(meta)``)."""
    _stub_competition_repo()
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location(f"submission_{id(folder)}", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {"submission_dir": folder, "sfreq": SFREQ, "device": "cpu", "n_classes": 3}
    if ch_names is not None:
        meta["ch_names"] = [str(c) for c in ch_names]
    return module.Solver().load_model(meta)


def check_zip(zip_path, n_windows=20, n_classes=3, seed=0):
    """Check a submission ZIP against the Codabench contract, without any data.

    Files at the root, imports available on the worker, ``Solver().load_model``
    works, ``deepcopy`` + ``reset_state`` + one-window ``predict`` return one
    class index per window. Prints the size and the time per window.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if "/" in n.rstrip("/")]
        if bad or "submission.py" not in names:
            raise ValueError(f"files must sit at the root of the zip with submission.py: {names}")
        source = z.read("submission.py").decode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            model = load_submission(Path(tmp))

            imported = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            missing = imported - WORKER_MODULES
            if missing:
                raise ValueError(f"not available on the worker: {sorted(missing)}")

            rng = np.random.default_rng(seed)
            n_chans = len(getattr(model, "channel_index", [])) or 41
            stream = copy.deepcopy(model)
            if hasattr(stream, "reset_state"):
                stream.reset_state()
            start = time.perf_counter()
            for _ in range(n_windows):
                X = torch.from_numpy(rng.standard_normal((1, n_chans, 480)).astype(np.float32) * 10)
                y = np.asarray(stream.predict(X))
                if y.shape != (1,) or not np.issubdtype(y.dtype, np.integer) or not 0 <= y[0] < n_classes:
                    raise ValueError(f"predict must return one class index in [0, {n_classes}), got {y!r}")
            per_window = (time.perf_counter() - start) / n_windows

    size_mb = zip_path.stat().st_size / 2**20
    n_experts = len(getattr(model, "experts", []))
    print(f"{zip_path.name}: OK  {size_mb:.1f} MB, {n_experts} experts, "
          f"{per_window * 1000:.0f} ms per window on CPU "
          f"(~{per_window * 5000 / 60:.0f} min for 5000 windows, limit 60 min)")
    return per_window
