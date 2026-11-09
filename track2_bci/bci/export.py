"""Package a trained model for Codabench and check the package locally.

The platform runs one self-contained ``submission.py`` (no own modules), so it
is generated from ``bci/stream.py`` plus a small ``Solver``; the model goes to
``model.npz`` as arrays plus a JSON skeleton (nothing pickled, readable by any
numpy version). Accepts a ``StreamModel`` and the older logreg / forest models.

    save_submission(model, "submissions/all", ch_names)   # folder + .zip
    packaged = load_submission("submissions/all", ch_names)
    check_zip("submissions/all.zip")                       # contract checks, no data needed
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
from bci.stream import save_spec

STREAM = Path(__file__).with_name("stream.py")

SOLVER = '''

# ---------------------------------------------------------------- Codabench entry point
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "RiemannStream"

    def load_model(self, meta):
        spec = load_spec(meta["submission_dir"] / "model.npz")
        if abs(float(meta.get("sfreq", spec["sfreq"])) - spec["sfreq"]) > 1e-6:
            raise ValueError(f"model trained at {spec['sfreq']} Hz, data at {meta['sfreq']} Hz")
        torch.set_num_threads(1)  # one window at a time: threads only add overhead
        names = list(meta.get("ch_names") or spec["ch_names"])
        return StreamModel(spec, [names.index(n) for n in spec["ch_names"]])
'''

# Top-level imports available on the Codabench worker (its requirements.txt + stdlib we use).
WORKER_MODULES = {"numpy", "scipy", "sklearn", "torch", "json", "math", "benchmark_utils"}


def spec_from_model(model, ch_names):
    """The StreamModel spec of a model: its own, or one built from an older logreg / forest."""
    if hasattr(model, "spec"):
        return model.spec
    from bci.ensemble import BANDS

    def expert(features, clf, channels=None):
        return {"band": BANDS.index(tuple(features.band)), "group": -1,
                "channels": None if channels is None else np.asarray(channels),
                "prior_weight": features.prior_weight, "log_prior": features.log_prior,
                "x_mean": features.x_scaler.mean_, "x_scale": features.x_scaler.scale_,
                "coef": clf.coef_, "intercept": clf.intercept_}

    if hasattr(model, "trees"):  # forest
        experts = [expert(t.features, t.clf, t.channels) for t in model.trees]
    elif hasattr(model, "clf"):  # logreg: a forest of one tree
        experts = [expert(model.features, model.clf)]
    else:
        raise NotImplementedError(f"export of {type(model).__name__} is not supported")
    return {"sfreq": SFREQ, "ch_names": [str(c) for c in ch_names], "bands": [list(b) for b in BANDS],
            "experts": experts, "router": None, "deep": None, "balance": None}


def save_submission(model, folder, ch_names):
    """Write submission.py + model.npz into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "submission.py").write_text(STREAM.read_text(encoding="utf-8") + SOLVER, encoding="utf-8")
    save_spec(spec_from_model(model, ch_names), folder / "model.npz")
    return Path(shutil.make_archive(str(folder), "zip", folder))  # files at the root of the zip


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


def check_zip(zip_path, n_windows=220, n_classes=3, sealed_windows=25_000, seed=0):
    """Check a submission ZIP against the Codabench contract, without any data.

    Files at the root, imports available on the worker, ``Solver().load_model``
    works, ``deepcopy`` + ``reset_state`` + one-window ``predict`` over a session-long
    stream (``n_windows``, so session-growing costs are counted) return one
    class index per window. Prints the size and the time per window, against
    the 60-minute limit for about ``sealed_windows`` sealed windows.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if "/" in n.rstrip("/")]
        if bad or "submission.py" not in names:
            raise ValueError(f"files must sit at the root of the zip with submission.py: {names}")
        source = z.read("submission.py").decode("utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        missing = imported - WORKER_MODULES
        if missing:
            raise ValueError(f"not available on the worker: {sorted(missing)}")
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            model = load_submission(Path(tmp))
            rng = np.random.default_rng(seed)
            n_chans = len(model.spec["ch_names"])
            stream = copy.deepcopy(model)
            stream.reset_state()
            start = time.perf_counter()
            for _ in range(n_windows):
                X = torch.from_numpy(rng.standard_normal((1, n_chans, 480)).astype(np.float32) * 10)
                y = np.asarray(stream.predict(X))
                if y.shape != (1,) or not np.issubdtype(y.dtype, np.integer) or not 0 <= y[0] < n_classes:
                    raise ValueError(f"predict must return one class index in [0, {n_classes}), got {y!r}")
            per_window = (time.perf_counter() - start) / n_windows

    spec = model.spec
    parts = [f"{len(spec['experts'])} experts"] + [k for k in ("router", "deep", "balance", "transduction") if spec.get(k)]
    total = per_window * sealed_windows / 60
    print(f"{zip_path.name}: OK  {zip_path.stat().st_size / 2**20:.1f} MB, {', '.join(parts)}, "
          f"{per_window * 1000:.0f} ms per window on CPU (~{total:.0f} min for {sealed_windows} windows, limit 60)")
    if total > 50:
        print("warning: close to the 60-minute limit")
    return per_window
