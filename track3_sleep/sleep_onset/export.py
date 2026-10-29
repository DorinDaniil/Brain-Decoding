"""Package an OnsetModel spec for Codabench and check the package locally.

``submission.py`` is generated from ``model.py`` plus a small ``Solver``; the
spec goes to ``model.npz`` as plain arrays (no pickled objects).

    zip_path = save_submission(spec, ch_names, "submissions/bayes_onset")
    check_zip(zip_path)
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

MODEL = Path(__file__).with_name("model.py")

SOLVER = '''

# ---------------------------------------------------------------- Codabench entry point
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "BayesOnset"

    def load_model(self, meta):
        with np.load(meta["submission_dir"] / "model.npz", allow_pickle=False) as f:
            spec = {k: f[k] for k in f.files}
        for k in ("gamma", "hop", "offset", "sfreq"):
            spec[k] = float(spec[k])
        names = [str(c) for c in spec.pop("ch_names")]
        given = [str(c).upper() for c in (meta.get("ch_names") or names)]
        return OnsetModel(spec, [given.index(n.upper()) for n in names])
'''

WORKER_MODULES = {"numpy", "scipy", "benchmark_utils"}


def save_submission(spec, ch_names, folder):
    """Write submission.py + model.npz into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "submission.py").write_text(MODEL.read_text(encoding="utf-8") + SOLVER, encoding="utf-8")
    np.savez(folder / "model.npz", ch_names=np.array(ch_names), **{k: np.asarray(v) for k, v in spec.items()})
    return Path(shutil.make_archive(str(folder), "zip", folder))


def load_submission(folder, ch_names=None):
    """Load the packaged model the way the platform does (``Solver().load_model(meta)``)."""
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils"))
    sys.modules.setdefault("benchmark_utils.base_solver", stub)
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location(f"submission_{id(folder)}", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {"submission_dir": folder, "device": "cpu", "sfreq": 128.0, "n_outputs": 1}
    if ch_names is not None:
        meta["ch_names"] = list(ch_names)
    return module.Solver().load_model(meta)


def check_zip(zip_path, n_windows=200, seed=0):
    """Contract checks without data: root files, imports, one finite latency in [0, 600] per window."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if any("/" in n.rstrip("/") for n in names) or "submission.py" not in names:
            raise ValueError(f"files must sit at the root of the zip with submission.py: {names}")
        imported = set()
        for node in ast.walk(ast.parse(z.read("submission.py").decode("utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        if imported - WORKER_MODULES:
            raise ValueError(f"not available on the worker: {sorted(imported - WORKER_MODULES)}")
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            model = load_submission(Path(tmp), ["TP10", "AF8", "AF7", "TP9"])  # another channel order
            stream = copy.deepcopy(model)
            stream.reset_state()
            rng = np.random.default_rng(seed)
            start = time.perf_counter()
            for _ in range(n_windows):
                y = np.asarray(stream.predict(torch.from_numpy(20 * rng.standard_normal((1, 4, 640)).astype(np.float32))))
                if y.shape != (1,) or not np.isfinite(y).all() or not 0 <= y[0] <= 600:
                    raise ValueError(f"predict must return one latency in [0, 600] s, got {y!r}")
            per_window = (time.perf_counter() - start) / n_windows
    print(f"{zip_path.name}: OK  {zip_path.stat().st_size / 2**20:.2f} MB, {per_window * 1000:.1f} ms per window")
