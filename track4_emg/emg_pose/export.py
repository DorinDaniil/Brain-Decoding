"""Package a PoseEnsemble for Codabench and check the package locally.

``submission.py`` is generated from ``nets.py`` plus a small ``Solver``; the
weights go to ``weights.pt`` as plain tensors and numbers (``weights_only=True``).

    zip_path = save_submission(ensemble, "submissions/posenet")
    check_zip(zip_path)
"""

import ast
import importlib.util
import shutil
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import torch

NETS = Path(__file__).with_name("nets.py")

SOLVER = '''

# ---------------------------------------------------------------- Codabench entry point
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "PoseEnsemble"

    def load_model(self, meta):
        state = torch.load(meta["submission_dir"] / "weights.pt", map_location="cpu", weights_only=True)
        nets = []
        for config, weights in zip(state["configs"], state["weights"]):
            net = PoseNet(**config)
            net.load_state_dict(weights)
            nets.append(net)
        return PoseEnsemble(nets, state["tta_offsets"]).to(meta["device"]).eval()
'''

WORKER_MODULES = {"torch", "math", "benchmark_utils"}


def save_submission(ensemble, folder):
    """Write submission.py + weights.pt into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "submission.py").write_text(NETS.read_text(encoding="utf-8") + SOLVER, encoding="utf-8")
    torch.save({
        "configs": [net.config for net in ensemble.nets],
        "weights": [{k: v.detach().cpu() for k, v in net.state_dict().items()} for net in ensemble.nets],
        "tta_offsets": list(ensemble.tta_offsets),
    }, folder / "weights.pt")
    return Path(shutil.make_archive(str(folder), "zip", folder))


def load_submission(folder, device="cpu"):
    """Load the packaged model the way the platform does (``Solver().load_model(meta)``)."""
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils"))
    sys.modules.setdefault("benchmark_utils.base_solver", stub)
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location(f"submission_{id(folder)}", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {"submission_dir": folder, "device": device, "n_chans": 16, "n_times": 10000,
            "sfreq": 2000.0, "n_joints": 20}
    return module.Solver().load_model(meta)


def check_zip(zip_path, batch_size=8, n_times=10000):
    """Contract checks without data: root files, imports, (B, 20, T) radians, no state across batches."""
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
            model = load_submission(Path(tmp))
            X = 30 * torch.randn(batch_size, 16, n_times)
            start = time.perf_counter()
            out = torch.as_tensor(model.predict(X))
            per_window = (time.perf_counter() - start) / batch_size
            if out.shape != (batch_size, 20, n_times) or not torch.isfinite(out).all():
                raise ValueError(f"predict must return finite (B, 20, T), got {tuple(out.shape)}")
            if not torch.allclose(torch.as_tensor(model.predict(X[:3])), out[:3], atol=1e-4):
                raise ValueError("a window's prediction depends on the rest of the batch")
    print(f"{zip_path.name}: OK  {zip_path.stat().st_size / 2**20:.1f} MB, {len(model.nets)} nets x "
          f"{len(model.tta_offsets)} rotations, {per_window * 1000:.0f} ms per window on CPU")
