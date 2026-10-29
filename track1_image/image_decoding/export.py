"""Package a trained ImageDecoder for Codabench and check the package locally.

The platform runs one self-contained ``submission.py``, so it is generated from
``nets.py`` (the networks) plus a small ``Solver``; the weights go to
``weights.pt`` as plain tensors and numbers (loadable with ``weights_only=True``).

    zip_path = save_submission(model, "submissions/spatial_conv")
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

    name = "ImageDecoder"

    def load_model(self, meta):
        state = torch.load(meta["submission_dir"] / "weights.pt", map_location="cpu", weights_only=True)
        config = state["config"]
        prior = VectorUNet(**config["prior"]) if config["prior"] is not None else None
        positions = dict(zip(config["table_names"], state["table"].tolist()))
        model = ImageDecoder([SpatialConvEncoder(**c) for c in config["encoders"]], state["mu"], state["s"],
                             state["radius"], positions, config["ch_names"], state["scales"], prior, config["n_steps"])
        model.load_state_dict(state["model"])
        if meta.get("ch_names"):
            model.match_channels(meta["ch_names"])
        return model.to(meta["device"]).eval()
'''

WORKER_MODULES = {"torch", "math", "benchmark_utils"}


def save_submission(model, folder):
    """Write submission.py + weights.pt into ``folder`` and zip them as ``folder.zip``."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "submission.py").write_text(NETS.read_text(encoding="utf-8") + SOLVER, encoding="utf-8")
    config = {
        "encoders": [enc.config for enc in model.encoders],
        "prior": None if model.prior is None else model.prior.config,
        "n_steps": int(model.n_steps),
        "ch_names": list(model.ch_names),
        "table_names": list(model.table_names),
    }
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": config, "model": state, **{k: state[k] for k in ("mu", "s", "radius", "scales", "table")}},
               folder / "weights.pt")
    return Path(shutil.make_archive(str(folder), "zip", folder))


def load_submission(folder, ch_names=None, device="cpu"):
    """Load the packaged model the way the platform does (``Solver().load_model(meta)``)."""
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils"))
    sys.modules.setdefault("benchmark_utils.base_solver", stub)
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location(f"submission_{id(folder)}", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {"submission_dir": folder, "device": device, "n_outputs": 1536}
    if ch_names is not None:
        meta["ch_names"] = list(ch_names)
    return module.Solver().load_model(meta)


def check_zip(zip_path, batch_size=64, sealed_windows=200_000):
    """Contract checks without data: root files, imports, (B, 1536) output, no state across batches.

    Also times the CPU forward pass against the 3600 s limit, for ``sealed_windows`` windows.
    """
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
            config = torch.load(Path(tmp) / "weights.pt", weights_only=True)["config"]
            train_names, n_times = config["ch_names"], config["encoders"][0]["n_times"]
            # the evaluation montage in another order, with one extra and one missing channel
            eval_names = list(reversed(train_names[1:])) + ["EXTRA"]
            model = load_submission(Path(tmp), eval_names)
            X = torch.randn(batch_size, len(eval_names), n_times)
            start = time.perf_counter()
            out = torch.as_tensor(model.predict(X))
            per_batch = time.perf_counter() - start
            if out.shape != (batch_size, 1536) or not torch.isfinite(out).all():
                raise ValueError(f"predict must return finite (B, 1536), got {tuple(out.shape)}")
            if not torch.allclose(torch.as_tensor(model.predict(X[:5])), out[:5], atol=1e-4):
                raise ValueError("a window's prediction depends on the rest of the batch")
    cpu_time = per_batch / batch_size * sealed_windows
    print(f"{zip_path.name}: OK  {zip_path.stat().st_size / 2**20:.1f} MB, "
          f"{per_batch * 1000 / batch_size:.1f} ms per window on CPU (~{cpu_time / 60:.0f} min for "
          f"{sealed_windows} windows), {len(config['encoders'])} encoders, montage matched by electrode name")
    if cpu_time > 3000:
        print("warning: on a CPU-only worker this may not finish within the 3600 s limit")
