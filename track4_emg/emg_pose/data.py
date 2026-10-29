"""Track 4 (EMG -> hand pose) data through the official neuralbench pipeline.

- EMG2Pose: 16-channel wrist sEMG at 2 kHz in microvolts, no filtering;
- targets: 20 joint angles in radians at the same rate;
- contiguous 5 s windows (T = 10000), windows with invalid pose frames dropped;
- split: the paper's train / val, test = unseen users x unseen movement stages
  (the warm-up evaluation set).

Batches are ``(X, y, subject)``: X (B, 16, T), y (B, 20, T), subject (B,) int.

The full study is large (about 330 GB to download and about 440 GB of cache):

    python -m emg_pose.data --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

# DataLoader workers pass batches through shared memory; in containers /dev/shm
# is often tiny (64 MB) and the worker dies with a bus error, taking the Jupyter
# kernel with it. Sharing through files avoids /dev/shm.
torch.multiprocessing.set_sharing_strategy("file_system")

TASK = ("emg", "pose")


class Windows(Dataset):
    """A prepared neuralset split as tensors (X, y, subject id)."""

    def __init__(self, segments, subject_ids):
        self.segments = segments
        subjects = [str(seg.trigger.to_dict()["subject"]) for seg in segments.segments]
        self.subject = np.array([subject_ids[s] for s in subjects])

    def __len__(self):
        return len(self.subject)

    def __getitem__(self, i):
        data = self.segments[int(i)].data  # neuralset accepts only a plain int
        X = torch.as_tensor(np.asarray(data["neuro"]), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(data["target"]), dtype=torch.float32)
        return X.reshape(-1, X.shape[-1]), y.reshape(-1, y.shape[-1]), int(self.subject[i])


def _task_config(data_dir):
    from neuralbench.experiment_config import merge_task_config

    cfg = merge_task_config(*TASK)["data"]
    cache = str(data_dir / "cache")
    cfg.update({
        "study.source.path": str(data_dir),
        "study.source.infra.folder": cache,
        "neuro.infra.cluster": None,
        "neuro.infra.folder": cache,
        "neuro.infra.permissions": None,
        "target.infra.cluster": None,
        "target.infra.folder": cache,
        "target.infra.permissions": None,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    })
    return cfg


def loaders(data_dir="data", batch_size=64, num_workers=8):
    """{"train", "val", "test"} DataLoaders and meta (n_chans, n_times, n_joints, sfreq, n_subjects)."""
    from neuralbench.data import Data

    splits = Data(**_task_config(Path(data_dir).resolve())).prepare()
    names = sorted({str(seg.trigger.to_dict()["subject"]) for l in splits.values() for seg in l.dataset.segments})
    subject_ids = {s: i for i, s in enumerate(names)}
    out = {name: DataLoader(Windows(loader.dataset, subject_ids), batch_size=batch_size,
                            shuffle=name == "train", num_workers=num_workers, drop_last=name == "train",
                            pin_memory=True, persistent_workers=num_workers > 0)
           for name, loader in splits.items()}
    X, y, _ = out["test"].dataset[0]
    return out, {"n_chans": X.shape[0], "n_times": X.shape[1], "n_joints": y.shape[0], "sfreq": 2000.0,
                 "n_subjects": len(names)}


def prepare(data_dir="data"):
    """Download EMG2Pose and run the pipeline once (fills the caches)."""
    import neuralset as ns

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    source = dict(_task_config(data_dir)["study"]["source"])
    source["path"] = Path(source["path"]) / source["name"]
    ns.Study(**source).download()
    splits, meta = loaders(data_dir, num_workers=0)
    print({name: len(l.dataset) for name, l in splits.items()}, meta)


def synthetic(n_subjects=6, n_windows=48, n_chans=16, n_joints=20, n_times=4000, sfreq=2000.0,
              batch_size=16, seed=0):
    """Small fake data in the same format, to check the code without downloading.

    Smooth random joint trajectories; each electrode's EMG power follows a mix of
    the joints, with a subject-specific gain per electrode and band rotation.
    """
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(n_chans, n_joints)) / np.sqrt(n_joints)
    t = np.arange(n_times) / sfreq

    def window(subject_gain, shift):
        freqs = rng.uniform(0.2, 1.5, size=(n_joints, 1))
        phases = rng.uniform(0, 2 * np.pi, size=(n_joints, 1))
        y = 0.6 * np.sin(2 * np.pi * freqs * t + phases)
        envelope = np.exp(np.roll(mix, shift, axis=0) @ y)  # (C, T), positive
        X = 20 * subject_gain[:, None] * envelope * rng.normal(size=(n_chans, n_times))
        return X.astype(np.float32), y.astype(np.float32)

    def make(subjects):
        X, y, s = [], [], []
        for subject in subjects:
            gain = np.exp(rng.normal(0, 0.3, n_chans))
            shift = int(rng.integers(-1, 2))
            for _ in range(n_windows):
                x_w, y_w = window(gain, shift)
                X.append(x_w); y.append(y_w); s.append(subject)
        return TensorDataset(torch.tensor(np.array(X)), torch.tensor(np.array(y)), torch.tensor(s))

    splits = {"train": make(range(n_subjects - 2)), "val": make([n_subjects - 2]), "test": make([n_subjects - 1])}
    out = {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train", drop_last=name == "train")
           for name, ds in splits.items()}
    return out, {"n_chans": n_chans, "n_times": n_times, "n_joints": n_joints, "sfreq": sfreq,
                 "n_subjects": n_subjects}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    prepare(parser.parse_args().data_dir)
