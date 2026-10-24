"""Track 1 (EEG -> image) data through the official neuralbench pipeline.

- windows: -0.2..0.8 s around each image, 120 Hz, baseline on the first 0.2 s,
  robust-scaled per recording and clamped at 20 (neuralbench task defaults);
- targets: DINOv2-giant embeddings of the viewed images, shape (1536,);
- split: the study's own train / test images (disjoint images), 20% of the
  training recordings held out as validation.

Batches are ``(X, y, subject)``: X (B, C, T) float32, y (B, D) float32, subject (B,) int.

Usage (once; the DINOv2 pass over the stimuli wants a GPU):

    python -m image_decoding.data --study xu2024alljoined --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

TASK = "image"

# study -> neuralbench dataset overlay (None = the task default, THINGS-EEG2)
STUDIES = {
    "gifford2022large": None,                         # THINGS-EEG2, 63 ch, ~220 GB
    "grootswagers2022human": "grootswagers2022human",  # THINGS-EEG1
    "xu2024alljoined": "xu2024alljoined",             # Alljoined-1, ~5 GB: quick start
    "xu2025alljoined": "xu2025alljoined",             # Alljoined-1.6M: same Emotiv hardware as the sealed test
}


class Windows(Dataset):
    """A prepared neuralset split as plain tensors (X, y, subject)."""

    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments.segments)

    def __getitem__(self, i):
        data = self.segments[i].data
        X = torch.as_tensor(np.asarray(data["neuro"]), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(data["target"]), dtype=torch.float32)
        subject = int(np.asarray(data["subject_id"]).reshape(-1)[0])
        return X.reshape(-1, X.shape[-1]), y.reshape(-1), subject


def _task_config(study, data_dir):
    from neuralbench.experiment_config import merge_task_config

    cfg = merge_task_config("eeg", TASK, STUDIES[study])["data"]
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


def loaders(study="xu2024alljoined", data_dir="data", batch_size=256, num_workers=4):
    """{"train", "val", "test"} DataLoaders and meta (n_chans, n_times, dim, n_subjects)."""
    from neuralbench.data import Data

    data_dir = Path(data_dir).resolve()
    splits = Data(**_task_config(study, data_dir)).prepare()
    out = {name: DataLoader(Windows(loader.dataset), batch_size=batch_size,
                            shuffle=name == "train", num_workers=num_workers,
                            drop_last=name == "train")
           for name, loader in splits.items()}
    X, y, _ = out["test"].dataset[0]
    n_subjects = len(set().union(*(set(l.dataset.triggers.subject) for l in splits.values())))
    return out, {"n_chans": X.shape[0], "n_times": X.shape[1], "dim": y.shape[0], "n_subjects": n_subjects}


def prepare(study="xu2024alljoined", data_dir="data"):
    """Download the study and run the pipeline once (fills the caches, incl. DINOv2)."""
    import neuralset as ns

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    source = dict(_task_config(study, data_dir)["study"]["source"])
    source["path"] = Path(source["path"]) / source["name"]
    ns.Study(**source).download()
    splits, meta = loaders(study, data_dir, num_workers=0)
    print({name: len(l.dataset) for name, l in splits.items()}, meta)


def synthetic(n_subjects=4, n_train=400, n_test=50, reps=4, n_chans=32, n_times=120,
              dim=1536, noise=40.0, batch_size=256, seed=0):
    """Tiny fake data in the same format, to check the code without downloading.

    Images have low-rank random embeddings; each subject's EEG is a noisy,
    subject-specific mixing of a projection of the viewed image's embedding.
    """
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(n_train + n_test, 16)) @ rng.normal(size=(16, dim)) + 3.0
    wave = np.sin(np.linspace(0, 3 * np.pi, n_times))
    read = rng.normal(size=(dim, 8)) / np.sqrt(dim)
    mixes = [rng.normal(size=(n_chans, 8)) for _ in range(n_subjects)]

    def make(image_ids, n_reps):
        X, y, s = [], [], []
        for subject, mix in enumerate(mixes):
            for i in np.repeat(image_ids, n_reps):
                source = (images[i] @ read)[:, None] * wave + noise * rng.normal(size=(8, n_times))
                X.append(mix @ source + noise * rng.normal(size=(n_chans, n_times)))
                y.append(images[i]); s.append(subject)
        X = np.array(X)
        X /= X.std(axis=(1, 2), keepdims=True)  # unit scale, as the real pipeline's scaler
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(np.array(y), dtype=torch.float32), torch.tensor(s))

    train_ids = np.arange(n_train)
    splits = {"train": make(train_ids, reps - 1), "val": make(train_ids, 1),
              "test": make(np.arange(n_train, n_train + n_test), reps)}
    out = {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train", drop_last=name == "train")
           for name, ds in splits.items()}
    return out, {"n_chans": n_chans, "n_times": n_times, "dim": dim, "n_subjects": n_subjects}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="xu2024alljoined", choices=sorted(STUDIES))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    prepare(args.study, args.data_dir)
