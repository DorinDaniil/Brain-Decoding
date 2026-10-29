"""Track 1 (EEG -> image) data through the official neuralbench pipeline, packed once.

- windows: -0.2..0.8 s around each image, 120 Hz, baseline on the first 0.2 s,
  robust-scaled per recording and clamped at 20 (neuralbench task defaults),
  exactly what the platform feeds the model;
- targets: DINOv2-giant embeddings of the viewed images, shape (1536,);
- ``test`` images: the study's own test images, never trained on.

The pipeline is run once and its output packed into ``data/packed/<study>``:
X (N, C, T) float16, the image and subject of every window, the unique image
embeddings, channel names and electrode positions. Training then reads plain
arrays held in memory (on the GPU when it fits): no DataLoader workers.

Usage (once; the DINOv2 pass over the stimuli wants a GPU):

    python -m image_decoding.data --study xu2025alljoined --data-dir data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# DataLoader workers pass batches through shared memory; in containers /dev/shm
# is often tiny (64 MB) and the worker dies with a bus error, taking the Jupyter
# kernel with it. Sharing through files avoids /dev/shm.
torch.multiprocessing.set_sharing_strategy("file_system")

TASK = "image"

# study -> neuralbench dataset overlay (None = the task default, THINGS-EEG2)
STUDIES = {
    "gifford2022large": None,                         # THINGS-EEG2, 63 ch, ~220 GB
    "grootswagers2022human": "grootswagers2022human",  # THINGS-EEG1
    "xu2024alljoined": "xu2024alljoined",             # Alljoined-1, ~5 GB: quick start
    "xu2025alljoined": "xu2025alljoined",             # Alljoined-1.6M: same Emotiv hardware as the sealed test
}


class Packed:
    """All windows of a study: X (N, C, T), image (N,), subject (N,), test (N,) bool, images (n_images, D).

    ``positions``: electrode name -> xyz in metres, for every name the model may
    meet (the standard 10-05 table), not only the training channels.
    """

    def __init__(self, X, image, subject, test, images, ch_names, positions):
        self.X = torch.as_tensor(X)
        self.image, self.subject, self.test = np.asarray(image), np.asarray(subject), np.asarray(test, bool)
        self.images = np.asarray(images, np.float32)
        self.ch_names, self.positions = list(ch_names), dict(positions)

    def __len__(self):
        return len(self.image)

    def windows(self, index, device):
        return self.X[torch.as_tensor(index, device=self.X.device)].to(device, torch.float32, non_blocking=True)

    def to(self, device, margin=4 * 2**30):
        """Move X to ``device`` when it fits with ``margin`` bytes to spare (else keep it in RAM)."""
        device = torch.device(device)
        need = self.X.numel() * self.X.element_size()
        if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < need + margin:
            print(f"X ({need / 2**30:.1f} GB) stays in RAM: not enough free GPU memory")
            return self
        X = torch.empty(self.X.shape, dtype=self.X.dtype, device=device)
        for i in range(0, len(X), 65536):
            X[i:i + 65536] = self.X[i:i + 65536].to(device)
        self.X = X
        return self

    def summary(self):
        return {"windows": len(self), "test windows": int(self.test.sum()), "images": len(self.images),
                "test images": len(np.unique(self.image[self.test])), "subjects": len(np.unique(self.subject)),
                "channels": len(self.ch_names), "times": self.X.shape[-1]}


def load(study="xu2025alljoined", data_dir="data"):
    """The packed study, X read into RAM."""
    folder = Path(data_dir) / "packed" / study
    meta = json.loads((folder / "meta.json").read_text())
    X = np.load(folder / "X.npy", mmap_mode="r")
    host = torch.empty(X.shape, dtype=torch.float16)
    for i in range(0, len(X), 65536):
        host[i:i + 65536] = torch.from_numpy(np.ascontiguousarray(X[i:i + 65536]))
    arrays = np.load(folder / "index.npz")
    return Packed(host, arrays["image"], arrays["subject"], arrays["test"], arrays["images"],
                  meta["ch_names"], meta["positions"])


class Windows(Dataset):
    """A prepared neuralset split as plain tensors (X, y, subject)."""

    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments.segments)

    def __getitem__(self, i):
        data = self.segments[int(i)].data  # neuralset accepts only a plain int
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


def _row_keys(target):
    """Exact integer hash of each row's bits: equal embeddings, equal keys."""
    bits = np.ascontiguousarray(target, np.float32).view(np.uint32).astype(np.uint64)
    weights = np.random.default_rng(0).integers(1, 2**63, size=bits.shape[1], dtype=np.uint64) | np.uint64(1)
    return (bits * weights).sum(1)


def standard_positions():
    """Electrode name -> xyz (metres) of the standard 10-05 montage."""
    import mne

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return {name: [float(v) for v in xyz] for name, xyz in pos.items()}


def pack(study="xu2025alljoined", data_dir="data", batch_size=512, num_workers=8):
    """Run the neuralbench pipeline over every split once and write ``data/packed/<study>``."""
    from neuralbench.data import Data

    data_dir = Path(data_dir).resolve()
    folder = data_dir / "packed" / study
    folder.mkdir(parents=True, exist_ok=True)
    splits = Data(**_task_config(study, data_dir)).prepare()
    datasets = {name: Windows(l.dataset) for name, l in splits.items()}
    n = sum(len(d) for d in datasets.values())
    C, T = datasets["test"][0][0].shape
    X = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float16, shape=(n, C, T))
    image, subject, test = np.empty(n, np.int32), np.empty(n, np.int32), np.zeros(n, bool)
    rows, images, start = {}, [], 0
    for name, dataset in datasets.items():
        for i, (x, y, s) in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)):
            stop = start + len(x)
            X[start:stop] = x.numpy().astype(np.float16)
            for j, key in enumerate(_row_keys(y.numpy())):
                if key not in rows:
                    rows[key] = len(images)
                    images.append(y[j].numpy())
                image[start + j] = rows[key]
            subject[start:stop], test[start:stop] = s.numpy(), name == "test"
            start = stop
            if i % 200 == 0:
                print(f"{name}: {start}/{n} windows packed")
    X.flush()
    channels = splits["test"].dataset.extractors["neuro"]._channels  # name -> row
    ch_names = sorted(channels, key=channels.get)
    positions = standard_positions()
    known = {k.upper() for k in positions}
    missing = [c for c in ch_names if c.upper() not in known]
    if missing:
        print(f"no standard position for {missing}: these channels will be ignored")
    np.savez(folder / "index.npz", image=image, subject=subject, test=test, images=np.stack(images))
    (folder / "meta.json").write_text(json.dumps({"study": study, "ch_names": ch_names, "positions": positions}))
    print(load(study, data_dir).summary())


def prepare(study="xu2025alljoined", data_dir="data"):
    """Download the study, run the pipeline once (fills the caches, incl. DINOv2) and pack it."""
    import neuralset as ns

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    source = dict(_task_config(study, data_dir)["study"]["source"])
    source["path"] = Path(source["path"]) / source["name"]
    from image_decoding import nemar_retry

    nemar_retry.patch()  # longer timeouts, resume, S3 copy of the manifest
    ns.Study(**source).download()
    pack(study, data_dir)


def synthetic(n_subjects=6, n_train=600, n_test=60, reps=4, test_reps=10, n_chans=32, n_times=120,
              dim=1536, noise=30.0, seed=0):
    """Tiny fake study in the packed format, to check the code without downloading.

    Images have low-rank random embeddings with a large common mean (like
    DINOv2). Sources sit at fixed scalp positions; every subject's EEG mixes
    them through the same head model, with its own gain and noise.
    """
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(n_train + n_test, 16)) @ rng.normal(size=(16, dim)) + 3.0
    wave = np.sin(np.linspace(0, 3 * np.pi, n_times))
    read = rng.normal(size=(dim, 8)) / np.sqrt(dim)
    positions = rng.normal(size=(n_chans, 3))
    positions = 0.09 * positions / np.linalg.norm(positions, axis=1, keepdims=True)
    sources = positions[rng.choice(n_chans, 8, replace=False)]
    mix = np.exp(-np.linalg.norm(positions[:, None] - sources[None], axis=-1) ** 2 / 0.05 ** 2)

    X, image, subject, test = [], [], [], []
    for s in range(n_subjects):
        gain = 1 + 0.3 * rng.normal(size=(8, 1))
        for ids, n_reps, is_test in ((np.arange(n_train), reps, False),
                                     (np.arange(n_train, n_train + n_test), test_reps, True)):
            for i in np.repeat(ids, n_reps):
                source = gain * (images[i] @ read)[:, None] * wave + noise * rng.normal(size=(8, n_times))
                X.append(mix @ source + noise * rng.normal(size=(n_chans, n_times)))
                image.append(i); subject.append(s); test.append(is_test)
    X = np.array(X)
    X /= X.std(axis=(1, 2), keepdims=True)  # unit scale, as the real pipeline's scaler
    names = [f"E{i}" for i in range(n_chans)]
    return Packed(torch.tensor(X, dtype=torch.float16), image, subject, test, images,
                  names, {n: [float(v) for v in p] for n, p in zip(names, positions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="xu2025alljoined", choices=sorted(STUDIES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pack-only", action="store_true", help="the study is downloaded already")
    args = parser.parse_args()
    (pack if args.pack_only else prepare)(args.study, args.data_dir)
