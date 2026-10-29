"""Track 3 (time to sleep onset) data through the official neuralbench pipeline, saved once to .npz.

Muse home EEG: TP9, AF7, AF8, TP10 at 128 Hz, microvolts, no filtering.
Each recording is cut into contiguous 5 s windows from its start to the first
N2 epoch; the target of a window is the seconds left to that epoch, capped at
600. The true (uncapped) time to onset is recovered by counting windows back
from the last one. Split: the release's train / test (the warm-up evaluation
set), with part of train held out as validation by neuralbench.

    python -m sleep_onset.data --data-dir data      # about 1.2 GB
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TASK = ("eeg", "_sleep_onset_stream")
SFREQ = 128.0
HOP = 5.0  # seconds per window


@dataclass
class Recording:
    X: np.ndarray       # (n, C, T) float32, microvolts
    y: np.ndarray       # (n,) seconds to onset, capped at 600 (the scored target)
    subject: str
    name: str

    @property
    def tau(self):
        """Uncapped seconds to onset of each window: the last window's target plus 5 s per window after it."""
        return self.y[-1] + HOP * np.arange(len(self.y) - 1, -1, -1)


def load_split(split, data_dir="data"):
    """List of Recording for ``train``, ``val`` or ``test``, and the channel names."""
    with np.load(Path(data_dir) / f"muse_{split}.npz") as f:
        X, y, rec, subject, names, ch_names = (f[k] for k in ("X", "y", "recording", "subject", "names", "ch_names"))
    recordings = [Recording(X[rec == r], y[rec == r], str(subject[rec == r][0]), str(names[r]))
                  for r in range(len(names))]
    return recordings, [str(c) for c in ch_names]


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
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    })
    return cfg


def prepare(data_dir="data"):
    """Download the Muse release, run the official pipeline, save one .npz per split."""
    import neuralset as ns
    from neuralbench.data import Data

    from sleep_onset import nemar_retry

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = _task_config(data_dir)
    source = dict(cfg["study"]["source"])
    source["path"] = Path(source["path"]) / source["name"]
    nemar_retry.patch()  # longer timeouts, resume, S3 copy of the manifest
    ns.Study(**source).download()

    for split, loader in Data(**cfg).prepare().items():
        segments = loader.dataset
        rows = {"X": [], "y": [], "key": [], "subject": [], "start": []}
        for i, seg in enumerate(segments.segments):
            trigger = seg.trigger.to_dict()
            data = segments[i].data
            x = np.asarray(data["neuro"], np.float32)
            rows["X"].append(x.reshape(-1, x.shape[-1]))
            rows["y"].append(float(np.asarray(data["target"]).reshape(-1)[0]))
            key = "/".join(str(trigger.get(k, "")) for k in ("subject", "session", "task", "run", "filepath"))
            rows["key"].append(key)
            rows["subject"].append(str(trigger.get("subject", "")))
            rows["start"].append(float(seg.start))
        key, start = np.array(rows["key"]), np.array(rows["start"])
        order = np.lexsort((start, key))  # recordings, windows in time order
        names, recording = np.unique(key[order], return_inverse=True)
        channels = segments.extractors["neuro"]._channels
        np.savez(data_dir / f"muse_{split}.npz", X=np.stack(rows["X"])[order], y=np.array(rows["y"])[order],
                 recording=recording, subject=np.array(rows["subject"])[order], names=names,
                 ch_names=np.array(sorted(channels, key=channels.get)))
        print(f"{split}: {len(order)} windows, {len(names)} recordings")


def synthetic(n_train=60, n_val=15, n_test=15, seed=0):
    """Fake recordings in the same format, to check the code without downloading.

    Awake alpha (10 Hz) fades and theta (6 Hz) grows over the last few minutes
    before onset; slow eye movements appear on AF7 - AF8; every sleeper has its
    own amplitude.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(HOP * SFREQ)) / SFREQ

    def recording(i):
        n = int(np.clip(rng.lognormal(np.log(150), 0.5), 30, 600))  # windows, about 12 min
        gain = np.exp(rng.normal(0, 0.3, 4))[:, None]
        drowsy_len = rng.uniform(30, 90)  # windows of transition before onset
        X, y = [], []
        for k in range(n):
            remaining = HOP * (n - 1 - k) + rng.uniform(0, HOP)
            s = 1 / (1 + np.exp((remaining / HOP - drowsy_len / 2) / (drowsy_len / 6)))  # 0 awake -> 1 asleep
            alpha = (1 - s) * 12 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.3))
            theta = s * 10 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.3))
            x = gain * (alpha + theta + 8 * rng.normal(size=(4, len(t))))
            x[1] += s * 30 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.3))  # slow eye movement
            X.append(x.astype(np.float32)); y.append(min(remaining, 600.0))
        return Recording(np.stack(X), np.array(y), f"s{i % 20}", f"rec{i}")

    split = {"train": range(n_train), "val": range(n_train, n_train + n_val),
             "test": range(n_train + n_val, n_train + n_val + n_test)}
    return {name: [recording(i) for i in ids] for name, ids in split.items()}, ["TP9", "AF7", "AF8", "TP10"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    prepare(parser.parse_args().data_dir)
