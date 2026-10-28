"""PROTEUS data for Track 2 (BCI decoding), prepared once into .npz files.

The windows come from the official neuralbench pipeline (task
``eeg/_motor_imagery_stream``), so preprocessing and the subject split are
exactly the ones behind the published baselines:

- notch 50/60 Hz, band-pass 0.1-75 Hz, resampled to 120 Hz, microvolts;
- one 4 s window per cue, starting 1 s after it -> X has shape (41, 480);
- 3 classes; subjects split 60/20/20 into train/val/test (random_state=33).

Usage (once, ~1.5 GB download):

    python -m bci.data --data-dir data
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TASK = "_motor_imagery_stream"
SFREQ = 120.0

# PROTEUS run task -> interface; the interface is the "context" of a window
INTERFACES = {
    "AcquisitionBH": "BrainHero",
    "OnlineRawBH": "BrainHero",
    "AcquisitionGraz": "Graz",
    "OnlineRawGraz": "Graz",
}


@dataclass
class Windows:
    """All windows of one split, sorted in recording order within each stream."""

    X: np.ndarray        # (N, C, T) float32, microvolts
    y: np.ndarray        # (N,) class index
    subject: np.ndarray  # (N,) str
    session: np.ndarray  # (N,) str
    context: np.ndarray  # (N,) "Graz" | "BrainHero"
    run: np.ndarray      # (N,) run number within the session
    onset: np.ndarray    # (N,) window start in its recording, seconds
    ch_names: np.ndarray  # (C,) channel names

    @property
    def stream(self):
        """A stream is one session of one participant."""
        return np.char.add(np.char.add(self.subject, "/"), self.session)

    def streams(self):
        """Indices of each stream's windows, in recording order."""
        stream = self.stream
        for s in dict.fromkeys(stream):
            yield np.flatnonzero(stream == s)

    def __len__(self):
        return len(self.y)

    def select(self, mask):
        """Windows where ``mask`` is True (order kept)."""
        fields = {k: getattr(self, k)[mask] for k in self.__dataclass_fields__ if k != "ch_names"}
        return Windows(**fields, ch_names=self.ch_names)


def concat(*parts):
    """One Windows object from several (e.g. train + val)."""
    fields = {k: np.concatenate([getattr(p, k) for p in parts])
              for k in Windows.__dataclass_fields__ if k != "ch_names"}
    return Windows(**fields, ch_names=parts[0].ch_names)


def load_split(split, data_dir="data"):
    """Load ``train``, ``val`` or ``test`` windows prepared by :func:`prepare`."""
    with np.load(Path(data_dir) / f"proteus_{split}.npz") as f:
        return Windows(**{k: f[k] for k in f.files})


def _task_config(data_dir):
    """neuralbench's data config for the task, with all files under data_dir."""
    from neuralbench.experiment_config import merge_task_config

    cfg = merge_task_config("eeg", TASK)["data"]
    cfg.update({
        "study.source.path": str(data_dir),
        "study.source.infra.folder": str(data_dir / "cache"),
        "neuro.infra.cluster": None,
        "neuro.infra.folder": str(data_dir / "cache"),
        "neuro.infra.permissions": None,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    })
    return cfg


def prepare(data_dir="data"):
    """Download PROTEUS, run the official pipeline, save one .npz per split."""
    import neuralset as ns
    from neuralbench.data import Data

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = _task_config(data_dir)

    source = dict(cfg["study"]["source"])
    source["path"] = Path(source["path"]) / source["name"]
    from bci import nemar_retry

    nemar_retry.patch()  # longer timeouts, resume, S3 copy of the manifest
    ns.Study(**source).download()

    loaders = Data(**cfg).prepare()
    for split, loader in loaders.items():
        segments = loader.dataset
        rows = {k: [] for k in Windows.__dataclass_fields__ if k != "ch_names"}
        for i, seg in enumerate(segments.segments):
            trigger = seg.trigger.to_dict()
            data = segments[i].data
            x = np.asarray(data["neuro"], np.float32)
            rows["X"].append(x.reshape(-1, x.shape[-1]))  # (1, C, T) -> (C, T)
            rows["y"].append(int(np.argmax(np.asarray(data["target"]))))
            rows["subject"].append(str(trigger["subject"]))
            rows["session"].append(str(trigger["session"]))
            rows["context"].append(INTERFACES[trigger["task"]])
            rows["run"].append(int(trigger["run"]))
            rows["onset"].append(float(seg.start))
        rows = {k: np.stack(v) for k, v in rows.items()}

        # recording order inside each stream: subject, session, run, time
        order = np.lexsort((rows["onset"], rows["run"], rows["session"], rows["subject"]))
        rows = {k: v[order] for k, v in rows.items()}
        channels = segments.extractors["neuro"]._channels  # name -> column
        rows["ch_names"] = np.array(sorted(channels, key=channels.get))
        np.savez(data_dir / f"proteus_{split}.npz", **rows)
        print(f"{split}: {len(order)} windows, {len(set(rows['subject']))} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    prepare(parser.parse_args().data_dir)
