# Brain Decoding

Notebooks on decoding (and encoding) brain signals, plus a starting point for
[Neural Interfaces 2026, Track 2: BCI decoding](https://neural-interfaces26.github.io/tracks.html).

## Track 2: BCI decoding (PROTEUS)

Classify 4 s EEG windows (41 channels, 120 Hz) into one of three mental commands:
motor imagery, mental subtraction, word generation. Test subjects are never seen in training.

```bash
pip install -r requirements.txt     # Python >= 3.12
python -m bci.data --data-dir data  # download PROTEUS (~1.5 GB), write data/proteus_{train,val,test}.npz
```

| File | What it does |
|---|---|
| `bci/data.py` | Builds the windows with the official neuralbench pipeline (same preprocessing and subject split as the published baselines) |
| `bci/evaluation.py` | Scores a model the way the platform does (one window at a time, a fresh model copy per session); leave-one-subject-out and later-sessions cross-validation |
| `bci/riemann.py` | Riemannian features: covariances, causal session recentering, session fingerprint on the manifold |
| `notebooks/bci_data.ipynb` | What the data looks like: sessions, interfaces, classes, spectra, subject shift |
| `notebooks/bci_riemann.ipynb` | Riemannian model whose learnable subject embedding is gated by the nearest training subjects on the manifold |

Published baselines (`session_mean`, chance = 0.333): EEGNet 0.326, CTNet 0.350, REVE 0.360.

## Extras

Older side notebooks, not part of Track 2.

| Notebook | Topic |
|---|---|
| `extras/brain_decoding.ipynb` | fMRI decoding of object categories (Haxby) with an MLP |
| `extras/encoding.ipynb` | fMRI encoding of visual stimuli (Miyawaki) with ridge regression |
| `extras/audio.ipynb` | Environmental sound classification (ESC-50) with a CNN |
