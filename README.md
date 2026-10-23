# Brain Decoding

Notebooks on decoding (and encoding) brain signals, plus a starting point for
[Neural Interfaces 2026, Track 2: BCI decoding](https://neural-interfaces26.github.io/tracks.html).

## Track 2: BCI decoding (PROTEUS)

Classify 4 s EEG windows (41 channels, 120 Hz) into one of three mental commands:
motor imagery, mental subtraction, word generation. Test subjects are never seen in training.

Setup: see `SETUP.txt` (Python 3.12, then `python -m bci.data --data-dir data`).

```
bci/
  data.py          windows via the official neuralbench pipeline (same preprocessing and split as the baselines)
  evaluation.py    platform-style scoring + cross-validation (new subjects, later sessions)
  riemann.py       shared features: covariances, causal session recentering, session fingerprint
  models/
    logreg.py      Riemann + LogReg
    gated.py       Riemann + manifold gate (subject embedding mixed over nearest subjects)
    fbstack.py     filter-bank ensemble: one Riemannian expert per band, weighted geometric mean
    forest.py      Riemannian random forest: random band + channel subset + subject bootstrap per tree
notebooks/
  bci_data.ipynb   what the data looks like
  logreg.ipynb     one notebook per model: cross-validation + official test vs baselines
  gated.ipynb
  fbstack.ipynb
  forest.ipynb
```

New hypothesis: copy a file in `bci/models/` and its notebook, change the model, compare the
cross-validation tables. Use the official test only to compare with the published baselines
(`session_mean`, chance = 0.333): EEGNet 0.326, CTNet 0.350, REVE 0.360.

## Extras

Older side notebooks, not part of Track 2.

| Notebook | Topic |
|---|---|
| `extras/brain_decoding.ipynb` | fMRI decoding of object categories (Haxby) with an MLP |
| `extras/encoding.ipynb` | fMRI encoding of visual stimuli (Miyawaki) with ridge regression |
| `extras/audio.ipynb` | Environmental sound classification (ESC-50) with a CNN |
