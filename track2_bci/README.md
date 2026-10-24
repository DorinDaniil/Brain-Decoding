# Track 2: BCI decoding (PROTEUS)

[Neural Interfaces 2026, Track 2](https://neural-interfaces26.github.io/tracks.html): classify 4 s EEG
windows (41 channels, 120 Hz) into one of three mental commands: motor imagery, mental subtraction,
word generation. Evaluation is streamed, one window at a time, a fresh model per session.

Data (run from this folder): `python -m bci.data --data-dir data`

```
bci/
  data.py          windows via the official neuralbench pipeline (same preprocessing and split as the baselines)
  evaluation.py    platform-style scoring + cross-validation (new subjects, later sessions)
  riemann.py       shared features: covariances, causal session recentering, session fingerprint
  models/
    logreg.py      Riemann + LogReg
    gated.py       Riemann + manifold gate (subject embedding mixed over nearest subjects)
    forest.py      Riemannian random forest: random band + channel subset + subject bootstrap per tree
  export.py        trained model -> submissions/<name>.zip, and loading it back as the platform does
  official.py      optional: the same scores computed by the competition's own code
submission/
  submission.py    self-contained inference file that goes into the zip (logreg, forest)
notebooks/
  bci_data.ipynb   what the data looks like
  logreg.ipynb     one notebook per model: cross-validation + official test vs baselines
  gated.ipynb
  forest.ipynb
  submit.ipynb     train on all public data, export the zip, check it
```

`bci.evaluation` reproduces the competition's scoring exactly (checked on real PROTEUS against
their `Objective` and `benchopt`). `tests/` re-check it when their repo is cloned next to this one.

New hypothesis: copy a file in `bci/models/` and its notebook, change the model, compare the
cross-validation tables. Use the official test only to compare with the published baselines
(`session_mean`, chance = 0.333): EEGNet 0.326, CTNet 0.350, REVE 0.360.
