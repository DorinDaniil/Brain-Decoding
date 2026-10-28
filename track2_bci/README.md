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
  export.py        trained model -> submissions/<name>.zip, loading it back, contract check of a zip
  make_submission.py  one command: train on all public data, build the zip, check it
submission/
  submission.py    self-contained inference file that goes into the zip (logreg, forest), numpy/scipy/sklearn only
notebooks/
  bci_data.ipynb   what the data looks like
  logreg.ipynb     one notebook per model: cross-validation + official test vs baselines
  gated.ipynb
  forest.ipynb
  submit.ipynb     the same as make_submission, step by step
```

`bci.evaluation` reproduces the competition's scoring exactly (checked on real PROTEUS against
their `Objective` and `benchopt`). The leaderboard ranks by `cell_mean`: balanced accuracy per
subject x session x interface, averaged.

New hypothesis: copy a file in `bci/models/` and its notebook, change the model, compare the
cross-validation tables. Use the official test only to compare with the published baselines
(`session_mean`, chance = 0.333): EEGNet 0.326, CTNet 0.350, REVE 0.360.

## Submit

```
python -m bci.make_submission                 # forest, 50 trees, 20 channels
python -m bci.make_submission --model logreg
```

It trains on all ten public subjects, writes `submissions/<name>.zip` (`submission.py` + `model.npz`
at the root), checks that the packaged model predicts exactly like the trained one, that it only
imports what the Codabench worker has, and that it is fast enough. Upload the zip in
**My Submissions**, phase **Warm-up Phase (update 1)** until Nov 1, **Sealed Final Phase** from Nov 5.

- Warm-up scores near 1.0 are expected and meaningless: the warm-up test is the public test
  participants, whose labels are public and which the final model is trained on. Use the warm-up
  only to check that a zip runs; compare models with the cross-validation in the notebooks.
- Sealed phase: 1 submission per day, 5 in total, logs hidden. Submit only zips that passed
  `make_submission` and ran in the warm-up.
