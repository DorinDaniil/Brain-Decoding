### Experiments

This folder is for reproducible experiment runs and notebooks.

Suggested structure per experiment:

- `experiments/<exp_name>/README.md`: hypothesis + design + data splits + metrics
- `experiments/<exp_name>/run.py`: entrypoint to run training/eval
- `experiments/<exp_name>/results/`: metrics, figures (keep small; large artifacts should be ignored)

Current experiments:

- `ecg_eeg_music_arx`: baseline idea (predict ECG from EEG, patient-specific).

