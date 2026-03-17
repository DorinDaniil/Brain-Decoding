### ECG-from-EEG with music (baseline ARX)

#### Dataset

We use OpenNeuro `ds004840` (BIDS): *EEG, ECG, EMG during music therapy with adult burn patients in ICU*.

- Source: [OpenNeuro dataset page](https://openneuro.org/datasets/ds004840/versions/1.0.1)
- DOI: `10.18112/openneuro.ds004840.v1.0.1`

Download:

```bash
python scripts/download_ds004840.py --out_dir data/ds004840 --version 1.0.1
```

#### Goal / hypothesis

Patient-specific modelling: predict short-horizon **ECG dynamics** from **EEG features** during music listening/therapy.

Baseline hypothesis: EEG contains information predictive of ECG changes (autonomic response) during music exposure.

#### Baseline model (your proposal)

Discrete-time ARX on ECG:

\[
\Delta y_{t+1} = y_{t+1}-y_t = X_t W + b
\]

Where:

- \(y_t\): ECG-derived 1D target (see “Target design”)
- \(X_t\): EEG feature vector at time \(t\) (optionally with lags)
- \(W,b\): linear parameters (ridge regression recommended)

#### Key design choices (to implement)

- **Target design** (choose one):
  - ECG amplitude (after filtering), or
  - instantaneous heart rate / RR interval from R-peaks, or
  - HRV features in sliding windows (slower, but often more stable)
- **EEG preprocessing**:
  - bandpass + notch, artifact handling, re-reference
  - feature extraction (bandpower per channel/band; or PCA/ICA components)
- **Alignment (EEG vs ECG)**:
  - handle different sampling rates via resampling to a common time base
  - estimate time lag via cross-correlation (or maximize predictive likelihood)
  - optionally model lag explicitly (include multiple EEG lags in \(X_t\))
- **Splits**:
  - within-subject chronological split (train early, test later)
  - report per-subject metrics and aggregate
- **Metrics**:
  - MSE/MAE on \(\Delta y\) and on reconstructed \(y\)
  - correlation \(r\), and baseline vs null (shuffle EEG time / circular shift)

#### “Bring music in” (ideas)

- add **audio-derived features** \(A_t\) (tempo, RMS energy, spectral flux, chroma, etc.)
- model ECG as function of both EEG and audio:
  - \(\Delta y_{t+1} = [X_t, A_t] W + b\)
- test mediation:
  - audio \(\rightarrow\) EEG \(\rightarrow\) ECG (partial out audio, compare predictive gains)
- stratify by music segments or conditions (if annotated in events files)

