# Track 1: EEG → image

[Neural Interfaces 2026, Track 1](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html):
from one EEG window (−0.2…0.8 s around an image, 120 Hz) predict the image's DINOv2-giant embedding
(1536 numbers). Scored by top-5 cosine retrieval among unseen test images; the sealed test is
11 new subjects on 32-channel Emotiv hardware (as in Alljoined-1.6M).

Data (run from this folder; the DINOv2 pass over the stimuli wants a GPU):
`python -m image_decoding.data --study xu2024alljoined --data-dir data`

```
image_decoding/
  data.py          windows + DINOv2 targets via the official neuralbench pipeline; synthetic data for code checks
  evaluation.py    top-1 / top-5 as the platform computes them (per window and subject-aggregated)
  encoder.py       stage 1: EEG encoder with a learnable subject channel, trained with a CLIP loss
  prior.py         stage 2: flow-matching prior, a vector U-Net that refines the encoder's embedding
notebooks/
  clip_flow.ipynb  train both stages and compare on test
```

Studies (`--study`): `xu2024alljoined` (~5 GB, quick start), `xu2025alljoined` (same hardware as the
sealed test), `gifford2022large` (THINGS-EEG2, ~220 GB), `grootswagers2022human`.
Reference top-5 on THINGS-EEG2: chance 2.22, EEGNet 28.13, REVE frozen probe 84.75.
