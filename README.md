# Brain Decoding

Two projects for the [Neural Interfaces 2026](https://neural-interfaces26.github.io/) challenge
(NeurIPS 2026), plus older side notebooks. One environment for everything: see `SETUP.txt`.

| Folder | Project |
|---|---|
| [`track1_image/`](track1_image/) | Track 1, EEG → image: CLIP encoder + flow-matching prior |
| [`track2_bci/`](track2_bci/) | Track 2, BCI decoding: Riemannian models, Codabench packaging |
| `extras/` | older notebooks, not part of the challenge |

Each project is self-contained: run its commands from its folder; its data goes to `<project>/data/`.

## Extras

| Notebook | Topic |
|---|---|
| `extras/brain_decoding.ipynb` | fMRI decoding of object categories (Haxby) with an MLP |
| `extras/encoding.ipynb` | fMRI encoding of visual stimuli (Miyawaki) with ridge regression |
| `extras/audio.ipynb` | Environmental sound classification (ESC-50) with a CNN |
