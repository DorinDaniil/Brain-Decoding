# Brain Decoding

Research on Riemannian approaches to EEG decoding, built around the
[Neural Interfaces 2026](https://neural-interfaces26.github.io/) challenge (NeurIPS 2026).
One environment for everything: see `SETUP.txt`.

**Riemannian forest.** An ensemble of tangent-space classifiers for cross-subject decoding.
Each tree takes a random frequency band, a random subset of channels (a principal submatrix
of the covariance matrix) and a bootstrap sample of training subjects, recenters the
covariances of each session causally, and fits a logistic regression on tangent-space
features. The forest averages the trees' probabilities. See [`track2_bci/`](track2_bci/).

**[Knowledge graph](https://dorindaniil.github.io/Brain-Decoding/knowledge-graph/)**:
an interactive map of the related literature (Riemannian BCI, transfer across subjects,
ensemble theory, the works of A. Barachant). Source: [`docs/`](docs/).

| Folder | Project |
|---|---|
| [`track1_image/`](track1_image/) | Track 1, EEG → image: CLIP encoder + flow-matching prior |
| [`track2_bci/`](track2_bci/) | Track 2, BCI decoding: Riemannian models, Codabench packaging |
| [`docs/`](docs/) | GitHub Pages site with the knowledge graph |
| `extras/` | older notebooks, not part of the challenge |

Each project is self-contained: run its commands from its folder; its data goes to `<project>/data/`.

## Extras

| Notebook | Topic |
|---|---|
| `extras/brain_decoding.ipynb` | fMRI decoding of object categories (Haxby) with an MLP |
| `extras/encoding.ipynb` | fMRI encoding of visual stimuli (Miyawaki) with ridge regression |
| `extras/audio.ipynb` | Environmental sound classification (ESC-50) with a CNN |
