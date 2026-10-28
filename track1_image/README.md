# Track 1: EEG → image

```
python -m image_decoding.data --study xu2025alljoined --data-dir data   # данные, один раз
python -m image_decoding.make_submission --study xu2025alljoined        # обучение + submissions/clip_flow.zip
```
Ноутбук: `notebooks/clip_flow.ipynb`.
