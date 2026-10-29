# Track 1: EEG → image

```
python -m image_decoding.data --study xu2025alljoined --data-dir data               # скачать и упаковать, один раз
python -m image_decoding.data --study xu2025alljoined --data-dir data --pack-only   # уже скачано: только упаковать
python -m image_decoding.make_submission --study xu2025alljoined                    # обучение + submissions/spatial_conv.zip
python -m image_decoding.make_submission --synthetic                                # проверка кода без данных
```
Ноутбук: `notebooks/spatial_conv.ipynb`.
