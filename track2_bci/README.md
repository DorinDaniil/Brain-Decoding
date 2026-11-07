# Track 2: BCI decoding

```
python -m bci.data --data-dir data                  # данные, один раз
python -m bci.make_submission --data-dir data       # обучение на всех данных + submissions/all.zip
python -m bci.benchmark --data-dir data --jobs 4    # кросс-валидация вариантов (новые люди / поздние сессии)
```
Ноутбуки: `notebooks/`.
