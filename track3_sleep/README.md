# Track 3: sleep onset

```
python -m sleep_onset.data --data-dir data     # данные, один раз (~1.2 ГБ)
python -m sleep_onset.make_submission          # обучение + submissions/bayes_onset.zip
python -m sleep_onset.make_submission --synthetic   # проверка кода без данных
```
Ноутбук: `notebooks/bayes_onset.ipynb`.
