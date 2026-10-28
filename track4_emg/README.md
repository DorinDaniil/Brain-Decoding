# Track 4: EMG → pose

```
python -m emg_pose.data --data-dir data                  # данные, один раз (~330 ГБ + ~440 ГБ кэша)
python -m emg_pose.make_submission --seeds 3 --epochs 40 # обучение + submissions/posenet.zip
python -m emg_pose.make_submission --synthetic           # проверка кода без данных
```
