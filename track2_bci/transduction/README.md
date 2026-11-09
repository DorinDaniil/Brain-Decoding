# Трансдукция внутри сессии

Отдельная реализация (только numpy + scipy) шага, который в сабмите добавлен поверх ансамбля.

```
python demo.py           # игрушечные сессии, данные не нужны
python check_parity.py   # совпадает ли с bci/stream.py, который уходит в сабмит
```

- `method.py` — `TangentFeatures` (окно ЭЭГ → признаки) и `Transducer` (признаки + вероятности любой модели → уточнённые вероятности).
- В сабмите: вариант `all+transduction` (`python -m bci.make_submission`, `python -m bci.benchmark --variants all all+transduction`).
