# LLM

**Ответственный:** Данил

## Идея
_Как именно LLM должен повысить точность OCR. Заполнить._

## Структура
```
LLM/
├── correct.py        # точка входа для каскада (контракт — в корневом README)
├── src/              # код модели
├── scripts/          # обучение / инференс
├── configs/          # конфиги экспериментов
├── notebooks/        # исследования, черновики
├── experiments/      # логи и метрики (в git); чекпоинты — НЕ в git
└── requirements.txt  # только свои доп. зависимости
```

## Как запустить
```bash
pip install -r requirements.txt -r LLM/requirements.txt
python -m OCR.evaluate --images data/b-mod_lines/lines --answers <test-файл> \
    --weights <чекпоинт OCR> --corrector LLM
```

## Лучший результат
| CER | WER | Δ CER к baseline | Конфиг | Коммит |
|---|---|---|---|---|
| — | — | — | — | — |

## Журнал экспериментов
| Дата | Что пробовал | CER | WER | Вывод |
|---|---|---|---|---|
