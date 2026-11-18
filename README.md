# AnimaForge OCR Research

Проект СНО: повышение точности OCR тремя технологиями — **DDPM**, **upscale** и **языковые модели**.
Одна OCR-модель и один датасет на все три исследования; итоговый каскад доступен через сайт.

```
картинка → [DDPM] → [Upscale] → OCR (CRNN VGG16-BN) → [LLM] → текст
```

- **Модель:** CRNN + VGG16-BN (порт из docTR) — `OCR/crnn_vgg16_bn/`
- **Датасет:** b-mod (кропы строк) — см. `data/README.md`
- **Метрики:** CER и WER по всему корпусу — `OCR/metrics.py`

## Кто за что

| Папка | Что внутри | Ответственный |
|---|---|---|
| `OCR/` | базовая модель, датасет, метрики, единая оценка | общее |
| `DDPM/` | очистка/восстановление изображения | Михаил |
| `Upscale/` | повышение разрешения изображения | Макс |
| `LLM/` | исправление распознанного текста | Данил |
| `Backend/` | сервер сайта | Хамди |
| `Frontend/` | интерфейс сайта | Хамди |
| `data/` | датасет (локально, не в git) + зафиксированные сплиты | общее |
| `results/` | общая таблица результатов | все |

## Структура

```
AnimaForge-OCR-Research/
├── OCR/
│   ├── crnn_vgg16_bn/     # модель: обучение, инференс, бенчмарк (подробности в её README)
│   ├── dataset.py         # загрузка b-mod
│   ├── metrics.py         # CER / WER
│   └── evaluate.py        # оценка каскада — единственный источник чисел
├── DDPM/  Upscale/  LLM/  # исследования, одинаковый шаблон:
│   ├── enhance.py / correct.py   # точка входа для каскада
│   ├── src/  scripts/  configs/  notebooks/  experiments/
│   ├── requirements.txt   # свои доп. зависимости
│   └── README.md          # идея, запуск, лучший результат, журнал экспериментов
├── Backend/  Frontend/
├── data/
│   ├── README.md
│   └── splits/
├── results/leaderboard.md
└── requirements.txt       # общие зависимости
```

## Контракт между частями

Чтобы все мерили одинаково, а бэкенд мог собрать каскад, каждый модуль даёт одну функцию:

| Модуль | Файл | Функция |
|---|---|---|
| DDPM | `DDPM/enhance.py` | `enhance(image: np.ndarray) -> np.ndarray` |
| Upscale | `Upscale/enhance.py` | `enhance(image: np.ndarray) -> np.ndarray` |
| LLM | `LLM/correct.py` | `correct(text: str) -> str` |

Картинка — `H x W x 3`, `uint8`, RGB. Для обработки пачкой можно добавить
`enhance_batch(list) -> list` / `correct_batch(list) -> list` — `evaluate.py` подхватит её сам.
Сигнатуры не меняем без договорённости команды.

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate — Linux/macOS
pip install -r requirements.txt

# baseline
python -m OCR.evaluate --images data/b-mod_lines/lines --answers data/b-mod_lines/test.easy \
    --weights <чекпоинт OCR>

# со своей технологией
python -m OCR.evaluate ... --enhancer DDPM
python -m OCR.evaluate ... --enhancer Upscale
python -m OCR.evaluate ... --corrector LLM
```

Все команды запускаются из корня репозитория.

## Правила

1. **Один тест на всех.** Тестовый сплит в `data/splits/` и чекпоинт OCR меняются только по договорённости.
   На тестовой части ничего не обучаем.
2. **Числа — только из `OCR/evaluate.py`.** Результат записываешь в `results/leaderboard.md` с коммитом.
3. **Не храним в git** данные, веса и чекпоинты (`.pt`, `.pth`, `.ckpt`, …) — они в `.gitignore`.
   Обученные веса кладём в общее хранилище: _указать где_.
4. **Работаешь в своей папке.** Изменения в `OCR/` и чужих папках — через PR с ревью.
5. **Ветки:** `main` — рабочая версия; свои ветки называем `ddpm/...`, `upscale/...`, `llm/...`,
   `backend/...`, `frontend/...`; в `main` — через pull request.
6. **Зависимости:** общие — в корневом `requirements.txt`, свои — в `requirements.txt` своей папки.
