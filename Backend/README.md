# Backend

**Ответственный:** Хамди

Сайт, через который пользователь работает с каскадом OCR.
Стек и структуру папок выбирает ответственный — опиши их здесь.

## Как каскад подключается к бэкенду
Бэкенд вызывает модули через те же функции, что и `OCR/evaluate.py`:
`DDPM.enhance.enhance` → `Upscale.enhance.enhance` → `OCR.crnn_vgg16_bn.predictor.Recognizer` → `LLM.correct.correct`.

## Как запустить
_Заполнить._
