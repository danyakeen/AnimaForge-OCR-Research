"""Точка входа Upscale для каскада. Контракт общий для всех улучшателей картинки.

OCR/evaluate.py вызывает:  enhance(image) -> image
Если удобнее обрабатывать пачкой (GPU), можно добавить enhance_batch(images) -> images,
тогда evaluate.py будет вызывать её.

Вход и выход: np.ndarray формы H x W x 3, dtype uint8, RGB.
Размер выхода может отличаться от входа (OCR сам приведёт к 32 x 128).
"""

import numpy as np


def enhance(image: np.ndarray) -> np.ndarray:
    # TODO: заглушка — пока возвращает картинку без изменений
    return image
