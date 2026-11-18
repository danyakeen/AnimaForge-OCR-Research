"""Загрузка датасета b-mod в одном формате для всех.

Пример:
    from OCR.dataset import load_samples
    samples = load_samples("data/b-mod_lines/lines", "data/b-mod_lines/test.easy")
    for s in samples:
        img = s.image()   # H x W x 3 uint8 RGB
        s.gt              # эталонный текст
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

from .crnn_vgg16_bn.benchmark import load_answers, load_split_file, resolve_images
from .crnn_vgg16_bn.predictor import load_image


@dataclass
class Sample:
    name: str
    path: str
    gt: str

    def image(self) -> np.ndarray:
        return load_image(self.path)


def load_samples(images_dir: str, answers: str, split_file: str | None = None,
                 limit: int | None = None) -> list[Sample]:
    gt = load_answers(answers)
    names = list(gt)
    if split_file:
        names = [n for n in load_split_file(split_file) if n in gt]
    if limit:
        names = names[:limit]
    paths, missing, _ = resolve_images(images_dir, names)
    if missing:
        print(f"warning: {len(missing)} entries have no image file (skipped)", file=sys.stderr)
    return [Sample(os.path.basename(p), p, gt[os.path.basename(p)]) for p in paths]
