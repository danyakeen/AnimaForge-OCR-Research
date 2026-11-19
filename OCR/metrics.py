"""Единые метрики CER / WER для всех трёх исследований.

Расчёт тот же, что в базовом бенчмарке (OCR/crnn_vgg16_bn/benchmark.py),
поэтому числа сравнимы с baseline один в один:

    CER = sum(Levenshtein(pred, gt)) / sum(len(gt))        (по символам)
    WER = sum(Levenshtein(pred_words, gt_words)) / sum(len(gt_words))

CER по умолчанию считается БЕЗ пробелов (в словаре модели нет пробела),
WER — всегда по исходным строкам.
"""

from dataclasses import dataclass, field

from .crnn_vgg16_bn.benchmark import _str_distance, _token_distance, normalize


@dataclass
class Scorer:
    """Накопитель метрик по всему корпусу (не среднее по строкам!)."""

    keep_spaces: bool = False
    char_dist: int = 0
    gt_chars: int = 0
    word_dist: int = 0
    gt_words: int = 0
    exact: int = 0
    n: int = 0
    rows: list = field(default_factory=list)

    def add(self, name: str, gt: str, pred: str) -> None:
        p, g = normalize(pred, self.keep_spaces), normalize(gt, self.keep_spaces)
        d = _str_distance(p, g)
        self.char_dist += d
        self.gt_chars += len(g)
        gw = gt.split()
        self.word_dist += _token_distance(pred.split(), gw)
        self.gt_words += len(gw)
        self.exact += int(p == g)
        self.n += 1
        self.rows.append((name, d / max(len(g), 1), gt, pred))

    @property
    def cer(self) -> float:
        return self.char_dist / max(self.gt_chars, 1)

    @property
    def wer(self) -> float:
        return self.word_dist / max(self.gt_words, 1)

    def summary(self) -> dict:
        return {
            "samples": self.n,
            "CER": round(self.cer, 6),
            "WER": round(self.wer, 6),
            "exact_match_acc": round(self.exact / max(self.n, 1), 6),
            "chars": self.gt_chars,
            "words": self.gt_words,
        }


def cer(preds: list[str], gts: list[str], keep_spaces: bool = False) -> float:
    s = Scorer(keep_spaces=keep_spaces)
    for i, (p, g) in enumerate(zip(preds, gts)):
        s.add(str(i), g, p)
    return s.cer


def wer(preds: list[str], gts: list[str]) -> float:
    s = Scorer()
    for i, (p, g) in enumerate(zip(preds, gts)):
        s.add(str(i), g, p)
    return s.wer
