#!/usr/bin/env python3
"""Единая оценка каскада. Только числа из этого скрипта идут в results/leaderboard.md.

Каскад:  картинка -> [DDPM] -> [Upscale] -> OCR (crnn_vgg16_bn) -> [LLM] -> текст

Примеры (из корня репозитория):
    # baseline, без улучшений
    python -m OCR.evaluate --images data/b-mod_lines/lines --answers data/b-mod_lines/test.easy \
        --weights OCR/crnn_vgg16_bn/checkpoints/<run>/best.pt

    # только своя технология
    python -m OCR.evaluate ... --enhancer DDPM
    python -m OCR.evaluate ... --enhancer Upscale
    python -m OCR.evaluate ... --corrector LLM

    # весь каскад (порядок улучшателей = порядок флагов)
    python -m OCR.evaluate ... --enhancer DDPM --enhancer Upscale --corrector LLM

Контракт модулей (см. README в корне):
    DDPM/enhance.py, Upscale/enhance.py:  enhance(image: np.ndarray) -> np.ndarray
                                          (опционально enhance_batch(list) -> list)
    LLM/correct.py:                       correct(text: str) -> str
                                          (опционально correct_batch(list) -> list)
"""

import argparse
import importlib
import json
import sys
import time

import torch

from .crnn_vgg16_bn.predictor import Recognizer
from .dataset import load_samples
from .metrics import Scorer


def _load_stage(module: str, func: str):
    """Возвращает функцию, работающую над списком: batch-версию, если она есть."""
    mod = importlib.import_module(f"{module}.{'enhance' if func == 'enhance' else 'correct'}")
    batch = getattr(mod, f"{func}_batch", None)
    if batch is not None:
        return batch
    single = getattr(mod, func)
    return lambda items: [single(x) for x in items]


def run(args: argparse.Namespace) -> dict:
    samples = load_samples(args.images, args.answers, args.split_file, args.limit)
    if not samples:
        sys.exit("error: no samples to evaluate")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")

    enhancers = [(m, _load_stage(m, "enhance")) for m in args.enhancer]
    correctors = [(m, _load_stage(m, "correct")) for m in args.corrector]
    recognizer = Recognizer(weights_path=args.weights, device=args.device,
                            batch_size=args.batch_size)

    scorer = Scorer(keep_spaces=args.keep_spaces)
    t0 = time.perf_counter()
    for i in range(0, len(samples), args.chunk_size):
        chunk = samples[i:i + args.chunk_size]
        images = [s.image() for s in chunk]
        for _, enh in enhancers:
            images = enh(images)
        texts = [t for t, _ in recognizer(images)]
        for _, corr in correctors:
            texts = corr(texts)
        for s, t in zip(chunk, texts):
            scorer.add(s.name, s.gt, t)
        if args.progress:
            print(f"[{scorer.n}/{len(samples)}] CER {scorer.cer:.4f} WER {scorer.wer:.4f}",
                  file=sys.stderr)

    summary = {
        "pipeline": [m for m, _ in enhancers] + ["OCR"] + [m for m, _ in correctors],
        **scorer.summary(),
        "time_s": round(time.perf_counter() - t0, 2),
        "answers": args.answers,
        "split_file": args.split_file,
        "weights": args.weights,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for name, c, gt, pred in scorer.rows:
                f.write(f"{name}\t{c:.6f}\t{gt}\t{pred}\n")
        print(f"per-sample details written to {args.output}", file=sys.stderr)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="CER/WER of the OCR cascade on b-mod")
    p.add_argument("--images", required=True, help="папка с кропами строк (b-mod_lines/lines)")
    p.add_argument("--answers", required=True, help="файл '<filename> <transcript>'")
    p.add_argument("--split-file", default=None, help="список имён, ограничивающий выборку")
    p.add_argument("--weights", default=None, help="обученный чекпоинт OCR (best.pt)")
    p.add_argument("--enhancer", action="append", default=[], choices=["DDPM", "Upscale"],
                   help="улучшатель картинки; можно указать несколько раз")
    p.add_argument("--corrector", action="append", default=[], choices=["LLM"],
                   help="корректор текста после OCR")
    p.add_argument("--device", default=None, help="cpu | cuda (по умолчанию авто)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="взять первые N строк (быстрая проверка)")
    p.add_argument("--keep-spaces", action="store_true", help="считать CER с пробелами")
    p.add_argument("--output", default=None, help="TSV с результатом по каждой строке")
    p.add_argument("--progress", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
