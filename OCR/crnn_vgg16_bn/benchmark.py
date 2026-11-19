#!/usr/bin/env python3
# Benchmark of the standalone pretrained docTR `crnn_vgg16_bn` model on the b-mod dataset.
#
# Computes CER (Character Error Rate) and WER (Word Error Rate) between the model's
# predictions and the ground-truth transcripts.
#
# Usage:
#   python -m OCR.crnn_vgg16_bn.benchmark --images <dir> --answers <file> [options]
#
# Arguments:
#   --images    path to the directory containing the crop images
#   --answers   answers file with "<filename> <transcript>" per line (b-mod format,
#               e.g. b-mod_lines/train.easy | train.medium | train.hard)
#   --split-file  optional list of filenames (one per line, e.g. b-mod_lines/test.easy)
#                 to restrict the benchmark to that subset
#
# Example (200-sample smoke run):
#   head -200 b-mod_lines/train.easy > /tmp/sample.tsv
#   python -m OCR.crnn_vgg16_bn.benchmark --images b-mod_lines/lines --answers /tmp/sample.tsv
#
# The images directory may be `b-mod_lines/lines` (filenames are resolved inside its
# `train/` and `test/` subdirectories) or one of those subdirectories directly.
#
# Metric definitions:
#   CER = sum(Levenshtein(pred, gt)) / sum(len(gt))       (per-character edit distance)
#   WER = sum(Levenshtein(pred_words, gt_words)) / sum(len(gt_words))  (per-word)
#
# Note: the model's vocabulary does NOT contain the space character (docTR trains it on
# word crops), so its predictions cannot contain spaces. Therefore:
#   - CER is computed on whitespace-STRIPPED text by default (--keep-spaces compares the
#     raw strings, counting the missing spaces as character errors),
#   - WER is always computed on RAW strings: the prediction is usually a single token,
#     so WER mostly reflects the (missing) word segmentation of the model.

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .predictor import Recognizer, load_image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# Edit distances
# --------------------------------------------------------------------------- #
try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def _str_distance(a: str, b: str) -> int:
        return int(_Lev.distance(a, b))

except ImportError:

    def _str_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = cur
        return prev[-1]


def _token_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two word (token) sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# Ground-truth loading
# --------------------------------------------------------------------------- #
def load_answers(answers_path: str) -> dict[str, str]:
    """Load a b-mod answers file: one '<filename> <transcript>' per line.

    The separator between filename and transcript is the first whitespace run
    (filenames never contain spaces; transcripts do).
    """
    gt: dict[str, str] = {}
    with open(answers_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{answers_path}:{line_no}: expected '<filename> <transcript>' "
                    f"(got: {line[:60]!r})"
                )
            name, text = parts
            gt[name.strip()] = text
    if not gt:
        raise ValueError(f"{answers_path}: no '<filename> <transcript>' entries found")
    return gt


def load_split_file(split_path: str) -> list[str]:
    """Load an optional list of filenames (one per line) to restrict the benchmark."""
    with open(split_path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def resolve_images(images_dir: str, filenames: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Resolve answer filenames to image paths relative to images_dir.

    Since the b-mod dataset reorganization, images live in `train/` and `test/`
    subdirectories of the lines dir, so each name is looked up at:
      <images_dir>/<name>, <images_dir>/train/<name>, <images_dir>/test/<name>
    """
    subdirs = ["", "train", "test"]
    found, missing = [], []
    for name in filenames:
        if not name.lower().endswith(IMG_EXTS):
            missing.append(name)
            continue
        for sub in subdirs:
            p = os.path.join(images_dir, sub, name) if sub else os.path.join(images_dir, name)
            if os.path.isfile(p):
                found.append(p)
                break
        else:
            missing.append(name)
    return found, missing, filenames


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def normalize(s: str, keep_spaces: bool) -> str:
    """Strip ALL whitespace from the string (the model cannot produce spaces)."""
    return s if keep_spaces else "".join(s.split())


def run(args: argparse.Namespace) -> int:
    gt = load_answers(args.answers)
    filenames = list(gt.keys())
    if args.split_file:
        split_names = load_split_file(args.split_file)
        unknown = [n for n in split_names if n not in gt]
        if unknown:
            print(f"warning: {len(unknown)} names in split file have no answer entry (skipped)", file=sys.stderr)
        filenames = [n for n in split_names if n in gt]
    if not filenames:
        sys.exit("error: no samples to benchmark")

    paths, missing, _ = resolve_images(args.images, filenames)
    if missing:
        print(f"warning: {len(missing)} answer entries have no image file (skipped)", file=sys.stderr)
    if not paths:
        sys.exit(f"error: no images found under {args.images}")

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit(
            "error: --device cuda requested but torch.cuda is not available "
            "(this machine has a CPU-only torch build and/or no GPU)"
        )

    recognizer = Recognizer(
        weights_path=args.weights,
        split_wide_crops=not args.no_split,
        device=args.device,
        batch_size=args.batch_size,
        amp=args.amp,
    )

    # name -> path for lookup of predictions
    name_of = {p: os.path.basename(p) for p in paths}

    total_char_dist, total_gt_chars = 0, 0
    total_word_dist, total_gt_words = 0, 0
    n_exact, n_done = 0, 0
    per_sample: list[dict] = []
    t0 = time.perf_counter()

    for chunk_start in range(0, len(paths), args.chunk_size):
        chunk = paths[chunk_start : chunk_start + args.chunk_size]
        crops = [load_image(p) for p in chunk]
        preds = recognizer(crops)

        for path, (pred_text, _conf) in zip(chunk, preds):
            gt_text = gt[name_of[path]]
            p_norm, g_norm = normalize(pred_text, args.keep_spaces), normalize(gt_text, args.keep_spaces)

            char_dist = _str_distance(p_norm, g_norm)
            total_char_dist += char_dist
            total_gt_chars += len(g_norm)

            # WER is always computed on the raw (unmodified) strings
            pred_words, gt_words = pred_text.split(), gt_text.split()
            total_word_dist += _token_distance(pred_words, gt_words)
            total_gt_words += len(gt_words)

            n_exact += int(p_norm == g_norm)
            n_done += 1
            per_sample.append({
                "image": os.path.basename(path),
                "gt": gt_text,
                "pred": pred_text,
                "cer": char_dist / max(len(g_norm), 1),
            })

        if args.progress and (chunk_start // args.chunk_size) % args.progress == 0:
            elapsed = time.perf_counter() - t0
            cer_now = total_char_dist / max(total_gt_chars, 1)
            wer_now = total_word_dist / max(total_gt_words, 1)
            print(
                f"[{n_done}/{len(paths)}] {elapsed:.1f}s | CER {cer_now:.4f} | WER {wer_now:.4f}",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t0
    cer = total_char_dist / max(total_gt_chars, 1)
    wer = total_word_dist / max(total_gt_words, 1)
    acc = n_exact / max(n_done, 1)

    summary = {
        "images": n_done,
        "missing": len(missing),
        "CER": round(cer, 6),
        "WER": round(wer, 6),
        "exact_match_acc": round(acc, 6),
        "chars": total_gt_chars,
        "words": total_gt_words,
        "time_s": round(elapsed, 2),
        "throughput_img_s": round(n_done / max(elapsed, 1e-9), 2),
        "device": str(recognizer.device),
        "amp": recognizer.amp,
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(f"{row['image']}\t{row['cer']:.6f}\t{row['gt']}\t{row['pred']}\n")
        print(f"per-sample details written to {args.output}", file=sys.stderr)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CER/WER benchmark on the b-mod dataset (standalone crnn_vgg16_bn)")
    parser.add_argument("--images", required=True, help="directory containing the crop images")
    parser.add_argument("--answers", required=True, help="b-mod answers file ('<filename><TAB><transcript>' per line)")
    parser.add_argument("--split-file", default=None, help="optional filename list (one per line) to restrict the set")
    parser.add_argument("--batch-size", type=int, default=128, help="model batch size (default 128)")
    parser.add_argument("--chunk-size", type=int, default=256, help="images loaded in memory per chunk (default 256)")
    parser.add_argument("--device", default=None, help="cpu | cuda (auto-detected by default)")
    parser.add_argument("--amp", action="store_true", help="fp16 mixed precision on GPU (ignored on CPU)")
    parser.add_argument("--weights", default=None, help="path to the .pt checkpoint (default: bundled weights/)")
    parser.add_argument("--no-split", action="store_true", help="disable wide-crop splitting (AR>8)")
    parser.add_argument("--keep-spaces", action="store_true", help="do NOT strip whitespace before computing CER")
    parser.add_argument("--output", default=None, help="optional TSV path with per-sample details")
    parser.add_argument("--progress", type=int, default=0, metavar="N", help="print progress every N chunks")
    args = parser.parse_args()

    if not os.path.isfile(args.answers):
        sys.exit(f"error: answers file not found: {args.answers}")
    if not os.path.isdir(args.images):
        sys.exit(f"error: images directory not found: {args.images}")

    sys.exit(run(args))


if __name__ == "__main__":
    main()
