#!/usr/bin/env python3
# Training for the standalone `crnn_vgg16_bn` recognition model on the b-mod dataset.
#
# Ported from doctr's training recipe (references/recognition/train.py):
#   Adam (lr=0.001), CTC loss (blank = len(vocab)), grad-clip 5,
#   OneCycleLR (default) or CosineAnnealingLR, optional AMP on CUDA.
#
# Dataset layout (b-mod_lines):
#   <data-dir>/lines/train/*.jpg     train crops
#   <data-dir>/lines/test/*.jpg      test crops
#   <data-dir>/train.*               "<filename> <transcript>" per line, files point into lines/train
#   <data-dir>/test.*                same for lines/test
#
# Usage:
#   python -m CRNN_VGG16_BN.train --data-dir b-mod_lines
#   python -m CRNN_VGG16_BN.train --data-dir b-mod_lines --device cuda --amp --epochs 10
#   python -m CRNN_VGG16_BN.train --data-dir b-mod_lines --train-splits train.easy --limit-train 1000
#   python -m CRNN_VGG16_BN.train --train-dir path/to/subfolder --train-splits path/to/answers.txt \
#       --data-dir b-mod_lines
#
# Checkpoints are saved in date-time format:
#   checkpoints/<YYYY-MM-DD_HHMMSS>/
#     ├── config.json         full training configuration
#     ├── history.jsonl       per-epoch metrics
#     ├── epoch_001.pt ...    model state_dict (loadable by inference.py / benchmark.py
#     │                       via `--weights`, or crnn_vgg16_bn(weights_path=...))
#     ├── best.pt             model state_dict with the lowest test loss
#     └── last.pt             full training state (model + optimizer + scheduler) for --resume
#
# Notes:
#   - The vocabulary has NO space character; by default characters outside the vocab
#     are stripped from targets (--oov {strip,skip,error}).
#   - Targets are truncated to max_length=32 like at decode time; the CTC loss uses
#     blank = len(vocab) (= 100 for the english vocab) and zero_infinity=True.

import argparse
import copy
import datetime as _dt
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .crnn import DEFAULT_CFG, crnn_vgg16_bn
from .preprocess import _resize_one

MEAN = torch.tensor(DEFAULT_CFG["mean"]).view(-1, 1, 1)
STD = torch.tensor(DEFAULT_CFG["std"]).view(-1, 1, 1)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _resolve_image(images_root: str, name: str) -> str | None:
    """Resolve an answer filename under images_root: <root>/<name>, then root/train, root/test."""
    for sub in ("", "train", "test"):
        p = os.path.join(images_root, sub, name) if sub else os.path.join(images_root, name)
        if os.path.isfile(p) and name.lower().endswith(IMG_EXTS):
            return p
    return None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _find_split_files(data_dir: str, prefix: str) -> list[str]:
    """All answer files in data_dir whose name starts with the prefix (e.g. 'train.')."""
    return sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith(prefix) and os.path.isfile(os.path.join(data_dir, f))
    )


def _read_answers(paths: list[str]) -> dict[str, str]:
    """b-mod answers format: one '<filename> <transcript>' per line (SPACE separator)."""
    gt: dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{line_no}: expected '<filename> <transcript>' (got: {line[:60]!r})")
                name, text = parts
                if name.strip() in gt:
                    raise ValueError(f"{path}:{line_no}: duplicate filename {name.strip()!r}")
                gt[name.strip()] = text
    return gt


class LineCropDataset(Dataset):
    """Line/word crops + transcripts from a b-mod-style answer set.

    Images are resolved under `images_root` (or its `train/` / `test/` subdirs),
    resized to (32, 128) preserving aspect ratio (padded bottom-right) and normalized.
    """

    def __init__(
        self,
        answers: dict[str, str],
        images_root: str,
        oov_policy: str = "strip",
        vocab: str = DEFAULT_CFG["vocab"],
        augment: bool = False,
    ) -> None:
        self.vocab = vocab
        self.augment = augment

        self.samples: list[tuple[str, str]] = []   # (image path, cleaned transcript)
        self.dropped_oov = 0
        self.dropped_missing = 0
        self.stripped_chars = 0

        allowed = set(vocab)
        for name, text in answers.items():
            if oov_policy == "strip":
                kept = "".join(c for c in text if c in allowed)
                self.stripped_chars += len(text) - len(kept)
                text = kept
            elif oov_policy == "skip" and any(c not in allowed for c in text):
                self.dropped_oov += 1
                continue
            elif oov_policy == "error":
                bad = sorted(set(c for c in text if c not in allowed))
                if bad:
                    raise ValueError(
                        f"transcript for {name} contains characters outside the vocab: {bad} "
                        f"(use --oov strip|skip to handle them)"
                    )
            if not text:
                self.dropped_oov += 1
                continue

            path = self._resolve(images_root, name)
            if path is None:
                self.dropped_missing += 1
                continue
            self.samples.append((path, text))

        if not self.samples:
            raise ValueError("no usable samples (all dropped as OOV or missing images?)")

    @staticmethod
    def _resolve(images_root: str, name: str) -> str | None:
        return _resolve_image(images_root, name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path, text = self.samples[idx]
        img = np.asarray(Image.open(path).convert("RGB"))
        tensor = _resize_one(img).float().div(255)
        tensor = (tensor - MEAN) / STD
        if self.augment and torch.rand(()) < 0.1:
            tensor = 1.0 - tensor  # color inversion, as in the doctr recipe (p=0.1)
        return tensor, text


def _collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    images = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return images, texts


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
def train_one_epoch(model, device, loader, optimizer, scheduler, amp: bool) -> float:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total_loss, n_batches = 0.0, 0
    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model.compute_loss(model(images), texts)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.compute_loss(model(images), texts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, device, loader, amp: bool) -> tuple[float, float]:
    """Returns (weighted val loss, exact-match accuracy on decoded strings)."""
    model.eval()
    total_loss, n_samples = 0.0, 0
    n_exact = 0
    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        total_loss += model.compute_loss(logits.float(), texts).item() * images.shape[0]
        n_samples += images.shape[0]
        preds = model.postprocessor(logits.float())
        n_exact += sum(int(p == t) for (p, _), t in zip(preds, texts))
    return total_loss / max(n_samples, 1), n_exact / max(n_samples, 1)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_model_state(model, path: str) -> None:
    """Save a pure model state_dict, loadable by crnn_vgg16_bn(weights_path=...)."""
    torch.save(model.state_dict(), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the standalone crnn_vgg16_bn model on a b-mod-style dataset"
    )
    parser.add_argument("--data-dir", required=True,
                        help="dataset root containing lines/{train,test} and train.*/test.* answer files")
    parser.add_argument("--train-splits", default=None, metavar="FILE[,FILE...]",
                        help="answer files used for training (default: every 'train*' file in --data-dir)")
    parser.add_argument("--train-dir", default=None, metavar="DIR",
                        help="image folder for training (default: <data-dir>/lines; "
                             "use this to train on a chosen subfolder instead of the full train/ set)")
    parser.add_argument("--test-splits", default=None, metavar="FILE[,FILE...]",
                        help="answer files used for testing (default: every 'test*' file in --data-dir)")
    parser.add_argument("--test-dir", default=None, metavar="DIR",
                        help="image folder for testing (default: <data-dir>/lines)")
    parser.add_argument("--output-dir", default=None,
                        help="root folder for the date-time checkpoint folder (default: checkpoints/ next to this script)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["onecycle", "cosine"], default="onecycle")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="cpu | cuda (auto-detected by default)")
    parser.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only)")
    parser.add_argument("--workers", type=int, default=None,
                        help="dataloader workers (default: min(16, cpu_count))")
    parser.add_argument("--from-scratch", action="store_true",
                        help="initialize randomly instead of loading the pretrained weights")
    parser.add_argument("--resume", default=None, metavar="LAST.PT",
                        help="resume training from a full training-state checkpoint (last.pt)")
    parser.add_argument("--oov", choices=["strip", "skip", "error"], default="strip",
                        help="handling of characters outside the vocab in transcripts (default: strip)")
    parser.add_argument("--augment", action="store_true",
                        help="enable light augmentation (10%% color inversion on train crops)")
    parser.add_argument("--limit-train", type=int, default=None, metavar="N",
                        help="use only the first N training samples (smoke tests)")
    parser.add_argument("--limit-val", type=int, default=None, metavar="N",
                        help="use only the first N test samples (smoke tests)")
    args = parser.parse_args()

    # ---- device ---------------------------------------------------------- #
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but torch.cuda is not available "
                 "(this machine has a CPU-only torch build and/or no GPU)")
    if args.amp and not args.device.startswith("cuda"):
        sys.exit("error: --amp (automatic mixed precision) is only supported on CUDA devices")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    workers = args.workers if args.workers is not None else min(16, multiprocessing.cpu_count())
    pin_memory = device.type == "cuda"

    # ---- data ------------------------------------------------------------ #
    if not os.path.isdir(args.data_dir):
        sys.exit(f"error: dataset directory not found: {args.data_dir}")

    def _split_list(given: str | None, prefix: str) -> list[str]:
        if given is not None:
            files = [p for p in (s.strip() for s in given.split(",")) if p]
            missing = [p for p in files if not os.path.isfile(p)]
            if missing:
                sys.exit(f"error: split file(s) not found: {missing}")
            return files
        files = _find_split_files(args.data_dir, prefix)
        if not files:
            sys.exit(f"error: no '{prefix}*' answer files found in {args.data_dir}")
        return files

    train_files = _split_list(args.train_splits, "train")
    test_files = _split_list(args.test_splits, "test")

    train_dir = args.train_dir or os.path.join(args.data_dir, "lines")
    test_dir = args.test_dir or os.path.join(args.data_dir, "lines")
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            sys.exit(f"error: image folder not found: {d}")

    print(f"train answers: {[os.path.basename(f) for f in train_files]}  (images: {train_dir})")
    print(f"test answers:  {[os.path.basename(f) for f in test_files]}  (images: {test_dir})")

    train_set = LineCropDataset(_read_answers(train_files), train_dir, oov_policy=args.oov, augment=args.augment)
    print(f"train set: {len(train_set)} samples "
          f"(dropped: {train_set.dropped_missing} missing images, "
          f"{train_set.dropped_oov} OOV/empty; stripped chars: {train_set.stripped_chars})")
    test_set = LineCropDataset(_read_answers(test_files), test_dir, oov_policy=args.oov)
    print(f"test set:  {len(test_set)} samples "
          f"(dropped: {test_set.dropped_missing} missing images, "
          f"{test_set.dropped_oov} OOV/empty; stripped chars: {test_set.stripped_chars})")

    if args.limit_train:
        train_set.samples = train_set.samples[: args.limit_train]
    if args.limit_val:
        test_set.samples = test_set.samples[: args.limit_val]

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=workers, pin_memory=pin_memory, collate_fn=_collate,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=pin_memory, collate_fn=_collate,
        persistent_workers=workers > 0,
    )

    # ---- model / optimizer ----------------------------------------------- #
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            sys.exit(f"error: resume checkpoint not found: {args.resume}")
        # full training state (optimizer/scheduler objects) — needs weights_only=False;
        # safe here because last.pt is written by this very script
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = crnn_vgg16_bn(weights_path=None, pretrained_backbone=False)  # weights come from the checkpoint
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0) + 1
        best_val_loss = state.get("best_val_loss", math.inf)
        print(f"resumed from {args.resume} (epoch {state.get('epoch')}, "
              f"best val loss {best_val_loss if isinstance(best_val_loss, float) else 'n/a'})")
    elif args.from_scratch:
        model = crnn_vgg16_bn(weights_path=None, pretrained_backbone=False)
        start_epoch, best_val_loss = 0, math.inf
        print("training from scratch (fully random init)")
    else:
        model = crnn_vgg16_bn()  # backbone-pretrained init (docTR vgg16_bn_r), fresh output head
        start_epoch, best_val_loss = 0, math.inf
        print("initialized the backbone from docTR's vgg16_bn_r pretraining checkpoint; "
              "the output head is freshly initialized")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs - start_epoch, steps_per_epoch=steps_per_epoch,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - start_epoch) * steps_per_epoch,
        )

    # ---- checkpoint folder (date-time format) ----------------------------- #
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    run_dir = os.path.join(output_dir, _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "data_dir": os.path.abspath(args.data_dir),
        "train_splits": [os.path.abspath(f) for f in train_files],
        "val_splits": [os.path.abspath(f) for f in test_files],
        "train_dir": os.path.abspath(train_dir),
        "test_dir": os.path.abspath(test_dir),
        "train_samples": len(train_set),
        "val_samples": len(test_set),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "amp": args.amp,
        "workers": workers,
        "from_scratch": args.from_scratch,
        "resume": os.path.abspath(args.resume) if args.resume else None,
        "oov": args.oov,
        "augment": args.augment,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "vocab_size": len(model.vocab),
        "blank_index": len(model.vocab),
        "max_length": model.max_length,
        "torch": torch.__version__,
        "saved_at": _dt.datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    history_path = os.path.join(run_dir, "history.jsonl")

    print(f"checkpoints: {run_dir}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- training loop ----------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, amp=args.amp)
        val_loss, exact_match = validate(model, device, val_loader, amp=args.amp)
        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0]

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "exact_match": round(exact_match, 6),
            "lr": lr_now,
            "time_s": round(elapsed, 1),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | exact {exact_match:.2%} | lr {lr_now:.6f} | {elapsed:.0f}s")

        # epoch checkpoint: pure state_dict, loadable by inference/benchmark
        save_model_state(model, os.path.join(run_dir, f"epoch_{epoch + 1:03d}.pt"))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_model_state(model, os.path.join(run_dir, "best.pt"))

        # full training state for --resume
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        }, os.path.join(run_dir, "last.pt"))

    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
