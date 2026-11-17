# Inference predictor for the standalone `crnn_vgg16_bn`, replicating doctr's
# RecognitionPredictor behavior, including the wide-crop splitting logic
# (doctr/models/recognition/predictor/{pytorch,_utils}.py + utils.py).
#
# doctr splits crops whose aspect ratio (w/h) exceeds 8 into overlapping slices
# targeting an aspect ratio of 6 (e.g. 32x192), runs recognition on all crops and
# merges the split predictions back (text alignment via Levenshtein, min confidence).
# This matters for long text lines, which are wider than 8:1.

import math
import os

import numpy as np
import torch
from PIL import Image

from .crnn import DEFAULT_CFG, crnn_vgg16_bn
from .preprocess import preprocess_crops

__all__ = ["Recognizer", "load_image"]


# --------------------------------------------------------------------------- #
# Wide-crop splitting (port of doctr/models/recognition/predictor/_utils.py)
# --------------------------------------------------------------------------- #
def _split_horizontally(
    image: np.ndarray,
    split_width: int,
    overlap_width: int,
) -> tuple[list[np.ndarray], float]:
    """Horizontally split a single image into overlapping slices."""
    image_width = image.shape[1]
    if image_width <= split_width:
        return [image], 0.0

    step = split_width - overlap_width
    starts = list(range(0, image_width - split_width + 1, step))

    if starts[-1] + split_width < image_width:
        starts.append(image_width - split_width)

    splits = [image[:, start : start + split_width, :] for start in starts]

    last_overlap = 0
    if len(starts) > 1:
        last_overlap = (starts[-2] + split_width) - starts[-1]
    last_overlap_ratio = last_overlap / split_width if split_width else 0.0

    return splits, last_overlap_ratio


def split_crops(
    crops: list[np.ndarray],
    max_ratio: float,
    target_ratio: int,
    split_overlap_ratio: float,
) -> tuple[list[np.ndarray], list[int | tuple[int, int, float]], bool]:
    """Split crops horizontally if they exceed a given aspect ratio.

    Returns:
        (new_crops, crop_map, remap_required) where crop_map entries are either an
        index into new_crops (crop untouched) or a tuple
        (start_idx, end_idx, last_overlap_ratio) describing a group of splits.
    """
    remap_required = False
    new_crops: list[np.ndarray] = []
    crop_map: list[int | tuple[int, int, float]] = []

    for crop in crops:
        h, w = crop.shape[:2]

        if h > 0 and w / h > max_ratio:
            split_width = max(1, math.ceil(h * target_ratio))
            overlap_width = max(0, math.floor(split_width * split_overlap_ratio))
            splits, last_overlap = _split_horizontally(crop, split_width, overlap_width)
            splits = [s for s in splits if all(dim > 0 for dim in s.shape)]

            if splits:
                crop_map.append((len(new_crops), len(new_crops) + len(splits), last_overlap))
                new_crops.extend(splits)
                remap_required = True
            else:
                crop_map.append(len(new_crops))
                new_crops.append(crop)
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)

    return new_crops, crop_map, remap_required


# --------------------------------------------------------------------------- #
# Levenshtein helpers
# --------------------------------------------------------------------------- #
# doctr uses rapidfuzz (distance + opcodes). Use it when installed for bit-exact
# behavior; otherwise fall back to an equivalent pure-Python DP implementation.
try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def _levenshtein_distance(a: str, b: str, score_cutoff: int | float = math.inf) -> int:
        return int(_Lev.distance(a, b, score_cutoff=score_cutoff))

    def _levenshtein_opcodes(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
        return [
            (op.tag, op.src_start, op.src_end, op.dest_start, op.dest_end)
            for op in _Lev.opcodes(a, b)
        ]

except ImportError:

    def _levenshtein_distance(a: str, b: str, score_cutoff: int | float = math.inf) -> int:
        """Edit distance with early cutoff (returns > score_cutoff when exceeding it)."""
        if abs(len(a) - len(b)) > score_cutoff:
            return int(score_cutoff) + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if min(cur) > score_cutoff:
                return int(score_cutoff) + 1
            prev = cur
        return prev[-1]

    def _levenshtein_opcodes(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
        """Edit-script as (tag, src_start, src_end, dest_start, dest_end) tuples.

        Tags: 'equal', 'replace', 'insert', 'delete' (semantics similar to difflib).
        """
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        ops: list[tuple[str, int, int, int, int]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                i, j = i - 1, j - 1
                continue
            if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                ops.append(("insert", i, i, j - 1, j))
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                ops.append(("delete", i - 1, i, j, j))
                i -= 1
            elif i > 0 and j > 0:
                ops.append(("replace", i - 1, i, j - 1, j))
                i, j = i - 1, j - 1
            else:
                break
        ops.reverse()

        # Merge adjacent 'equal' regions into difflib-like segments
        merged: list[tuple[str, int, int, int, int]] = []
        for op in ops:
            if merged and op[0] == "equal" and merged[-1][0] == "equal" and merged[-1][2] == op[1]:
                last = merged[-1]
                merged[-1] = ("equal", last[1], op[2], last[3], op[4])
            else:
                merged.append(op)
        return merged


def _merge_strings(a: str, b: str, overlap_ratio: float) -> str:
    """Merge 2 char sequences in the best way to maximize the alignment of their
    overlapping characters (port of doctr.models.recognition.utils.merge_strings).
    """
    overlap_ratio = min(max(overlap_ratio, 0.0), 1.0)
    if min(len(a), len(b)) <= 1:
        return a + b

    a_crop, b_crop = a[:-1], b[1:]  # border chars might be cut off
    max_overlap = min(len(a_crop), len(b_crop))
    expected = len(b) * overlap_ratio - 2

    best_cost, best_i, best_j, best_dist = max(expected, 0.0) + 1.0, 0, 0, math.inf

    for i in sorted(range(1, max_overlap + 1), key=lambda n: (abs(n - expected), -n)):
        length_cost = abs(i - expected)
        if length_cost > best_cost:
            break
        for j in range(max(1, i - 2), min(len(b_crop), i + 2) + 1):
            fixed_cost = length_cost + 0.25 * abs(i - j)
            cutoff = min(math.floor(0.8 * max(i, j)), math.floor(best_cost - fixed_cost))
            if cutoff < abs(i - j):
                continue
            dist = _levenshtein_distance(a_crop[-i:], b_crop[:j], score_cutoff=cutoff)
            if dist > cutoff:
                continue
            cost = dist + fixed_cost
            if cost < best_cost or (cost == best_cost and (dist, -i) < (best_dist, -best_i)):
                best_cost, best_i, best_j, best_dist = cost, i, j, dist

    if best_i == 0:
        return a + b if len(b) * overlap_ratio < 1 else a_crop + b_crop

    tail, head = a_crop[-best_i:], b_crop[:best_j]
    cuts = [(0, 0)]
    for tag, src_start, src_end, dest_start, dest_end in _levenshtein_opcodes(tail, head):
        if tag == "equal":
            cuts.extend((src_start + t, dest_start + t) for t in range(1, src_end - src_start + 1))
        else:
            cuts.append((src_end, dest_end))
    cut_a, cut_b = min(cuts, key=lambda c: abs(c[0] + c[1] - (best_i + best_j) / 2))
    return a_crop[: len(a_crop) - best_i] + tail[:cut_a] + head[cut_b:] + b_crop[best_j:]


def _merge_multi_strings(seq_list: list[str], overlap_ratio: float, last_overlap_ratio: float) -> str:
    """Merge consecutive left-to-right sequences with overlapping characters."""
    if not seq_list:
        return ""
    result = seq_list[0]
    for i in range(1, len(seq_list)):
        text_b = seq_list[i]
        ratio = last_overlap_ratio if i == len(seq_list) - 1 else overlap_ratio
        result = _merge_strings(result, text_b, ratio)
    return result


def _remap_preds(
    preds: list[tuple[str, float]],
    crop_map: list[int | tuple[int, int, float]],
    overlap_ratio: float,
) -> list[tuple[str, float]]:
    """Reconstruct predictions of the original crops from possibly-split crop predictions."""
    remapped = []
    for item in crop_map:
        if isinstance(item, int):
            remapped.append(preds[item])
        else:
            start_idx, end_idx, last_overlap = item
            text_parts, confidences = zip(*preds[start_idx:end_idx])
            merged_text = _merge_multi_strings(list(text_parts), overlap_ratio, last_overlap)
            text_confidences = [conf for text, conf in zip(text_parts, confidences) if text]
            merged_conf = min(text_confidences) if text_confidences else min(confidences)
            remapped.append((merged_text, merged_conf))
    return remapped


# --------------------------------------------------------------------------- #
# Recognizer
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Load an image file as an H x W x 3 uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


class Recognizer:
    """Line/word crop text recognizer built on the pretrained crnn_vgg16_bn.

    Args:
        weights_path: path to the .pt checkpoint (default: bundled in weights/)
        batch_size: number of (possibly split) crops per forward pass
        split_wide_crops: split crops with aspect ratio > `critical_ar` (matches doctr)
    """

    def __init__(
        self,
        weights_path: str | None = None,
        batch_size: int = 128,
        split_wide_crops: bool = True,
        device: str | None = None,
        amp: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda is not available — "
                "this torch build is CPU-only (torch==2.13.0+cpu in the current venv)."
            )
        self.model = crnn_vgg16_bn(weights_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.split_wide_crops = split_wide_crops
        self.amp = bool(amp) and self.device.type == "cuda"  # mixed precision (GPU only)
        self.critical_ar = 8      # Critical aspect ratio
        self.target_ar = 6        # Target aspect ratio after splitting
        self.overlap_ratio = 0.5  # Overlap ratio between neighboring crops

    @torch.inference_mode()
    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize a list of H x W x 3 uint8 RGB crops.

        Returns:
            list of (text, confidence), one per input crop
        """
        if len(crops) == 0:
            return []
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")
        empty = next((crop for crop in crops if not all(s > 0 for s in crop.shape)), None)
        if empty is not None:
            raise ValueError(f"incorrect input shape: crops are expected to be non-empty, got shape {empty.shape}.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(crops, self.critical_ar, self.target_ar, self.overlap_ratio)
            if remapped:
                crops = new_crops

        # Resize & batch
        batches = preprocess_crops(crops, batch_size=self.batch_size)

        # Forward
        preds: list[tuple[str, float]] = []
        for batch in batches:
            batch = batch.to(self.device)
            if self.amp:
                # fp16 autocast for the conv/LSTM forward; decode in fp32
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model.forward(batch)
                preds.extend(self.model.postprocessor(logits.float()))
            else:
                preds.extend(self.model.predict(batch))

        # Remap split crops
        if self.split_wide_crops and remapped:
            preds = _remap_preds(preds, crop_map, self.overlap_ratio)

        return preds
