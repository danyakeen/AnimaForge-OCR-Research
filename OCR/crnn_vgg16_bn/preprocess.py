# Preprocessing for the `crnn_vgg16_bn` recognition model, standalone version.
#
# Replicates doctr's PreProcessor + Resize(preserve_aspect_ratio=True) behavior:
#   - resize the crop so it fits within (32, 128) keeping its aspect ratio,
#   - pad the remainder (bottom-right, zeros) — recognition zoo does NOT use symmetric pad,
#   - scale to [0, 1], normalize with the model's mean/std, permute to (C, H, W).
#
# The checkpoint's crop-splitting behavior (aspect ratio > 8) is handled in predictor.py.

import math

import numpy as np
import torch
from torch.nn.functional import pad as torch_pad
from torchvision.transforms import functional as F_v

from .crnn import DEFAULT_CFG

MEAN = DEFAULT_CFG["mean"]
STD = DEFAULT_CFG["std"]
TARGET_SIZE = (32, 128)  # (H, W)


def _resize_one(img: np.ndarray) -> np.ndarray:
    """Resize an H x W x C uint8 crop to (32, 128) preserving aspect ratio, pad bottom-right.

    Matches doctr.transforms.Resize((32, 128), preserve_aspect_ratio=True, symmetric_pad=False).
    """
    # copy: PIL-produced arrays are often read-only, and torch raises a warning on them
    tensor = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1)
    h, w = tensor.shape[-2:]
    target_h, target_w = TARGET_SIZE
    target_ratio = target_h / target_w
    actual_ratio = h / w

    if actual_ratio == target_ratio:
        out = F_v.resize(tensor, [target_h, target_w], antialias=True)
    else:
        if actual_ratio > target_ratio:
            tmp_size = (target_h, max(int(target_h / actual_ratio), 1))
        else:
            tmp_size = (max(int(target_w * actual_ratio), 1), target_w)
        img_r = F_v.resize(tensor, tmp_size, antialias=True)

        # Pad bottom-right (torch.nn.functional.pad order: left, right, top, bottom)
        _pad = (0, target_w - img_r.shape[-1], 0, target_h - img_r.shape[-2])
        out = torch_pad(img_r, _pad)

    return out


def preprocess_crops(crops: list[np.ndarray], batch_size: int = 128) -> list[torch.Tensor]:
    """Prepare line/word crops for the CRNN.

    Args:
        crops: list of H x W x C numpy arrays (uint8 RGB)
        batch_size: images per batch

    Returns:
        list of normalized float tensors of shape (B, 3, 32, 128)
    """
    samples = []
    for crop in crops:
        t = _resize_one(crop).float().div(255)
        t = (t - torch.tensor(MEAN, dtype=t.dtype).view(-1, 1, 1)) / torch.tensor(STD, dtype=t.dtype).view(-1, 1, 1)
        samples.append(t)

    n_batches = int(math.ceil(len(samples) / batch_size))
    return [
        torch.stack(samples[i * batch_size : min((i + 1) * batch_size, len(samples))], dim=0)
        for i in range(n_batches)
    ]


__all__ = ["preprocess_crops", "TARGET_SIZE", "MEAN", "STD"]
