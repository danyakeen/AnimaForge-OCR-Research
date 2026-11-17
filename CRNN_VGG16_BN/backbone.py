# VGG-16 backbone used by the docTR `crnn_vgg16_bn` recognition model.
#
# Extracted from doctr/models/classification/vgg/pytorch.py (torch backend).
# The backbone is torchvision's `vgg16_bn` with two modifications:
#   1. the last 3 MaxPool2d layers use rectangular kernels (2, 1) instead of (2, 2)
#      -> input (3, 32, 128) yields features of shape (N, 512, 4, 16)
#   2. classification head replaced; irrelevant here since CRNN only uses `.features`.
#
# When `pretrained_backbone=True`, the feature extractor is initialized from docTR's
# backbone-pretraining checkpoint (`vgg16_bn_r-d108c19c.pt`, doctr-static v0.4.1),
# exactly like `doctr.models.recognition.crnn` does when building a model without
# recognition weights. The checkpoint's classification head (french-vocab, unused)
# is ignored. The english CRNN (100-char vocab) has no official pretrained
# recognition checkpoint; its `linear` head is always freshly initialized and must
# be trained.

import os
import urllib.request

import torch
from torch import nn
from torchvision.models import vgg as tv_vgg

__all__ = ["vgg16_bn_r_features", "BACKBONE_URL", "BACKBONE_FILENAME"]

BACKBONE_URL = "https://doctr-static.mindee.com/models?id=v0.4.1/vgg16_bn_r-d108c19c.pt&src=0"
BACKBONE_FILENAME = "vgg16_bn_r-d108c19c.pt"


def _backbone_weights_path() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", BACKBONE_FILENAME)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading backbone-pretrained weights to {path} ...", flush=True)
        urllib.request.urlretrieve(BACKBONE_URL, path)
        print("done.", flush=True)
    return path


def _load_backbone_pretrained(features: nn.Module) -> None:
    """Load the docTR backbone-pretraining checkpoint into `features`.

    Mirrors doctr's `load_pretrained_params(..., ignore_keys=['classifier.weight',
    'classifier.bias'])` for `vgg16_bn_r`.
    """
    state_dict = torch.load(_backbone_weights_path(), map_location="cpu", weights_only=True)

    own_state = features.state_dict()
    loaded, skipped = 0, []
    for key, param in state_dict.items():
        if key.startswith("classifier."):
            continue
        # checkpoint keys are relative to the full VGG model: 'features.<i>.*'
        own_key = key[len("features."):]
        if own_key in own_state and own_state[own_key].shape == param.shape:
            own_state[own_key] = param
            loaded += 1
        else:
            skipped.append(key)
    features.load_state_dict(own_state, strict=True)
    if skipped:
        print(
            f"warning: {len(skipped)} backbone tensor(s) not loaded "
            f"({', '.join(sk[:40] for sk in skipped)})",
            flush=True,
        )
    print(f"backbone initialized from {BACKBONE_FILENAME} ({loaded} tensors)", flush=True)


def vgg16_bn_r_features(pretrained_backbone: bool = False) -> nn.Module:
    """Build the VGG-16-BN feature extractor (rectangular-pool variant) used by CRNN.

    Args:
        pretrained_backbone: when True, load docTR's backbone-pretraining checkpoint
            (`vgg16_bn_r-d108c19c.pt`) into the features — this is the initialization
            docTR itself uses when building a CRNN without recognition weights.

    Returns:
        nn.Module: `model.features`, the part consumed by the CRNN.
    """
    model = tv_vgg.__dict__["vgg16_bn"](weights=None)
    # list the MaxPool2d
    pool_idcs = [idx for idx, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
    # Replace their kernel with rectangular ones
    for idx in pool_idcs[-3:]:
        model.features[idx] = nn.MaxPool2d((2, 1))

    if pretrained_backbone:
        _load_backbone_pretrained(model.features)

    return model.features
