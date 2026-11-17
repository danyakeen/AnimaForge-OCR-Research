# CRNN text recognition model (VGG-16-BN backbone), standalone version.
#
# Extracted from doctr/models/recognition/crnn/pytorch.py (torch backend).
# No dependency on the `doctr` package: only torch, torchvision, numpy.
#
# Input : RGB image tensor (N, 3, 32, 128), float in [0, 1], normalized (see preprocess.py)
# Output: logits (N, T, len(vocab) + 1) with T = 32 max decode steps; english vocab -> 101 classes

from typing import Any

import numpy as np
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import vgg16_bn_r_features
from .vocab import DEFAULT_VOCAB

__all__ = ["CRNN", "CTCPostProcessor", "crnn_vgg16_bn"]


# --------------------------------------------------------------------------- #
# Sequence encoding (ported from doctr/datasets/utils.py)
# --------------------------------------------------------------------------- #
def encode_string(input_string: str, vocab: str) -> list[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers."""
    table = {char: idx for idx, char in enumerate(vocab)}
    try:
        return [table[char] for char in input_string]
    except KeyError as e:
        missing_chars = [char for char in input_string if char not in table]
        raise ValueError(
            f"Some characters cannot be found in 'vocab': {set(missing_chars)}. "
            f"Please check the input string `{input_string}` and the vocabulary `{vocab}`"
        ) from e


def encode_sequences(
    sequences: list[str],
    vocab: str,
    target_size: int | None = None,
    eos: int = -1,
    sos: int | None = None,
    pad: int | None = None,
    dynamic_seq_length: bool = False,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping.

    Ported from `doctr.datasets.utils.encode_sequences`.
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    if not isinstance(target_size, int) or dynamic_seq_length:
        # Maximum string length + EOS
        max_length = max((len(w) for w in sequences), default=0) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)

    # Pad all sequences
    if isinstance(pad, int):  # pad with padding symbol
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        default_symbol = pad
    else:  # pad with eos symbol
        default_symbol = eos
    encoded_data: np.ndarray = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)

    # Encode the strings
    for idx, seq in enumerate(map(lambda s: encode_string(s, vocab), sequences)):
        if isinstance(pad, int):  # add eos at the end of the sequence
            seq.append(eos)
        encoded_data[idx, : min(len(seq), target_size)] = seq[: min(len(seq), target_size)]

    if isinstance(sos, int):  # place sos symbol at the beginning of each sequence
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos

    return encoded_data

DEFAULT_CFG: dict[str, Any] = {
    "mean": (0.694, 0.695, 0.693),
    "std": (0.299, 0.296, 0.301),
    "input_shape": (3, 32, 128),
    "vocab": DEFAULT_VOCAB,
}


class CTCPostProcessor:
    """Postprocess raw prediction (logits) to a list of (text, confidence) using CTC best-path decoding.

    Highly inspired from https://github.com/githubharald/CTCDecoder
    """

    def __init__(self, vocab: str = DEFAULT_CFG["vocab"], confidence_aggregation: str = "min") -> None:
        self.vocab = vocab
        self.confidence_aggregation = confidence_aggregation

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: str = DEFAULT_CFG["vocab"],
        blank: int = 0,
        confidence_aggregation: str = "min",
    ) -> list[tuple[str, float]]:
        """Best-path (greedy) CTC decoding.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of the blank label
            confidence_aggregation: "min" | "mean" | ... how to merge per-char probs into word confidence

        Returns:
            A list of tuples: (text, confidence)
        """
        best_paths = logits.argmax(dim=-1).detach().cpu().numpy()
        probs = F.softmax(logits.float(), dim=-1).max(dim=-1).values.detach().cpu().numpy()

        results = []
        for path, path_probs in zip(best_paths, probs):
            # Collapse the repeated labels: probability of a char = max within its run
            run_starts = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
            labels, label_probs = path[run_starts], np.maximum.reduceat(path_probs, run_starts)
            # Remove the blanks
            is_char = labels != blank
            char_probs = label_probs[is_char]
            results.append((
                "".join([vocab[idx] for idx in labels[is_char]]),
                (float(char_probs.min()) if char_probs.size else 0.0)
                if confidence_aggregation == "min"
                else (float(char_probs.mean()) if char_probs.size else 0.0),
            ))

        return results

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        return self.ctc_best_path(
            logits=logits,
            vocab=self.vocab,
            blank=len(self.vocab),
            confidence_aggregation=self.confidence_aggregation,
        )


class CRNN(nn.Module):
    """CRNN as described in "An End-to-End Trainable Neural Network for Image-based Sequence
    Recognition and Its Application to Scene Text Recognition" (https://arxiv.org/pdf/1507.05717.pdf).
    """

    def __init__(
        self,
        vocab: str = DEFAULT_CFG["vocab"],
        input_shape: tuple[int, int, int] = DEFAULT_CFG["input_shape"],
        rnn_units: int = 128,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.input_shape = input_shape
        self.max_length = 32
        self.feat_extractor = vgg16_bn_r_features(pretrained_backbone=pretrained_backbone)

        # Resolve the input_size of the LSTM.
        # eval() first: a training-mode forward would pollute the freshly loaded
        # BatchNorm running statistics with statistics of this zeros dummy input.
        self.feat_extractor.eval()
        with torch.inference_mode():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        self.feat_extractor.train()
        lstm_in = out_shape[1] * out_shape[2]  # 512 * 4 = 2048 for a (3, 32, 128) input

        self.decoder = nn.LSTM(
            input_size=lstm_in,
            hidden_size=rnn_units,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        # features units = 2 * rnn_units because bidirectional layers
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits.

        Args:
            x: (N, 3, 32, 128) float tensor, normalized

        Returns:
            logits of shape (N, T, len(vocab) + 1)

        Note: NOT decorated with `torch.inference_mode()` so the model can be trained
        (`inference_mode` tensors cannot feed autograd). Callers doing inference should
        wrap calls in `torch.inference_mode()` / `no_grad` themselves.
        """
        features = self.feat_extractor(x)
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        return self.linear(logits)

    def predict(self, x: torch.Tensor) -> list[tuple[str, float]]:
        """Forward + CTC decoding.

        Args:
            x: (N, 3, 32, 128) normalized input tensor

        Returns:
            list of (text, confidence), one per input image
        """
        logits = self.forward(x)
        # N x T x C -> N x T x C (postprocessor expects (N, T, C))
        return self.postprocessor(logits)

    # ------------------------------------------------------------------ #
    # Training helpers (ported from doctr/models/recognition)
    # ------------------------------------------------------------------ #
    def build_target(
        self,
        gts: list[str],
    ) -> tuple[np.ndarray, list[int]]:
        """Encode a list of ground-truth strings into a padded index array + sequence lengths.

        Ported from `doctr.models.recognition.core.RecognitionModel.build_target`:
        `encode_sequences(..., target_size=self.max_length, eos=len(self.vocab))`.

        Args:
            gts: list of ground-truth labels

        Returns:
            (encoded (N, max_length) int32 array, per-sample target lengths)
        """
        encoded = encode_sequences(
            sequences=gts,
            vocab=self.vocab,
            target_size=self.max_length,
            eos=len(self.vocab),
        )
        # Cap at max_length: sequences longer than the decode budget are truncated,
        # and F.ctc_loss requires target_length <= input_length (zero_infinity handles the rest).
        seq_len = [min(len(word), self.max_length) for word in gts]
        return encoded, seq_len

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: list[str],
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            model_output: predicted logits (N, T, len(vocab) + 1)
            target: list of target strings

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt),
            input_length,
            torch.tensor(seq_len, dtype=torch.int),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss


def crnn_vgg16_bn(
    weights_path: str | None = None,
    pretrained_backbone: bool = True,
) -> CRNN:
    """Instantiate the crnn_vgg16_bn recognition model (english vocab).

    Args:
        weights_path: path to a trained CRNN checkpoint (state_dict, e.g. a
            `best.pt`/`epoch_*.pt` produced by train.py). When omitted, the model is
            built fresh with docTR's backbone-pretraining initialization
            (`vgg16_bn_r-d108c19c.pt`, downloaded on first use if missing) and a
            freshly initialized output head.
        pretrained_backbone: initialize the backbone from docTR's backbone-pretraining
            checkpoint when no `weights_path` is given (this is the initialization
            docTR uses for a CRNN without recognition weights).

    Returns:
        CRNN model in eval mode.

    Note:
        There is NO official english-vocab pretrained recognition checkpoint for this
        architecture (docTR's zoo models are all french-vocab). The output `linear`
        head must be trained (train.py) before predictions are meaningful.
    """
    import os

    model = CRNN(
        vocab=DEFAULT_CFG["vocab"],
        input_shape=DEFAULT_CFG["input_shape"],
        pretrained_backbone=pretrained_backbone and weights_path is None,
    )

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            # training-state checkpoints (last.pt) contain optimizer/scheduler objects;
            # they are written by this package's train.py, so loading is safe
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Training-state checkpoints wrap the model under "model"; unwrap if needed.
        if not any(k.startswith("feat_extractor.") for k in state_dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        # A docTR-style checkpoint may carry the full torchvision VGG (incl. its
        # `classifier` head) and a `postprocessor` entry; drop non-module keys.
        for key in list(state_dict):
            if key.startswith("classifier.") or key == "postprocessor":
                state_dict.pop(key)

        own_state = model.state_dict()
        skipped = []
        for key, param in state_dict.items():
            if key in own_state and own_state[key].shape == param.shape:
                own_state[key] = param
            else:
                skipped.append(key)
        model.load_state_dict(own_state, strict=True)
        if skipped:
            print(
                f"warning: {len(skipped)} checkpoint tensor(s) not loaded "
                f"({', '.join(skipped)}); those layers keep their current weights",
                file=sys.stderr,
            )

    model.eval()
    return model
