"""Vocabulary used by the `crnn_vgg16_bn` recognition model.

Extracted from doctr/datasets/vocabs.py (VOCABS["english"]).
Index i in this string corresponds to output class i. The CTC blank is
appended AFTER the vocabulary (class index = len(ENGLISH_VOCAB)).

Note: there is no english-vocab pretrained recognition checkpoint for this architecture;
the model's output head is always trained from scratch (see train.py).
"""

ENGLISH_VOCAB = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿'

DEFAULT_VOCAB = ENGLISH_VOCAB

__all__ = ["ENGLISH_VOCAB", "DEFAULT_VOCAB"]
