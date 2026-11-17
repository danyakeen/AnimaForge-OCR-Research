# CRNN_VGG16_BN — standalone `crnn_vgg16_bn` recognition model

Self-contained port of docTR's **CRNN + VGG-16-BN** text recognition model
(`doctr/models/recognition/crnn/pytorch.py`, torch backend), ready to run **without
installing the `doctr` package**.

No detection module — input is a single text **line/word crop**, output is the recognized
string + confidence.

**English configuration** (100-char vocab, 101 output classes). **No official english-vocab
pretrained recognition checkpoint exists** — every docTR zoo recognition model (crnn, sar,
parseq, vitstr, master, viptr) is french-vocab. The french-trained `crnn_vgg16_bn` weights
were therefore removed; the model instead initializes its backbone from docTR's
**backbone-pretraining checkpoint** (`vgg16_bn_r-d108c19c.pt` — the same initialization
docTR uses when building a CRNN without recognition weights), while the output head starts
fresh. **Train the model (see below) before using it for inference.**

## Layout

```
CRNN_VGG16_BN/
├── crnn.py                     # CRNN model, CTC best-path decoding, weight loading
├── backbone.py                 # torchvision VGG-16-BN with rectangular (2,1) MaxPools
├── vocab.py                    # english vocabulary, 100 chars (index i -> class i)
├── preprocess.py               # resize (32,128), preserve aspect ratio, pad, normalize
├── predictor.py                # Recognizer: batching + wide-crop splitting/merging (AR > 8)
├── inference.py                # CLI: single image -> recognized character sequence
├── benchmark.py                # CLI: CER/WER benchmark on the b-mod dataset
├── train.py                    # CLI: fine-tune/train on the b-mod dataset, date-time checkpoints
├── README.md
└── weights/
    └── vgg16_bn_r-d108c19c.pt  # docTR backbone-pretraining checkpoint (doctr-static, v0.4.1;
                                 #   downloaded automatically on first use)
```

## What was ported (and where it came from)

| Piece | Source in the repo |
|---|---|
| `CRNN` forward / structure | `doctr/models/recognition/crnn/pytorch.py` |
| VGG-16-BN rectangular-pool backbone | `doctr/models/classification/vgg/pytorch.py` (`vgg16_bn_r`) |
| CTC post-processor | same crnn file (`CTCPostProcessor`) |
| Preprocessing (resize AR-preserve, normalize) | `doctr/models/preprocessor/pytorch.py`, `doctr/transforms` (recognition zoo does **not** symmetric-pad) |
| Wide-crop splitting & merging | `doctr/models/recognition/predictor/_utils.py`, `doctr/models/recognition/utils.py` |
| Vocabulary | `doctr/datasets/vocabs.py` → `VOCABS["english"]` (100 chars) |
| Backbone pretraining checkpoint | `default_cfgs["vgg16_bn_r"]["url"]` (doctr-static v0.4.1) |

## Usage

All commands are run from the repo root (`AnimaForge-OCR-Research/`), so the
`CRNN_VGG16_BN` package is importable.

### Single-image inference

> Note: until the model has been trained, predictions are meaningless (fresh head).

```bash
python -m CRNN_VGG16_BN.inference <image_path>
python -m CRNN_VGG16_BN.inference <image> --confidence --json
python -m CRNN_VGG16_BN.inference <image> --weights CRNN_VGG16_BN/checkpoints/<run>/best.pt
```

### Benchmark (CER / WER) on the b-mod dataset

```bash
python -m CRNN_VGG16_BN.benchmark \
    --images   b-mod_lines/lines \
    --answers  b-mod_lines/train.easy \
    --batch-size 256 --progress 2
```

- `--answers` expects the b-mod format, one `<filename> <transcript>` per line
  (`train.easy`, `train.medium`, `train.hard` in `b-mod_lines/`).
- `--split-file` (optional) restricts the run to a listed subset of filenames
  (e.g. `b-mod_lines/test.easy`).
- Prints a JSON summary: `CER`, `WER`, exact-match accuracy, throughput;
  `--output details.tsv` stores per-sample CER, GT and prediction.
- `--progress N` prints running CER/WER every N chunks to stderr.
- GPU: `--device cuda` runs on a CUDA GPU; `--amp` enables fp16 mixed precision
  (GPU only, off by default). On a CPU-only torch build, `--device cuda` exits
  with a clear error.

Metric notes (the model's vocab has **no space character**, it was trained on word crops):

- **CER** = total char-level Levenshtein distance / total GT chars, computed on
  whitespace-stripped text by default (`--keep-spaces` compares raw strings, counting
  the model's missing spaces as errors).
- **WER** = word-level Levenshtein distance / total GT words, always on raw strings;
  since predictions are space-less (usually one token per line), WER mostly reflects
  the missing word segmentation.

### Training / fine-tuning on the b-mod dataset

```bash
python -m CRNN_VGG16_BN.train --data-dir b-mod_lines --device cuda --amp --epochs 10

# smoke run on a small subset (CPU)
python -m CRNN_VGG16_BN.train --data-dir b-mod_lines --limit-train 128 --limit-val 64 --epochs 1
```

- `--data-dir` points at the dataset root with `lines/{train,test}` and the
  `train.*` / `test.*` answer files (`<filename> <transcript>` per line).
- `--train-splits` / `--test-splits` restrict the answer files
  (e.g. `--train-splits b-mod_lines/train.easy`).
- `--train-dir` / `--test-dir` point the image lookup at a chosen folder instead of
  the full `<data-dir>/lines` (e.g. train on a subfolder, test on `lines/test`).
- Recipe (ported from doctr's `references/recognition/train.py`): Adam (lr 1e-3),
  CTC loss with blank = len(vocab), gradient clipping at 5, OneCycleLR by default
  (`--scheduler cosine` for CosineAnnealingLR), `--amp` for fp16 on CUDA.
- `--from-scratch` trains with fully random init; by default the backbone loads docTR's
  `vgg16_bn_r` pretraining checkpoint and the output head starts fresh;
  `--resume <run_dir>/last.pt` continues a previous run.
- Checkpoints are saved in **date-time format** under `checkpoints/<YYYY-MM-DD_HHMMSS>/`:
  `config.json` + `history.jsonl`, `epoch_001.pt`… (pure model state dicts, usable as
  `--weights` in inference.py / benchmark.py), `best.pt` (lowest val loss) and
  `last.pt` (full training state for `--resume`).
- The vocab has no space char and transcripts are full line sentences: characters
  outside the vocab are stripped from targets by default (`--oov {strip,skip,error}`).
  Validation reports the CTC test loss and exact-match accuracy.

Install dependencies with `pip install -r requirements.txt` (`torch`, `torchvision`,
`numpy`, `Pillow`; `rapidfuzz` and `tqdm` are used when available — CPU is fine).

### Python API

```python
from CRNN_VGG16_BN.predictor import Recognizer, load_image

reco = Recognizer()                                  # loads weights/, auto device
crop = load_image("b-mod_lines/lines/train/<file>.jpg")  # H x W x 3 uint8 RGB
text, conf = reco([crop])[0]
print(text, conf)
```

Batch any list of crops: `texts = [t for t, _ in reco(crops)]`.

## Model facts

- Input: RGB crop, resized to **32 x 128** preserving aspect ratio (padding bottom-right, zeros),
  normalized with mean `(0.694, 0.695, 0.693)`, std `(0.299, 0.296, 0.301)`.
- Backbone: torchvision `vgg16_bn` with the **last 3 MaxPool kernels replaced by (2, 1)**
  → features `(N, 512, 4, 16)` for a 32x128 input.
- Head: reshape to sequence `(N, 16, 2048)` → 2-layer bi-LSTM (hidden 128) → Linear(256 → 101).
- Output: CTC logits `(N, T, 101)`; 100 vocab classes + 1 blank (index 100, last).
  Best-path decoding collapses repeats/blanks; word confidence = **min** char probability.
- Wide crops (aspect ratio > 8) are split into slices targeting AR 6 with 50 % overlap and
  the per-slice predictions are merged (edit-distance alignment) — identical to docTR.
  ~88 % of the crops in `b-mod_lines/lines` exceed AR 8, so this path is exercised heavily.
- Max decode length: 32 timesteps (fixed, as in docTR).

## Notes

- There is no english-vocab pretrained recognition checkpoint for this architecture; the
  bundled `weights/vgg16_bn_r-d108c19c.pt` only pretrains the VGG backbone (its french
  classification head is ignored on load). The CRNN's output head is always trained by
  `train.py`. Any trained checkpoint (`best.pt`, `epoch_*.pt` — or even a training-state
  `last.pt`) can be passed as `--weights` to inference.py / benchmark.py.
- The original `merge_strings` uses `rapidfuzz` for Levenshtein distance/opcodes. This port
  uses `rapidfuzz` when installed (bit-exact with docTR) and otherwise falls back to an
  equivalent pure-Python DP implementation — **both paths were verified to produce 0
  mismatches against the original docTR `recognition_predictor` on 200–500 real line crops**
  from `b-mod_lines/lines` (including the wide-crop split/merge path).
