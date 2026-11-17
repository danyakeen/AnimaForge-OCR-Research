#!/usr/bin/env python3
# Single-image inference for the standalone pretrained docTR `crnn_vgg16_bn` model.
#
# Usage:
#   python -m CRNN_VGG16_BN.inference <image_path> [--device cpu|cuda] [--no-split] [--confidence] [--json]
#
# Example:
#   python -m CRNN_VGG16_BN.inference b-mod_lines/lines/000107da14ad9242de30c504a45f7750.jpg_rec_l0000.jpg
#
# Output: the recognized character sequence printed to stdout (plain by default;
# with --confidence appends the confidence; with --json prints a JSON object).

import argparse
import json
import os
import sys

from .predictor import Recognizer, load_image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="crnn_vgg16_bn single-image text recognition (standalone, no doctr)"
    )
    parser.add_argument("image", help="path to a single image (line/word crop)")
    parser.add_argument("--weights", default=None, help="path to the .pt checkpoint (default: bundled weights/)")
    parser.add_argument("--device", default=None, help="cpu | cuda (auto-detected by default)")
    parser.add_argument("--no-split", action="store_true", help="disable wide-crop splitting (AR>8)")
    parser.add_argument("--confidence", action="store_true", help="also print the recognition confidence")
    parser.add_argument("--json", action="store_true", help="print output as JSON")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"error: not a file: {args.image}")
    if not args.image.lower().endswith(IMG_EXTS):
        sys.exit(f"error: unsupported image extension: {args.image}")

    recognizer = Recognizer(
        weights_path=args.weights,
        split_wide_crops=not args.no_split,
        device=args.device,
    )
    text, confidence = recognizer([load_image(args.image)])[0]

    if args.json:
        print(json.dumps({"image": args.image, "text": text, "confidence": confidence}))
    elif args.confidence:
        print(f"{text}\t{confidence:.4f}")
    else:
        print(text)


if __name__ == "__main__":
    main()
