"""Decode a response file with static configuration only; no model or run trace."""

import argparse
import json
from pathlib import Path

from experiments.ad_stego.prototype import Config, decode
from experiments.ad_stego.tokenizer_check import QwenTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, required=True)
    parser.add_argument("--key", type=int, default=1901)
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--scheme", choices=("affine", "race", "tilt", "tilt-race"),
                        default="affine")
    args = parser.parse_args()
    tokenizer = QwenTokenizer(args.tokenizer_json)
    # The model has padded output rows beyond the tokenizer's assigned IDs.
    # This fixed codebook size must match sampling; do not filter those rows.
    tokenizer.vocab_size = args.vocabulary_size
    result = decode(args.response.read_text(), tokenizer, key=args.key, scheme=args.scheme,
                    config=Config(block_tokens=args.block_tokens, alpha=args.alpha))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
