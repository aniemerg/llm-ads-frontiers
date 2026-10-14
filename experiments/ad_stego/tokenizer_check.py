"""Cached Qwen tokenizer check with synthetic probabilities; never loads a model.

Optional isolated environment: tokenizers==0.22.2 and numpy==2.4.6.
No tokenizer/model download is performed by this script.
"""

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from experiments.ad_stego.prototype import Config, decode
from experiments.ad_stego.synthetic import generate

REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class QwenTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(path))
        self.vocab_size = self.backend.get_vocab_size(with_added_tokens=True)

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(list(map(int, ids)), skip_special_tokens=False)


def main():
    path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507"
            / "snapshots" / REVISION / "tokenizer.json")
    tokenizer = QwenTokenizer(path)
    # Stable, real token IDs for leading-space ASCII words. Not a language model.
    background = []
    for token in range(tokenizer.vocab_size):
        surface = tokenizer.decode([token])
        if (surface.startswith(" ") and 2 <= len(surface[1:]) <= 10
                and surface[1:].isascii() and surface[1:].isalpha()
                and tokenizer.encode(surface) == [token]):
            background.append(token)
    background = background[::max(1, len(background)//256)][:256]
    phrases = [tokenizer.encode(" Paradigm House"), tokenizer.encode(" Fort Mason Express")]
    reserved = {t for phrase in phrases for t in phrase}
    background = [t for t in background if t not in reserved]
    occurrences = ((8, tuple(phrases[0])), (76, tuple(phrases[1])))
    config = Config()
    output = Path("artifacts/ad-stego-tokenizer") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("prototype.py"),
               Path(__file__).with_name("synthetic.py"), Path("auction.py")]
    manifest = {
        "mode": "synthetic_probabilities_real_tokenizer",
        "model": None, "model_revision": None, "prompts": None,
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507", "tokenizer_revision": REVISION,
        "tokenizer_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dependencies": {"python": sys.version, "numpy": np.__version__,
                         "tokenizers": version("tokenizers")},
        "platform": platform.platform(), "seed": 7, "nonsecret_key": 1901,
        "bids": [1, 1], "tokens": 224, "temperature": None, "continuation_tokens": 0,
        "config": asdict(config), "background_ids": background,
        "scripted_occurrences": occurrences,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": "Synthetic distributions; no live LLM, fluency, or general BPE guarantee.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fixture = generate(config=config, tokenizer=tokenizer, background_ids=background,
                       occurrences=occurrences)
    result = decode(fixture["text"], tokenizer, key=1901, config=config)
    report = {"mode": manifest["mode"], "text": fixture["text"],
              "expected_spans": fixture["expected_spans"], "decoded": result,
              "canonical_token_ids": tokenizer.encode(fixture["text"])}
    (output / "example.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = []
    special = {int(k) for k, v in tokenizer.backend.get_added_tokens_decoder().items()
               if v.special}
    for file in sorted(Path("smoke_results").rglob("result.json")):
        data = json.loads(file.read_text())
        ids = data["token_ids"]
        visible = [t for t in ids if t not in special]
        decoded = tokenizer.backend.decode(ids, skip_special_tokens=True)
        reencoded = tokenizer.encode(decoded)
        archive.append({"file": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                        "mode": "REPLAY_TOKENIZER_CHECK", "visible_tokens": len(visible),
                        "retokenized_tokens": len(reencoded), "same_ids": visible == reencoded})
    (output / "archive_tokenizer_check.json").write_text(json.dumps(archive, indent=2) + "\n")
    print(json.dumps({"decoded": result, "archive_roundtrips": archive,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
