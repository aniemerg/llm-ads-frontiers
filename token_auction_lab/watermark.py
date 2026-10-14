"""Versioned experimental watermark adapter. Decoding uses text and a fixed profile only."""

import hashlib
import json
import re
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.ad_stego import tilt, tilt_race, word_redundant, word_window
from experiments.ad_stego.prototype import Config, Span, decode
from experiments.ad_stego.tokenizer_check import REVISION, QwenTokenizer

KEY = 1901  # Public research key, not an authentication secret.
VOCABULARY_SIZE = 151936
CONFIG = Config()
MENTION_POLICY = "one-region-per-brand-repeated-evidence-v3"
LEGACY_MENTION_POLICY = "first-eligible-mention-per-brand-v1"
EVIDENCE = Path(__file__).resolve().parents[1] / "experiments/ad_stego/evidence"
PROBABILITY_NOTE = (
    "Coded mixture under the ideal independent random-field model; the finite fixed-key "
    "sampler is experimental. This is not a fixed-key probability guarantee."
)


@lru_cache(maxsize=1)
def tokenizer_path():
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        "Qwen/Qwen3-4B-Instruct-2507", "tokenizer.json", revision=REVISION
    )
    if not isinstance(path, str):
        raise ValueError("The pinned Qwen tokenizer is not cached locally.")
    return Path(path)


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = QwenTokenizer(tokenizer_path())
    tokenizer.vocab_size = VOCABULARY_SIZE  # Retain the model's padded vocabulary rows.
    return tokenizer


def profile(legacy=False):
    return {
        "protocol": tilt_race.PROTOCOL if legacy else word_redundant.PROTOCOL,
        "scheme": "tilt-race",
        "key": KEY,
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": REVISION,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path().read_bytes()).hexdigest(),
        "vocabulary_size": VOCABULARY_SIZE,
        "config": asdict(CONFIG),  # Legacy token-grid parameters and shared length/alpha limits.
        "region_grid_words": word_window.GRID,
        "max_region_words": word_window.MAX_WORDS,
        "carrier_words": word_redundant.CHUNK_WORDS,
        "region_width_words": word_redundant.REGION_WORDS,
        "evidence_policy": "repeat alternating lanes to EOS; borrow empty lane",
        "minimum_evidence_tokens": word_window.MIN_EVIDENCE_TOKENS,
        "decoder_alpha_per_protocol": CONFIG.alpha / 2,
        "historical_decoder_alpha": CONFIG.alpha / 4,
        "position_reference": "token index" if legacy else "visible prefix UTF-8 byte length",
        "strength": tilt.STRENGTH,
        "preferred_fraction": tilt.GREEN_PROBABILITY,
        "probability_note": PROBABILITY_NOTE,
        "experimental": True,
    }


def decode_legacy_response(text, alpha=CONFIG.alpha):
    if not isinstance(text, str) or not text or len(text) > 60000:
        raise ValueError("Paste an exact response of 1–60,000 characters.")
    tokenizer = get_tokenizer()
    encoding = tokenizer.backend.encode(text, add_special_tokens=False)
    result = decode(text, tokenizer, key=KEY, config=replace(CONFIG, alpha=alpha),
                    scheme="tilt-race")
    for span in result["spans"]:
        offsets = encoding.offsets[span["start_token"] : span["end_token"]]
        lo, hi = min(x[0] for x in offsets), max(x[1] for x in offsets)
        span.update(
            protocol=tilt_race.PROTOCOL,
            start_char=lo,
            end_char=hi,
            literal_text=text[lo:hi],
            start_utf16=len(text[:lo].encode("utf-16-le")) // 2,
            end_utf16=len(text[:hi].encode("utf-16-le")) // 2,
        )
    return {
        **result,
        "profile": profile(legacy=True),
        "token_count": len(encoding.ids),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "minimum_tokens": 2 * CONFIG.block_tokens,
    }


def decode_response(text):
    """Try current and historical protocols with a shared nominal error budget.

    No response-specific selector, advertiser list, run ID, or encoder audit.
    The original alpha=.01 decoder remains available for historical regression.
    """
    legacy = decode_legacy_response(text, alpha=CONFIG.alpha / 4)
    previous = word_window.decode(text, get_tokenizer(), key=KEY, alpha=CONFIG.alpha / 4)
    current = word_redundant.decode(text, get_tokenizer(), key=KEY, alpha=CONFIG.alpha / 2)
    spans = []
    for family in (legacy, previous, current):
        for s in family["spans"]:
            lo, hi = s["start_char"], s["end_char"]
            if len(text[lo:hi].split()) > word_window.MAX_WORDS:
                continue
            spans.append({**s, "literal_text": text[lo:hi],
                          "start_utf16": len(text[:lo].encode("utf-16-le")) // 2,
                          "end_utf16": len(text[:hi].encode("utf-16-le")) // 2})
    statuses = {legacy["status"], previous["status"], current["status"]}
    status = ("detected" if spans else "unsupported_length" if "unsupported_length" in statuses
              else "insufficient_evidence" if statuses == {"insufficient_evidence"}
              else "ambiguous" if "ambiguous" in statuses else "not_detected")
    return {"status": status,
            "spans": sorted(spans, key=lambda s: (s["start_char"], s["end_char"])),
            "coverage": "unknown", "profile": profile(), "token_count": legacy["token_count"],
            "text_sha256": legacy["text_sha256"], "max_region_words": word_window.MAX_WORDS,
            "families": [{k: v for k, v in f.items() if k not in
                          ("profile", "text_sha256", "token_count", "minimum_tokens")}
                         for f in (current, previous, legacy)]}


class Encoder(word_redundant.Encoder):
    def __init__(self, brands, tokenizer=None):
        super().__init__(brands, tokenizer or get_tokenizer(), KEY)


class WordWindowEncoder(word_window.Encoder):
    def __init__(self, brands, tokenizer=None):
        super().__init__(brands, tokenizer or get_tokenizer(), KEY)


class LegacyEncoder:
    """FIFO first eligible mention per brand; never changes model contexts."""

    def __init__(self, brands, tokenizer=None):
        self.tokenizer = tokenizer or get_tokenizer()
        self.brands = tuple(dict.fromkeys(b.strip() for b in brands))
        self.queue, self.records, self.alignment_failures = [], [], []
        self.scheduled_brands = set()
        self.active = None

    def at_step(self, step, generated, text):
        if step % CONFIG.block_tokens:
            return self.active
        self.active = None
        encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
        if encoding.ids != generated:
            self.alignment_failures.append(step)
        # Once alignment breaks, abstain from further encoding in this run.
        if self.alignment_failures:
            return None
        mentions = []
        for brand in self.brands:
            if brand in self.scheduled_brands:
                continue
            for match in re.finditer(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", text):
                touched = [
                    i
                    for i, (lo, hi) in enumerate(encoding.offsets)
                    if lo < match.end() and hi > match.start()
                ]
                if not touched or len(touched) > CONFIG.max_span_tokens:
                    continue
                start, end = touched[0], touched[-1] + 1
                surface = self.tokenizer.decode(encoding.ids[start:end])
                if end < step:
                    # Reserve at enqueue time, including while another brand's
                    # block is active. Repeated mentions never consume a block.
                    self.scheduled_brands.add(brand)
                    mentions.append({"start_token": start, "end_token": end, "text": surface})
                    break
        self.queue.extend(sorted(mentions, key=lambda r: (r["start_token"], r["end_token"])))
        if self.queue:
            span = self.queue.pop(0)
            self.active = {
                **span,
                "carrier_start": step,
                "carrier_end": step + CONFIG.block_tokens,
                "key": KEY,
                "message_seed": tilt_race.message_seed(
                    Span(span["start_token"], span["end_token"]), span["text"]
                ),
            }
            self.records.append(self.active.copy())
        return self.active

    def finish(self, text, visible_ids):
        canonical = self.tokenizer.encode(text)
        return {
            "mention_policy": LEGACY_MENTION_POLICY,
            "records": [
                {**r, "complete": r["carrier_end"] <= len(canonical)} for r in self.records
            ],
            "pending_mentions": self.queue,
            "prefix_alignment_failures": self.alignment_failures,
            "canonical_roundtrip": canonical == visible_ids,
            "canonical_token_count": len(canonical),
        }


def coded_distributions(raw, payload, step):
    if payload is None:
        return raw
    return tilt.transform(*raw, key=payload["key"], message=payload["message_seed"],
                          step=payload.get("byte_position", step))


def sample(raw, bids, tape, payload, step):
    return tilt_race.sample(
        *raw, bids, tape, key=payload["key"], message=payload["message_seed"],
        step=payload.get("byte_position", step)
    )


def mixture_tv(raw, coded, bids):
    a, b = bids
    return float(
        np.abs((a * raw[0] + b * raw[1]) / (a + b) - (a * coded[0] + b * coded[1]) / (a + b)).sum()
        / 2
    )


def examples():
    return json.loads((EVIDENCE / "ui_examples.json").read_text())
