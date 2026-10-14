"""Coarse text-only regions with byte-positioned evidence (experimental).

Whitespace words define 8/16-word payloads and 48-word carrier blocks. UTF-8
prefix lengths index the random field: token merges lose local evidence rather
than shifting every subsequent position. No model context is retokenized.
"""

import math
import re
from bisect import bisect_left

import numpy as np

from experiments.ad_stego import tilt_race
from experiments.ad_stego.prototype import _digest, gamma_log_tail, score_threshold

PROTOCOL = "word-window16-byte-green4-exp-v2"
GRID = 8
MAX_WORDS = 16
CARRIER_WORDS = 48
MIN_EVIDENCE_TOKENS = 32


def words(text):
    return list(re.finditer(r"\S+", text))


def region(text, matches, start, end):
    lo, hi = matches[start].start(), matches[end - 1].end()
    return {"start_word": start, "end_word": end, "start_char": lo, "end_char": hi,
            "text": text[lo:hi]}


def message_seed(record):
    return _digest(f"{PROTOCOL}|{record['start_word']}|{record['end_word']}|{record['text']}")


def carrier_groups(text, encoding):
    """The count of words in each visible prefix, including an unfinished word."""
    starts = [w.start() for w in words(text)]
    groups = {}
    for i, (lo, _hi) in enumerate(encoding.offsets):
        block = bisect_left(starts, lo) // CARRIER_WORDS
        if block:
            groups.setdefault(block * CARRIER_WORDS, []).append(i)
    return groups


def decode(text, tokenizer, *, key, alpha=0.005, max_tokens=4096):
    encoding = tokenizer.backend.encode(text, add_special_tokens=False)
    base = {"spans": [], "coverage": "unknown", "protocol": PROTOCOL}
    if len(encoding.ids) > max_tokens:
        return {**base, "status": "unsupported_length"}
    matches = words(text)
    groups = {b: ids for b, ids in carrier_groups(text, encoding).items()
              if len(ids) >= MIN_EVIDENCE_TOKENS}
    candidates = {b: [region(text, matches, start, end)
                      for start in range(0, b, GRID)
                      for end in (start + GRID, start + MAX_WORDS) if end < b]
                  for b in groups}
    tests = sum(map(len, candidates.values()))
    if not tests:
        return {**base, "status": "insufficient_evidence", "hypotheses_tested": 0}
    found, diagnostics = {}, []
    byte_offsets = [len(text[:lo].encode("utf-8")) for lo, _hi in encoding.offsets]
    for block, regions in candidates.items():
        messages = np.array([message_seed(r) for r in regions], dtype=np.uint64)
        scores = np.zeros(len(regions))
        indices = groups[block]
        for i in indices:
            scores += tilt_race.token_evidence(
                encoding.ids[i], key, messages, byte_offsets[i], tokenizer.vocab_size)
        threshold = score_threshold(len(indices), tests, alpha)
        diagnostics.append({"carrier_start_word": block, "scored_tokens": len(indices),
                            "hypotheses": len(regions), "score_threshold": threshold})
        # Each significant hypothesis remains a detection. Overlap is allowed;
        # Bonferroni counts every region/block hypothesis, including overlaps.
        for index in np.flatnonzero(scores >= threshold):
            r = regions[index]
            touched = [i for i, (lo, hi) in enumerate(encoding.offsets)
                       if lo < r["end_char"] and hi > r["start_char"]]
            record = {
                **r, "protocol": PROTOCOL,
                "start_token": touched[0], "end_token": touched[-1] + 1,
                "carrier_start": indices[0], "carrier_end": indices[-1] + 1,
                "carrier_start_word": block, "carrier_end_word": block + CARRIER_WORDS,
                "scored_tokens": len(indices), "score": float(scores[index]),
                "score_threshold": threshold,
                "nominal_corrected_p": min(1., math.exp(
                    gamma_log_tail(float(scores[index]), len(indices)) + math.log(tests))),
            }
            coords = (r["start_word"], r["end_word"])
            if coords not in found or record["score"] > found[coords]["score"]:
                found[coords] = record
    return {**base, "status": "detected" if found else "not_detected",
            "spans": [found[c] for c in sorted(found)], "hypotheses_tested": tests,
            "blocks": diagnostics, "alpha": alpha}


class Encoder:
    def __init__(self, brands, tokenizer, key):
        self.brands = tuple(dict.fromkeys(b.strip() for b in brands))
        self.tokenizer, self.key = tokenizer, key
        self.scheduled_brands, self.scheduled_regions = set(), set()
        self.queue, self.records = [], []
        self.active, self.block = None, 0

    def at_step(self, step, generated, text):
        matches = words(text)
        block = len(matches) // CARRIER_WORDS
        if block != self.block:
            if self.active:
                self.active["carrier_end"] = step
            self.active, self.block = None, block
            before = block * CARRIER_WORDS
            mentions = []
            for brand in self.brands:
                if brand in self.scheduled_brands:
                    continue
                for match in re.finditer(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", text):
                    touched = [i for i, w in enumerate(matches)
                               if w.start() < match.end() and w.end() > match.start()]
                    if not touched:
                        continue
                    start = touched[0] // GRID * GRID
                    end = ((touched[-1] // GRID) + 1) * GRID
                    if end >= before or end - start > MAX_WORDS:
                        continue
                    self.scheduled_brands.add(brand)
                    if (start, end) not in self.scheduled_regions:
                        self.scheduled_regions.add((start, end))
                        mentions.append(region(text, matches, start, end))
                    break
            self.queue.extend(sorted(mentions, key=lambda r: (r["start_word"], r["end_word"])))
            if self.queue:
                record = self.queue.pop(0)
                encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
                touched = [i for i, (lo, hi) in enumerate(encoding.offsets)
                           if lo < record["end_char"] and hi > record["start_char"]]
                self.active = {
                    **record, "protocol": PROTOCOL, "key": self.key,
                    "message_seed": message_seed(record),
                    "start_token": touched[0], "end_token": touched[-1] + 1,
                    "carrier_start_word": before, "carrier_end_word": before + CARRIER_WORDS,
                    "carrier_start": step, "carrier_end": None,
                }
                self.records.append(self.active)
        # A fresh copy prevents later carrier closure from mutating saved events.
        return ({**self.active, "byte_position": len(text.encode("utf-8"))}
                if self.active else None)

    def finish(self, text, visible_ids):
        encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
        groups = carrier_groups(text, encoding)
        records = []
        for r in self.records:
            count = len(groups.get(r["carrier_start_word"], []))
            records.append({**r, "carrier_end": r["carrier_end"] or len(visible_ids),
                            "scored_tokens": count, "complete": count >= MIN_EVIDENCE_TOKENS,
                            "full_word_block": len(words(text)) >= r["carrier_end_word"]})
        return {"mention_policy": "first-eligible-mention-per-brand-v1", "records": records,
                "pending_mentions": self.queue, "prefix_alignment_failures": [],
                "canonical_roundtrip": encoding.ids == visible_ids,
                "canonical_token_count": len(encoding.ids), "protocol": PROTOCOL,
                "max_region_words": MAX_WORDS, "word_definition": "whitespace-delimited",
                "position_policy": ("UTF-8 bytes in visible prefix; "
                                    "local token merges may erase evidence")}
