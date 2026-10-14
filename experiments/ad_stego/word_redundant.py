"""Repeat two fixed-width region codewords in public, text-derived lanes."""

import math
import re
from bisect import bisect_left

import numpy as np

from experiments.ad_stego import tilt_race, word_window
from experiments.ad_stego.prototype import _digest, gamma_log_tail, score_threshold

PROTOCOL = "word16-byte-lanes-repeat-green4-exp-v3"
GRID = 8
REGION_WORDS = 16
CHUNK_WORDS = 16
MIN_EVIDENCE_TOKENS = 32


def message_seed(record):
    return _digest(f"{PROTOCOL}|{record['start_word']}|{record['end_word']}|{record['text']}")


def eligible_word(end_word):
    # Strictly after the region closes, so its last word cannot still change.
    return (end_word // CHUNK_WORDS + 1) * CHUNK_WORDS


def positions(text, encoding):
    starts = [m.start() for m in word_window.words(text)]
    chunks = np.array([bisect_left(starts, lo) // CHUNK_WORDS
                       for lo, _hi in encoding.offsets])
    byte_offsets = [len(text[:lo].encode()) for lo, _hi in encoding.offsets]
    return chunks, byte_offsets


def decode(text, tokenizer, *, key, alpha=0.005, max_tokens=4096):
    encoding = tokenizer.backend.encode(text, add_special_tokens=False)
    base = {"spans": [], "coverage": "unknown", "protocol": PROTOCOL, "alpha": alpha}
    if len(encoding.ids) > max_tokens:
        return {**base, "status": "unsupported_length"}
    words = word_window.words(text)
    regions = [word_window.region(text, words, start, start + REGION_WORDS)
               for start in range(0, len(words) - REGION_WORDS + 1, GRID)]
    chunks, byte_offsets = positions(text, encoding)
    eligible = np.array([eligible_word(r["end_word"]) // CHUNK_WORDS for r in regions])
    messages = np.array([message_seed(r) for r in regions], dtype=np.uint64)
    counts = np.zeros((len(regions), 2), dtype=int)
    scores = np.zeros((len(regions), 2))
    for i, token in enumerate(encoding.ids):
        active = chunks[i] >= eligible
        lane = int((chunks[i] - 1) % 2)
        scores[active, lane] += tilt_race.token_evidence(
            token, key, messages[active], byte_offsets[i], tokenizer.vocab_size)
        counts[active, lane] += 1
    # The third hypothesis catches a single region borrowing both empty lanes.
    scores = np.column_stack((scores, scores.sum(axis=1)))
    counts = np.column_stack((counts, counts.sum(axis=1)))
    tests = int((counts >= MIN_EVIDENCE_TOKENS).sum())
    if not tests:
        return {**base, "status": "insufficient_evidence", "hypotheses_tested": 0}
    found = []
    for index, r in enumerate(regions):
        matches = []
        for lane in range(3):
            count = int(counts[index, lane])
            if count < MIN_EVIDENCE_TOKENS:
                continue
            score = float(scores[index, lane])
            threshold = score_threshold(count, tests, alpha)
            if score < threshold:
                continue
            indices = [i for i, chunk in enumerate(chunks) if chunk >= eligible[index]
                       and (lane == 2 or (chunk - 1) % 2 == lane)]
            touched = [i for i, (lo, hi) in enumerate(encoding.offsets)
                       if lo < r["end_char"] and hi > r["start_char"]]
            matches.append({
                **r, "protocol": PROTOCOL, "lane": lane if lane < 2 else "all",
                "start_token": touched[0], "end_token": touched[-1] + 1,
                "carrier_start_word": eligible_word(r["end_word"]),
                "carrier_start": indices[0], "carrier_end": indices[-1] + 1,
                "carrier_token_indices": indices, "scored_tokens": count,
                "score": score, "score_threshold": threshold,
                "nominal_corrected_p": min(1., math.exp(
                    gamma_log_tail(score, count) + math.log(tests))),
            })
        if matches:
            # One region shown once, even if both dedicated and combined tests pass.
            found.append(min(matches, key=lambda m: m["nominal_corrected_p"]))
    return {**base, "status": "detected" if found else "not_detected", "spans": found,
            "hypotheses_tested": tests,
            "schedule": "16-word alternating lanes; all later eligible tokens scored"}


class Encoder:
    def __init__(self, brands, tokenizer, key):
        self.brands = tuple(dict.fromkeys(b.strip() for b in brands))
        self.tokenizer, self.key = tokenizer, key
        self.scheduled_brands, self.scheduled_regions = set(), set()
        self.records, self.queue = [], []
        self.chunk, self.active = -1, None

    def at_step(self, step, generated, text):
        words = word_window.words(text)
        chunk = len(words) // CHUNK_WORDS
        if chunk != self.chunk:
            if self.active:
                self.active["carrier_end"] = step
            self.chunk, self.active = chunk, None
            discovered = []
            for brand in self.brands:
                if brand in self.scheduled_brands:
                    continue
                for match in re.finditer(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", text):
                    touched = [i for i, w in enumerate(words)
                               if w.start() < match.end() and w.end() > match.start()]
                    if not touched:
                        continue
                    start = touched[0] // GRID * GRID
                    end = start + REGION_WORDS
                    if touched[-1] >= end or eligible_word(end) > chunk * CHUNK_WORDS:
                        continue
                    self.scheduled_brands.add(brand)
                    if (start, end) not in self.scheduled_regions:
                        self.scheduled_regions.add((start, end))
                        discovered.append(word_window.region(text, words, start, end))
                    break
            for r in sorted(discovered, key=lambda r: r["start_word"]):
                encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
                touched = [i for i, (lo, hi) in enumerate(encoding.offsets)
                           if lo < r["end_char"] and hi > r["start_char"]]
                self.records.append({
                    **r, "protocol": PROTOCOL, "key": self.key, "message_seed": message_seed(r),
                    "lane": len(self.records), "start_token": touched[0],
                    "end_token": touched[-1] + 1, "carriers": [],
                    "carrier_start_word": eligible_word(r["end_word"]),
                })
            if self.records:
                lane = (chunk - 1) % 2
                r = self.records[lane] if lane < len(self.records) else self.records[0]
                self.active = {"carrier_start": step, "carrier_end": None,
                               "carrier_start_word": chunk * CHUNK_WORDS,
                               "carrier_end_word": (chunk + 1) * CHUNK_WORDS}
                r["carriers"].append(self.active)
                self.active_region = r
        if self.active:
            return {**{k: v for k, v in self.active_region.items() if k != "carriers"},
                    **self.active, "byte_position": len(text.encode())}
        return None

    def finish(self, text, visible_ids):
        encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
        chunks, _ = positions(text, encoding)
        records = []
        for r in self.records:
            carriers = [{**c, "carrier_end": c["carrier_end"] or len(visible_ids)}
                        for c in r["carriers"]]
            transmitted_chunks = {c["carrier_start_word"] // CHUNK_WORDS for c in carriers}
            count = int(sum(chunk in transmitted_chunks for chunk in chunks))
            records.append({**r, "carriers": carriers, "scored_tokens": count,
                            "complete": count >= MIN_EVIDENCE_TOKENS,
                            "carrier_start": carriers[0]["carrier_start"] if carriers else 0,
                            "carrier_end": carriers[-1]["carrier_end"] if carriers else 0})
        return {"mention_policy": "one-region-per-brand-repeated-evidence-v3", "records": records,
                "pending_mentions": [], "prefix_alignment_failures": [],
                "canonical_roundtrip": encoding.ids == visible_ids,
                "canonical_token_count": len(encoding.ids), "protocol": PROTOCOL,
                "max_region_words": REGION_WORDS, "word_definition": "whitespace-delimited",
                "position_policy": "UTF-8 prefix bytes; 16-word alternating evidence lanes"}
