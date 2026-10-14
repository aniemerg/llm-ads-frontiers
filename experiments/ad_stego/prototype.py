"""Coordinate codewords through the unchanged two-bidder auction.

Research prototype: public nonsecret test keys; synthetic and isolated live callers.
No production authentication or fixed-key distribution guarantee is claimed.
The decoder enumerates ALL earlier spans, without an advertiser dictionary.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from auction import Allocation, RandomTape, TwoBidderAuction

PROTOCOL = "coordinate-codeword-affine-v1"
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Config:
    block_tokens: int = 64
    max_span_tokens: int = 32
    max_response_tokens: int = 4096
    alpha: float = 0.01

    def __post_init__(self):
        if not 8 <= self.block_tokens <= 256:
            raise ValueError("Block length must be between 8 and 256.")
        if not 1 <= self.max_span_tokens <= 32:
            raise ValueError("Span limit must be between 1 and 32.")
        if not self.block_tokens <= self.max_response_tokens <= 4096:
            raise ValueError("Response limit must be between block length and 4096.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0,1).")


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self):
        if not 0 <= self.start < self.end <= 4096 or self.end - self.start > 32:
            raise ValueError("Invalid span coordinates.")


class SyntheticTokenizer:
    """One pinned toy tokenizer used by both bidders AND text-only decoder.

    Each whitespace-delimited vocabulary item is one token. Never evidence of
    production Qwen segmentation, fluency, model quality, or robustness.
    """

    revision = "synthetic-whitespace-256-v1"
    vocab_size = 256

    def __init__(self):
        self.words = ("Paradigm", "House", "Fort", "Mason", "Express") + tuple(
            f"word{i:03d}" for i in range(251)
        )
        self.lookup = {word: i for i, word in enumerate(self.words)}

    def encode(self, text: str) -> list[int]:
        return [self.lookup[word] for word in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.words[int(i)] for i in ids)


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def message_seed(span: Span, surface: str) -> int:
    """Bind actual coordinates AND their text; no name dictionary at decoding."""
    return _digest(f"{PROTOCOL}|{span.start}|{span.end}|{surface}")


def _mix(x):
    """Vectorized noncryptographic counter mixer, explicitly experimental."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> 31)


@lru_cache(maxsize=8)
def _units(size: int) -> np.ndarray:
    return np.array([i for i in range(1, size) if math.gcd(i, size) == 1], dtype=np.int64)


def coordinates(key: int, messages, step: int, size: int):
    """Rank permutation r(t)=(a*t+b) mod V and two candidate uniforms.

    Affine bijections permit O(1) rank evaluation by a text-only decoder.
    All arrays are keyed independently of bids. Fresh independent random
    coordinates are the theoretical model; this finite mixer is a simulation.
    """
    if size < 2:
        raise ValueError("Need a vocabulary of at least two tokens.")
    base = _mix(np.asarray(messages, dtype=np.uint64) ^ np.uint64(key & MASK64))
    base = base ^ _mix(np.uint64(step))
    fields = [_mix(base ^ np.uint64(c)) for c in (
        0x243F6A8885A308D3, 0x13198A2E03707344,
        0xA4093822299F31D0, 0x082EFA98EC4E6C89,
    )]
    units = _units(size)
    a = units[(fields[0] % len(units)).astype(np.int64)]
    b = (fields[1] % size).astype(np.int64)
    # Midpoints avoid zero/one and permit stable log scoring.
    ua = ((fields[2] >> 12).astype(np.float64) + 0.5) / (1 << 52)
    ub = ((fields[3] >> 12).astype(np.float64) + 0.5) / (1 << 52)
    return a, b, ua, ub


def sample(p_a, p_b, bids, tape: RandomTape, *, key: int, message: int, step: int):
    """Wrap the original sampler; preserve branch and contest draws exactly."""
    a, b, ua, ub = coordinates(key, np.uint64(message), step, len(p_a))
    inv = pow(int(a), -1, len(p_a))
    permutation = ((np.arange(len(p_a)) - int(b)) * inv) % len(p_a)
    auction = TwoBidderAuction(np.asarray(p_a)[permutation], np.asarray(p_b)[permutation])
    allocation = auction.resolve(*bids, replace(tape, token_a=float(ua), token_b=float(ub)))

    def original(token):
        return None if token is None else int(permutation[token])

    return Allocation(
        token_id=original(allocation.token_id), branch=allocation.branch,
        payment_a=allocation.payment_a, payment_b=allocation.payment_b,
        candidate_a=original(allocation.candidate_a),
        candidate_b=original(allocation.candidate_b),
    )


def token_evidence(token: int, key: int, messages, step: int, size: int):
    a, b, ua, ub = coordinates(key, messages, step, size)
    rank = (((a * token + b) % size).astype(np.float64) + 0.5) / size
    distance = np.minimum(np.abs(rank - ua), np.abs(rank - ub))
    width = np.minimum(1.0, rank + distance) - np.maximum(0.0, rank - distance)
    # Exact CDF of min(|rank-Ua|,|rank-Ub|), conditional on rank,
    # for independent continuous uniforms under the ideal random-key null.
    probability = width * (2.0 - width)
    return -np.log(np.maximum(probability, np.finfo(float).tiny))


def gamma_log_tail(score: float, count: int) -> float:
    """log P(Gamma(integer count,1) >= score), using the Poisson sum."""
    if score <= 0:
        return 0.0
    terms = [k * math.log(score) - math.lgamma(k + 1) for k in range(count)]
    largest = max(terms)
    return min(0.0, -score + largest + math.log(sum(math.exp(t-largest) for t in terms)))


@lru_cache(maxsize=128)
def score_threshold(count: int, tests: int, alpha: float) -> float:
    log_alpha = math.log(alpha / tests)
    lo, hi = 0.0, float(count)
    while gamma_log_tail(hi, count) > log_alpha:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if gamma_log_tail(mid, count) > log_alpha:
            lo = mid
        else:
            hi = mid
    return hi


def possible_spans(before: int, max_length: int) -> list[Span]:
    return [Span(start, end) for start in range(before)
            for end in range(start + 1, min(before, start + max_length) + 1)]


def decode(text: str, tokenizer: SyntheticTokenizer, *, key: int, config=Config(),
           scheme="affine") -> dict:
    """Only text plus static configuration. No prompt, catalog, trace, or bids.

    Nominal corrected p-values assume ideal independent random coordinates;
    finite-key and natural-text false positives must also be measured.
    """
    seed_function, evidence_function, protocol = message_seed, token_evidence, PROTOCOL
    threshold_function, tail_function = score_threshold, gamma_log_tail
    if scheme == "race":
        from experiments.ad_stego import race
        seed_function, evidence_function, protocol = (
            race.message_seed, race.token_evidence, race.PROTOCOL)
    elif scheme == "tilt":
        from experiments.ad_stego import tilt
        seed_function, evidence_function, protocol = (
            tilt.message_seed, tilt.token_evidence, tilt.PROTOCOL)
        threshold_function, tail_function = tilt.score_threshold, tilt.log_tail
    elif scheme == "tilt-race":
        from experiments.ad_stego import tilt_race
        seed_function, evidence_function, protocol = (
            tilt_race.message_seed, tilt_race.token_evidence, tilt_race.PROTOCOL)
    elif scheme != "affine":
        raise ValueError("Unknown sampling scheme.")
    try:
        ids = tokenizer.encode(text)
    except KeyError:
        return {"status": "unsupported_text", "spans": [], "coverage": "unknown"}
    if len(ids) > config.max_response_tokens:
        return {"status": "unsupported_length", "spans": [], "coverage": "unknown"}
    width = config.block_tokens
    blocks = list(range(width, len(ids) - width + 1, width))
    candidates = {block: possible_spans(block, config.max_span_tokens) for block in blocks}
    tests = sum(map(len, candidates.values()))
    if tests == 0:
        return {"status": "insufficient_evidence", "spans": [], "coverage": "unknown"}
    threshold = threshold_function(width, tests, config.alpha)
    found, ambiguous = {}, []
    for block, spans in candidates.items():
        messages = np.array([
            seed_function(span, tokenizer.decode(ids[span.start:span.end])) for span in spans
        ], dtype=np.uint64)
        scores = np.zeros(len(spans))
        for step in range(block, block + width):
            scores += evidence_function(ids[step], key, messages, step, tokenizer.vocab_size)
        matches = np.flatnonzero(scores >= threshold)
        if len(matches) > 1:
            ambiguous.append(block)
            continue
        for index in matches:
            span = spans[index]
            record = {
                "start_token": span.start, "end_token": span.end,
                "text": tokenizer.decode(ids[span.start:span.end]),
                "carrier_start": block, "carrier_end": block + width,
                "score": float(scores[index]),
                "nominal_corrected_p": min(1.0, math.exp(
                    tail_function(float(scores[index]), width) + math.log(tests))),
            }
            if span not in found or record["score"] > found[span]["score"]:
                found[span] = record
    return {
        "status": "detected" if found else ("ambiguous" if ambiguous else "not_detected"),
        "spans": [found[span] for span in sorted(found)], "coverage": "unknown",
        "hypotheses_tested": tests, "score_threshold": threshold,
        "ambiguous_blocks": ambiguous, "protocol": protocol,
    }
