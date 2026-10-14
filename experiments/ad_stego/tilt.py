"""Experimental shared preference BEFORE the auction; changes its input probabilities.

One quarter of vocabulary IDs are preferred independently under the ideal PRF.
No token is removed. Each bidder's distribution is tilted and renormalized before
using the original auction and payment calculation. Original-mixture guarantees
do not apply to the resulting changed distributions.
"""

import math
from functools import lru_cache

import numpy as np

from auction import TwoBidderAuction
from experiments.ad_stego.prototype import _digest
from experiments.ad_stego.race import uniform

PROTOCOL = "coordinate-codeword-green4-v1"
GREEN_PROBABILITY = .25
STRENGTH = 4.0


def message_seed(span, surface):
    return _digest(f"{PROTOCOL}|{span.start}|{span.end}|{surface}")


def green(key, messages, step, tokens):
    return uniform(key, messages, step, tokens) < GREEN_PROBABILITY


def transform(p_a, p_b, *, key, message, step, strength=STRENGTH):
    if not math.isfinite(strength) or not 0 <= strength <= 8:
        raise ValueError("Experiment strength must be finite and between zero and eight.")
    mask = green(key, np.uint64(message), step, np.arange(len(p_a), dtype=np.uint64))
    multiplier = np.where(mask, math.exp(strength), 1.)
    coded = [np.asarray(p, dtype=np.float64) * multiplier for p in (p_a, p_b)]
    return tuple(p / p.sum(dtype=np.float64) for p in coded)


def sample(p_a, p_b, bids, tape, *, key, message, step):
    pa, pb = transform(p_a, p_b, key=key, message=message, step=step)
    return TwoBidderAuction(pa, pb).resolve(*bids, tape)


def token_evidence(token, key, messages, step, size):
    del size
    return green(key, messages, step, token).astype(np.float64)


def log_tail(score, count):
    """Exact binomial upper tail under the ideal independent-field null."""
    start = math.ceil(score)
    if start <= 0:
        return 0.
    if start > count:
        return -math.inf
    p = GREEN_PROBABILITY
    terms = [math.lgamma(count + 1) - math.lgamma(k + 1) - math.lgamma(count - k + 1)
             + k * math.log(p) + (count - k) * math.log1p(-p)
             for k in range(start, count + 1)]
    largest = max(terms)
    return min(0., largest + math.log(sum(math.exp(x - largest) for x in terms)))


@lru_cache(maxsize=128)
def score_threshold(count, tests, alpha):
    for score in range(count + 1):
        if log_tail(score, count) <= math.log(alpha / tests):
            return score
    return count + 1  # Not enough possible evidence; always abstain.
