"""Experimental coordinate codewords using exponential-race candidate sampling.

The ideal independent-key ensemble samples the original probabilities exactly.
The finite counter mixer is not a cryptographic PRF or a fixed-key guarantee.
"""

from dataclasses import replace

import numpy as np

from auction import TwoBidderAuction
from experiments.ad_stego.prototype import MASK64, _digest, _mix

PROTOCOL = "coordinate-codeword-exp-v1"


def message_seed(span, surface):
    return _digest(f"{PROTOCOL}|{span.start}|{span.end}|{surface}")


def uniform(key, messages, step, tokens):
    base = _mix(np.asarray(messages, dtype=np.uint64) ^ np.uint64(key & MASK64))
    base = base ^ _mix(np.uint64(step) ^ np.uint64(0x243F6A8885A308D3))
    base = base ^ _mix(np.asarray(tokens, dtype=np.uint64) ^ np.uint64(0x13198A2E03707344))
    return ((_mix(base ^ np.uint64(0xA4093822299F31D0)) >> 12).astype(np.float64) + .5) / (1 << 52)


def _draw(weights, u):
    # Exponential clocks are random, so this is categorical probability sampling,
    # not greedy selection of the most probable token. Zero-mass tokens cannot win.
    clocks = np.full(len(weights), np.inf)
    with np.errstate(over="ignore"):
        np.divide(-np.log(u), weights, out=clocks, where=weights > 0)
    return int(np.argmin(clocks))


def sample(p_a, p_b, bids, tape, *, key, message, step):
    auction = TwoBidderAuction(p_a, p_b)
    allocation = auction.resolve(*bids, tape)
    common = np.minimum(auction.p_a, auction.p_b)
    u = uniform(key, np.uint64(message), step, np.arange(len(p_a), dtype=np.uint64))
    if allocation.branch == "shared":
        return replace(allocation, token_id=_draw(common, u))
    # Residual supports are disjoint. Under ideal independent per-token uniforms,
    # their candidate draws are independent despite using one token-level field.
    a = _draw(auction.p_a - common, u)
    b = _draw(auction.p_b - common, u)
    return replace(allocation, token_id=a if allocation.branch == "A" else b,
                   candidate_a=a, candidate_b=b)


def token_evidence(token, key, messages, step, size):
    del size  # The keyed uniform at an observed token needs no vocabulary scan.
    u = uniform(key, messages, step, token)
    return -np.log1p(-u)
