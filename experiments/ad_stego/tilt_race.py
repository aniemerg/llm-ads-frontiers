"""Combine pre-auction preference with exponential candidate sampling.

Unlike the reference distribution-preserving candidate-only scheme, this mode
changes auction input probabilities. Branches/payments use those coded inputs.
"""

import numpy as np

from experiments.ad_stego import race, tilt
from experiments.ad_stego.prototype import _digest

PROTOCOL = "coordinate-codeword-green4-exp-v1"
RACE_DOMAIN = 0xD1310BA698DFB5AC


def message_seed(span, surface):
    return _digest(f"{PROTOCOL}|{span.start}|{span.end}|{surface}")


def sample(p_a, p_b, bids, tape, *, key, message, step):
    coded = tilt.transform(p_a, p_b, key=key, message=message, step=step)
    return race.sample(*coded, bids, tape, key=key,
                       message=int(message) ^ RACE_DOMAIN, step=step)


def token_evidence(token, key, messages, step, size):
    del size
    preferred = tilt.green(key, messages, step, token)
    domain = np.asarray(messages, dtype=np.uint64) ^ np.uint64(RACE_DOMAIN)
    v = 1. - race.uniform(key, domain, step, token)
    # Under the ideal null, the preferred indicator is Bernoulli(pi), independent
    # of v~Uniform(0,1). These two intervals yield an exactly uniform null p-value.
    pi = tilt.GREEN_PROBABILITY
    p = np.where(preferred, pi*v, pi + (1-pi)*v)
    return -np.log(p)
