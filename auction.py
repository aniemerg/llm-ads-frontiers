"""Two-bidder, one-token linear auction: reference math, not a serving stack.

One model/tokenizer must define both probability vectors over the same token IDs.
Distributions and the shared text prefix are fixed for each call. Bids do not enter
model prompts. The coupling splits common probability mass from disagreement,
then runs a random-threshold contest only on disagreement.

This module does not claim truthful bidding for arbitrary advertiser utilities,
end-to-end text guarantees, production currency accounting, or >2 bidders.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]


def _distribution(values: Sequence[float] | FloatVector) -> FloatVector:
    p = np.array(values, dtype=np.float64, copy=True)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("A distribution must be a nonempty one-dimensional vector.")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    total = float(p.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities must sum to one; got {total}.")
    p /= total  # Only correct tiny input rounding error, not arbitrary scores.
    p.flags.writeable = False
    return p


def _bids(a: float, b: float) -> tuple[float, float, float]:
    a, b = float(a), float(b)
    if not all(math.isfinite(x) and x >= 0.0 for x in (a, b)):
        raise ValueError("Bids must be finite and nonnegative.")
    total = a + b
    if not math.isfinite(total):
        raise ValueError("Bid sum overflow; rescale the credit unit.")
    if total == 0.0:
        raise ValueError("All-zero bids: use the separate no-ad path, not this auction.")
    return a, b, total


def _cdf(weights: FloatVector) -> FloatVector:
    total = float(weights.sum())
    if total <= 0.0:
        return np.array([], dtype=np.float64)
    result = np.cumsum(weights / total)
    result[-1] = 1.0
    return result


def _pick(cdf: FloatVector, u: float) -> int:
    if len(cdf) == 0:
        raise RuntimeError("Attempted to sample zero probability mass.")
    return int(np.searchsorted(cdf, u, side="right"))


@dataclass(frozen=True)
class RandomTape:
    """Independent uniforms. Reuse this object for FIXED-PREFIX counterfactuals."""

    branch: float
    token_a: float
    token_b: float
    contest: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and 0.0 <= x < 1.0
                   for x in (self.branch, self.token_a, self.token_b)):
            raise ValueError("Branch/token uniforms must lie in [0, 1).")
        if not math.isfinite(self.contest) or not 0.0 < self.contest < 1.0:
            raise ValueError("Contest uniform must lie strictly between zero and one.")

    @classmethod
    def draw(cls, rng: np.random.Generator) -> RandomTape:
        values = rng.random(4)
        # Finite-precision sampling: exclude the zero endpoint for division safety.
        if values[3] == 0.0:
            values[3] = np.nextafter(0.0, 1.0)
        return cls(*(float(x) for x in values))


@dataclass(frozen=True)
class Allocation:
    token_id: int
    branch: Literal["shared", "A", "B"]
    payment_a: float
    payment_b: float
    candidate_a: int | None
    candidate_b: int | None


class TwoBidderAuction:
    """Prepare one token step; resolve it at arbitrary nonnegative bid pairs."""

    def __init__(self, p_a: Sequence[float], p_b: Sequence[float]) -> None:
        self.p_a, self.p_b = _distribution(p_a), _distribution(p_b)
        if self.p_a.shape != self.p_b.shape:
            raise ValueError("Distributions must use the same vocabulary shape.")
        common = np.minimum(self.p_a, self.p_b)
        residual_a, residual_b = self.p_a - common, self.p_b - common
        self.shared_mass = min(1.0, max(0.0, float(common.sum())))
        self.disagreement = 0.5 * float(np.abs(self.p_a - self.p_b).sum())
        self._shared_cdf = _cdf(common)
        self._a_cdf, self._b_cdf = _cdf(residual_a), _cdf(residual_b)

    def mixture(self, bid_a: float, bid_b: float) -> FloatVector:
        a, b, total = _bids(bid_a, bid_b)
        return (a / total) * self.p_a + (b / total) * self.p_b

    def resolve(self, bid_a: float, bid_b: float, tape: RandomTape) -> Allocation:
        a, b, total = _bids(bid_a, bid_b)
        if tape.branch < self.shared_mass or self.disagreement == 0.0:
            token = _pick(self._shared_cdf, tape.token_a)
            return Allocation(token, "shared", 0.0, 0.0, None, None)

        token_a = _pick(self._a_cdf, tape.token_a)
        token_b = _pick(self._b_cdf, tape.token_b)
        u = tape.contest
        if u < a / total:
            # Critical A bid: solve u = a/(a+b) for a, holding b fixed.
            price = b * (u / (1.0 - u))
            return Allocation(token_a, "A", price, 0.0, token_a, token_b)
        # Critical B bid: solve u = a/(a+b) for b, holding a fixed.
        price = a * ((1.0 - u) / u)
        return Allocation(token_b, "B", 0.0, price, token_a, token_b)

    def expected_payments(self, bid_a: float, bid_b: float) -> tuple[float, float]:
        """Closed-form expectations of resolve(), useful as Monte Carlo oracles.

        E[payment_i] = d*s*[log(1+b_i/s) - b_i/(b_i+s)], for s>0,
        where d=TV(p_A,p_B) and s is the rival bid. A sole bidder pays zero.
        Small ratios use a series to avoid cancellation.
        """
        a, b, total = _bids(bid_a, bid_b)

        def charge(own: float, rival: float) -> float:
            if own == 0.0 or rival == 0.0 or self.disagreement == 0.0:
                return 0.0
            ratio = own / rival
            if ratio < 1e-4:
                factor = sum(((-1.0) ** n) * ((n - 1.0) / n) * ratio**n
                             for n in range(2, 9))
            elif math.isfinite(ratio):
                factor = math.log1p(ratio) - own / total
            else:
                factor = math.log(total) - math.log(rival) - own / total
            return self.disagreement * rival * max(0.0, factor)

        return charge(a, b), charge(b, a)
