"""Tests for the stronger sampler and its independent text-only decoder."""

import unittest
from unittest.mock import patch

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego import race
from experiments.ad_stego.prototype import SyntheticTokenizer, decode
from experiments.ad_stego.synthetic import generate


class RaceTests(unittest.TestCase):
    def test_fresh_key_mixture_payments_and_residual_independence(self):
        pa = np.array([.4, .2, .15, .15, .05, .05])
        pb = pa[::-1].copy()
        auction = TwoBidderAuction(pa, pb)
        rng = np.random.default_rng(509)
        counts = np.zeros(6)
        candidates = np.zeros((6, 6))
        for i in range(12000):
            tape = RandomTape.draw(rng)
            chosen = race.sample(pa, pb, (1, 3), tape, key=i, message=71, step=0)
            reference = auction.resolve(1, 3, tape)
            self.assertEqual((chosen.branch, chosen.payment_a, chosen.payment_b),
                             (reference.branch, reference.payment_a, reference.payment_b))
            counts[chosen.token_id] += 1
            if chosen.branch != "shared":
                candidates[chosen.candidate_a, chosen.candidate_b] += 1
                self.assertGreater(pa[chosen.candidate_a], pb[chosen.candidate_a])
                self.assertGreater(pb[chosen.candidate_b], pa[chosen.candidate_b])
        np.testing.assert_allclose(counts / counts.sum(), auction.mixture(1, 3), atol=.015)
        common = np.minimum(pa, pb)
        expected = np.outer(pa - common, pb - common) / (1 - common.sum()) ** 2
        np.testing.assert_allclose(candidates / candidates.sum(), expected, atol=.025)

    def test_candidates_fixed_across_bid_switch_and_zero_mass_excluded(self):
        pa, pb = [.6, .3, .1, 0], [.1, .3, .6, 0]
        tape = RandomTape(.99, .1, .9, .4)
        chosen = [race.sample(pa, pb, (a, 1), tape, key=3, message=5, step=7)
                  for a in (0, .4, 1, 4)]
        self.assertEqual(len({(x.candidate_a, x.candidate_b) for x in chosen}), 1)
        self.assertEqual([x.branch for x in chosen], ["B", "B", "A", "A"])
        self.assertNotIn(3, [x.token_id for x in chosen])

    def test_two_spans_from_text_without_brand_list(self):
        with patch("experiments.ad_stego.synthetic.sample", race.sample), patch(
                "experiments.ad_stego.synthetic.message_seed", race.message_seed):
            fixture = generate()
        result = decode(fixture["text"], SyntheticTokenizer(), key=1901, scheme="race")
        self.assertEqual([(r["start_token"], r["end_token"]) for r in result["spans"]],
                         [(8, 10), (76, 79)])

    def test_null_score_and_point_mass_have_no_information(self):
        scores = race.token_evidence(37, 815, np.arange(30000, dtype=np.uint64), 0, 256)
        self.assertAlmostEqual(float(scores.mean()), 1., delta=.035)
        self.assertAlmostEqual(float(np.mean(scores > -np.log(.01))), .01, delta=.003)
        for key in range(30):
            chosen = race.sample([0, 1, 0], [0, 1, 0], (1, 1),
                                 RandomTape(.5, .3, .7, .4), key=key, message=5, step=0)
            self.assertEqual(chosen.token_id, 1)


if __name__ == "__main__":
    unittest.main()
