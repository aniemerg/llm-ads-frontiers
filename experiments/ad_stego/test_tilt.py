"""The modified auction samples coded probabilities; old targets remain unchanged."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego import tilt, tilt_race
from experiments.ad_stego.capacity import entropy_split
from experiments.ad_stego.prototype import SyntheticTokenizer, decode
from experiments.ad_stego.synthetic import generate


class TiltTests(unittest.TestCase):
    def test_entropy_loss_is_branch_information(self):
        split = entropy_split(np.array([1., 0.]), np.array([0., 1.]))
        self.assertAlmostEqual(split["mixture_entropy_bits"], 1.)
        self.assertAlmostEqual(split["candidate_entropy_bits"], 0.)
        self.assertAlmostEqual(split["branch_information_bits"], 1.)
        split = entropy_split(np.array([.2, .8]), np.array([.2, .8]))
        self.assertAlmostEqual(split["branch_information_bits"], 0.)

    def test_zero_strength_and_point_mass_no_created_support(self):
        pa, pb = np.array([.5, .4, .1, 0]), np.array([.1, .4, .5, 0])
        qa, qb = tilt.transform(pa, pb, key=1, message=2, step=3, strength=0)
        np.testing.assert_allclose(qa, pa)
        np.testing.assert_allclose(qb, pb)
        for key in range(20):
            qa, qb = tilt.transform([0, 1, 0], [1, 0, 0], key=key, message=2, step=3)
            np.testing.assert_array_equal(qa, [0, 1, 0])
            np.testing.assert_array_equal(qb, [1, 0, 0])

    def test_coded_mixture_and_payments_follow_original_auction(self):
        pa, pb = np.array([.5, .3, .15, .05]), np.array([.05, .15, .3, .5])
        coded = tilt.transform(pa, pb, key=3, message=5, step=7)
        auction = TwoBidderAuction(*coded)
        rng = np.random.default_rng(501)
        counts, payments = np.zeros(4), np.zeros(2)
        for _ in range(10000):
            tape = RandomTape.draw(rng)
            chosen = tilt.sample(pa, pb, (1, 3), tape, key=3, message=5, step=7)
            self.assertEqual(chosen, auction.resolve(1, 3, tape))
            counts[chosen.token_id] += 1
            payments += chosen.payment_a, chosen.payment_b
        np.testing.assert_allclose(counts/10000, auction.mixture(1, 3), atol=.02)
        np.testing.assert_allclose(payments/10000, auction.expected_payments(1, 3), atol=.02)

    def test_exact_binomial_tail_and_impossible_threshold(self):
        self.assertAlmostEqual(math.exp(tilt.log_tail(1, 1)), .25)
        self.assertAlmostEqual(math.exp(tilt.log_tail(2, 2)), .0625)
        self.assertEqual(tilt.score_threshold(8, 10000, .01), 9)
        messages = np.arange(30000, dtype=np.uint64)
        hits = tilt.token_evidence(37, 815, messages, 0, 256)
        self.assertAlmostEqual(float(hits.mean()), .25, delta=.01)

    def test_two_spans_decode_without_dictionary(self):
        with patch("experiments.ad_stego.synthetic.sample", tilt.sample), patch(
                "experiments.ad_stego.synthetic.message_seed", tilt.message_seed):
            fixture = generate()
        result = decode(fixture["text"], SyntheticTokenizer(), key=1901, scheme="tilt")
        self.assertEqual([(r["start_token"], r["end_token"]) for r in result["spans"]],
                         [(8, 10), (76, 79)])

    def test_joint_null_and_two_text_only_payloads(self):
        scores = tilt_race.token_evidence(37, 815, np.arange(30000, dtype=np.uint64), 0, 256)
        self.assertAlmostEqual(float(scores.mean()), 1., delta=.035)
        self.assertAlmostEqual(float(np.mean(scores > -math.log(.01))), .01, delta=.003)
        with patch("experiments.ad_stego.synthetic.sample", tilt_race.sample), patch(
                "experiments.ad_stego.synthetic.message_seed", tilt_race.message_seed):
            fixture = generate()
        result = decode(fixture["text"], SyntheticTokenizer(), key=1901, scheme="tilt-race")
        self.assertEqual([(r["start_token"], r["end_token"]) for r in result["spans"]],
                         [(8, 10), (76, 79)])

    def test_joint_sampler_uses_coded_distributions_for_payments(self):
        pa, pb = np.array([.5, .3, .15, .05]), np.array([.05, .15, .3, .5])
        rng = np.random.default_rng(502)
        counts, expected = np.zeros(4), np.zeros(4)
        for _ in range(8000):
            key = int(rng.integers(0, 1 << 63))
            coded = tilt.transform(pa, pb, key=key, message=5, step=7)
            tape = RandomTape.draw(rng)
            actual = tilt_race.sample(pa, pb, (1, 3), tape, key=key, message=5, step=7)
            reference = TwoBidderAuction(*coded).resolve(1, 3, tape)
            self.assertEqual((actual.branch, actual.payment_a, actual.payment_b),
                             (reference.branch, reference.payment_a, reference.payment_b))
            counts[actual.token_id] += 1
            expected += TwoBidderAuction(*coded).mixture(1, 3)
        np.testing.assert_allclose(counts/8000, expected/8000, atol=.02)


if __name__ == "__main__":
    unittest.main()
