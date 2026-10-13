"""Run: python -m unittest -v (NumPy required; no GPU or model download)."""
import math
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction


class AuctionTests(unittest.TestCase):
    def setUp(self):
        self.auction = TwoBidderAuction([0.6, 0.3, 0.1], [0.1, 0.3, 0.6])
        self.contested = RandomTape(0.9, 0.2, 0.7, 0.4)

    def test_normalization_and_endpoints(self):
        for a, b in [(1, 1), (0, 5), (7, 0), (100, 0.1)]:
            q = self.auction.mixture(a, b)
            self.assertAlmostEqual(float(q.sum()), 1.0)
            self.assertTrue((q >= 0).all())
        np.testing.assert_allclose(self.auction.mixture(1, 1), [0.35, 0.3, 0.35])
        np.testing.assert_allclose(self.auction.mixture(1, 0), self.auction.p_a)
        np.testing.assert_allclose(self.auction.mixture(0, 1), self.auction.p_b)

    def test_shared_branch_is_free(self):
        out = self.auction.resolve(100, 0.01, RandomTape(0.1, 0.5, 0.5, 0.9))
        self.assertEqual(out.branch, "shared")
        self.assertEqual((out.payment_a, out.payment_b), (0.0, 0.0))

    def test_identical_distributions_are_always_free(self):
        auction = TwoBidderAuction([0.2, 0.8], [0.2, 0.8])
        rng = np.random.default_rng(11)
        for _ in range(200):
            out = auction.resolve(1, 40, RandomTape.draw(rng))
            self.assertEqual(out.branch, "shared")
            self.assertEqual(out.payment_a + out.payment_b, 0.0)
        self.assertEqual(auction.expected_payments(1, 40), (0.0, 0.0))

    def test_concrete_threshold_fixture(self):
        # With equal bids and u=.4, A wins and pays 2/3, not the rival bid of 1.
        out = self.auction.resolve(1, 1, self.contested)
        self.assertEqual((out.token_id, out.branch), (0, "A"))
        self.assertAlmostEqual(out.payment_a, 2.0 / 3.0)
        self.assertEqual(out.payment_b, 0.0)
        self.assertEqual((out.candidate_a, out.candidate_b), (0, 2))

    def test_fixed_tape_single_switch_for_each_bidder(self):
        a_path = [self.auction.resolve(a, 1, self.contested).branch
                  for a in [0, .1, .6, .7, 1, 10]]
        self.assertEqual(a_path, ["B", "B", "B", "A", "A", "A"])
        b_path = [self.auction.resolve(1, b, self.contested).branch
                  for b in [0, .1, 1, 2, 10]]
        self.assertEqual(b_path, ["A", "A", "A", "B", "B"])

    def test_single_bidder_pays_zero(self):
        for a, b in [(0, 1), (1, 0)]:
            out = self.auction.resolve(a, b, self.contested)
            self.assertEqual(out.payment_a + out.payment_b, 0.0)
            self.assertEqual(self.auction.expected_payments(a, b), (0.0, 0.0))

    def test_scale_invariance_and_linear_price_scaling(self):
        original = self.auction.resolve(1, 1, self.contested)
        scaled = self.auction.resolve(25, 25, self.contested)
        self.assertEqual(original.token_id, scaled.token_id)
        self.assertAlmostEqual(scaled.payment_a, 25 * original.payment_a)
        np.testing.assert_allclose(self.auction.mixture(1, 1), self.auction.mixture(25, 25))

    def test_expected_payment_known_value_positive(self):
        # This catches the apparent sign reversal in Proposition 3.13's TV formula.
        expected = 0.5 * (math.log(2) - 0.5)
        for value in self.auction.expected_payments(1, 1):
            self.assertAlmostEqual(value, expected)
            self.assertGreater(value, 0.0)

    def test_fixed_prefix_probability_monotonicity(self):
        grid = [self.auction.mixture(a, 1) for a in [0, .1, .5, 1, 2, 10]]
        distances = [float(np.abs(q - self.auction.p_a).sum()) for q in grid]
        self.assertTrue(all(x >= y for x, y in zip(distances, distances[1:])))

    def test_monte_carlo_distribution_payments_and_bounds(self):
        rng = np.random.default_rng(12345)
        n = 40000
        counts = np.zeros(3)
        payments = np.zeros(2)
        for _ in range(n):
            out = self.auction.resolve(1, 1, RandomTape.draw(rng))
            counts[out.token_id] += 1
            payments += [out.payment_a, out.payment_b]
            self.assertTrue(0 <= out.payment_a <= 1 + 1e-12)
            self.assertTrue(0 <= out.payment_b <= 1 + 1e-12)
            self.assertFalse(out.payment_a > 0 and out.payment_b > 0)
        np.testing.assert_allclose(counts / n, self.auction.mixture(1, 1), atol=.012, rtol=0)
        np.testing.assert_allclose(payments / n, self.auction.expected_payments(1, 1),
                                   atol=.006, rtol=0)

    def test_invalid_inputs(self):
        for a, b in [(-1, 1), (float("nan"), 1), (1, float("inf")), (0, 0)]:
            with self.assertRaises(ValueError):
                self.auction.mixture(a, b)
        for a, b in [([1, 0], [1]), ([-.1, 1.1], [0, 1]), ([.2, .2], [.5, .5])]:
            with self.assertRaises(ValueError):
                TwoBidderAuction(a, b)
        for u in [0, 1, float("nan")]:
            with self.assertRaises(ValueError):
                RandomTape(.5, .5, .5, u)


if __name__ == "__main__":
    unittest.main()
