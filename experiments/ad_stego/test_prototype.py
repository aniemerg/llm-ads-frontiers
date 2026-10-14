"""Independent channel, sampler, and text-only decoding checks."""

import math
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego.prototype import (
    Config,
    Span,
    SyntheticTokenizer,
    decode,
    gamma_log_tail,
    message_seed,
    sample,
    token_evidence,
)
from experiments.ad_stego.synthetic import generate


class SpanStegoTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SyntheticTokenizer()

    def test_text_only_recovers_two_exact_spans_without_brand_dictionary(self):
        fixture = generate()
        response = decode(fixture["text"], self.tokenizer, key=1901)
        self.assertEqual(response["status"], "detected")
        self.assertEqual([(r["start_token"], r["end_token"], r["text"])
                          for r in response["spans"]],
                         [(8, 10, "Paradigm House"), (76, 79, "Fort Mason Express")])
        self.assertEqual(response["coverage"], "unknown")
        self.assertTrue(fixture["shared_suffix_verified"])

    def test_two_occurrences_of_same_name_have_distinct_coordinates(self):
        fixture = generate(occurrences=((8, (0, 1)), (76, (0, 1))))
        response = decode(fixture["text"], self.tokenizer, key=1901)
        self.assertEqual([(r["start_token"], r["end_token"]) for r in response["spans"]],
                         [(8, 10), (76, 78)])

    def test_truncated_carrier_returns_only_completed_record(self):
        fixture = generate(tokens=160)
        response = decode(fixture["text"], self.tokenizer, key=1901)
        self.assertEqual([(r["start_token"], r["end_token"]) for r in response["spans"]],
                         [(8, 10)])
        self.assertEqual(response["coverage"], "unknown")

    def test_no_room_and_unmarked_and_wrong_key_do_not_claim_absence(self):
        for fixture, key in ((generate(tokens=100), 1901),
                             (generate(marked=False), 1901), (generate(), 1902)):
            response = decode(fixture["text"], self.tokenizer, key=key)
            self.assertEqual(response["spans"], [])
            self.assertEqual(response["coverage"], "unknown")

    def test_bound_surface_prevents_silent_pointer_drift(self):
        fixture = generate()
        ids = fixture["token_ids"][:]
        ids[8:10] = [5, 6]
        response = decode(self.tokenizer.decode(ids), self.tokenizer, key=1901)
        self.assertEqual([(r["start_token"], r["end_token"]) for r in response["spans"]],
                         [(76, 79)])
        self.assertNotEqual(message_seed(Span(8, 10), "Paradigm House"),
                            message_seed(Span(8, 10), "word000 word001"))

    def test_point_mass_has_no_carrier_capacity(self):
        response = decode(generate(entropy="point")["text"], self.tokenizer, key=1901)
        self.assertEqual(response["spans"], [])

    def test_validation_and_unsupported_input(self):
        for args in ((15, 15), (-1, 3), (0, 33), (4095, 4097)):
            with self.assertRaises(ValueError):
                Span(*args)
        with self.assertRaises(ValueError):
            Config(alpha=0)
        self.assertEqual(decode("not-in-vocabulary", self.tokenizer, key=1)["status"],
                         "unsupported_text")

    def test_mixture_and_payments_over_fresh_key_ensemble(self):
        p_a = np.array([.4, .2, .15, .15, .05, .05])
        p_b = np.array([.05, .05, .15, .15, .2, .4])
        auction = TwoBidderAuction(p_a, p_b)
        rng = np.random.default_rng(509)
        counts = np.zeros(6)
        payments = np.zeros(2)
        n = 12000
        for _ in range(n):
            tape = RandomTape.draw(rng)
            key = int(rng.integers(0, 1 << 63))
            chosen = sample(p_a, p_b, (1, 3), tape, key=key, message=71, step=0)
            baseline = auction.resolve(1, 3, tape)
            self.assertEqual(chosen.branch, baseline.branch)
            self.assertEqual((chosen.payment_a, chosen.payment_b),
                             (baseline.payment_a, baseline.payment_b))
            counts[chosen.token_id] += 1
            payments += [chosen.payment_a, chosen.payment_b]
        np.testing.assert_allclose(counts/n, auction.mixture(1, 3), atol=.015, rtol=0)
        np.testing.assert_allclose(payments/n, auction.expected_payments(1, 3),
                                   atol=.015, rtol=0)

    def test_fixed_tape_candidates_do_not_change_when_bids_change(self):
        pa, pb = [0.6, 0.3, 0.1], [0.1, 0.3, 0.6]
        tape = RandomTape(.99, .1, .9, .4)
        allocations = [sample(pa, pb, (a, 1), tape, key=3, message=5, step=7)
                       for a in (0, .4, 1, 4)]
        self.assertEqual(len({(a.candidate_a, a.candidate_b) for a in allocations}), 1)
        self.assertEqual([a.branch for a in allocations], ["B", "B", "A", "A"])
        self.assertAlmostEqual(allocations[-1].payment_a, 2/3)

    def test_null_score_and_gamma_tail(self):
        # Fresh candidate keys on fixed non-watermarked token, outside generation.
        messages = np.arange(30000, dtype=np.uint64)
        evidence = token_evidence(37, 815, messages, 0, 256)
        self.assertAlmostEqual(float(evidence.mean()), 1.0, delta=.035)
        self.assertAlmostEqual(float(np.mean(evidence >= -math.log(.01))), .01, delta=.003)
        self.assertAlmostEqual(gamma_log_tail(3, 1), -3)
        self.assertAlmostEqual(gamma_log_tail(3, 2), math.log(4)-3)


if __name__ == "__main__":
    unittest.main()
