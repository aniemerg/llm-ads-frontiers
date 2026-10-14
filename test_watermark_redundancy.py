"""Repeated watermark evidence, using synthetic probabilities and cached tokenizer."""

import unittest

import numpy as np

from auction import RandomTape
from experiments.ad_stego import word_redundant as wr
from token_auction_lab import watermark as wm


class RedundantRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = wm.get_tokenizer()
        except (ImportError, ValueError):
            raise unittest.SkipTest("Requires cached tokenizer; never downloads it.")

    def test_empty_lane_is_borrowed_then_both_records_repeat(self):
        encoder = wm.Encoder(("Paradigm House", "Fort Mason Express"))
        text = "Paradigm House" + " plan" * 30
        ids = self.tokenizer.encode(text)
        for count in (32, 48):
            if count == 48:
                text += " plan" * 16
                ids = self.tokenizer.encode(text)
            payload = encoder.at_step(len(ids), ids, text)
            self.assertIn("Paradigm House", payload["text"])
            self.assertEqual(len(payload["text"].split()), 16)
        text += " Fort Mason Express" + " plan" * 13
        ids = self.tokenizer.encode(text)
        encoder.at_step(len(ids), ids, text)
        lanes = []
        for _ in range(6):
            text += " plan" * 16
            ids = self.tokenizer.encode(text)
            payload = encoder.at_step(len(ids), ids, text)
            lanes.append(payload["lane"])
        self.assertEqual(lanes, [0, 1, 0, 1, 0, 1])
        self.assertEqual(len(encoder.records), 2)
        self.assertTrue(all(len(r["carriers"]) >= 3 for r in encoder.records))
        self.assertEqual(ids, self.tokenizer.encode(text))

    def test_both_overlapping_regions_decode_from_repeated_evidence_and_text_only(self):
        words = ["plan"] * 32
        words[7:9] = ["Paradigm", "House"]
        words[14:17] = ["Fort", "Mason", "Express"]
        text = " ".join(words)
        ids = self.tokenizer.encode(text)
        encoder = wm.Encoder(("Paradigm House", "Fort Mason Express"))
        pool = [" city", " trip", " path", " place", " plan", " visit", " guide", " day",
                " road", " view", " coast", " town", " route", " space", " hotel", " bus",
                " train", " water", " stay", " walk", " travel", " time", " local", " park"]
        support = [self.tokenizer.encode(w)[0] for w in pool
                   if len(self.tokenizer.encode(w)) == 1]
        p = np.zeros(wm.VOCABULARY_SIZE)
        p[support] = 1 / len(support)
        rng = np.random.default_rng(712)
        for i in range(150):
            payload = encoder.at_step(len(ids), ids, text)
            self.assertIsNotNone(payload)
            # Force 1/4 of tokens without watermarking, mimicking a noisy channel.
            token = (support[4] if i % 4 == 0 else wm.sample(
                (p, p), (1, 1), RandomTape(*rng.random(4)), payload, len(ids)).token_id)
            ids.append(token)
            text = self.tokenizer.decode(ids)
        decoded = wm.decode_response(text)
        regions = [r for r in decoded["spans"] if r["protocol"] == wr.PROTOCOL]
        self.assertEqual([(r["start_word"], r["end_word"]) for r in regions], [(0, 16), (8, 24)])
        self.assertTrue(all(r["scored_tokens"] > 64 for r in regions))
        self.assertEqual(len({r["text"] for r in regions}), 2)
        from test_watermark_isolation import isolated_decode
        isolated = isolated_decode(text)
        self.assertEqual(isolated["result"], decoded)
        self.assertFalse(isolated["denied_accesses"])


if __name__ == "__main__":
    unittest.main()
