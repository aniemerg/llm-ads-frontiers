"""Synthetic probabilities with the real cached tokenizer; no model-quality claim."""

import json
import re
import shutil
import subprocess
import unittest
from html import unescape
from pathlib import Path

import numpy as np

from auction import RandomTape
from experiments.ad_stego import word_window as ww
from token_auction_lab import watermark as wm


class WordRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = wm.get_tokenizer()
        except (ImportError, ValueError):
            raise unittest.SkipTest("Requires cached tokenizer; never downloads it.")

    def prefix(self):
        words = ["plan"] * 48
        words[7:9] = ["Paradigm", "House"]
        words[14:17] = ["Fort", "Mason", "Express"]
        return " ".join(words)

    def test_overlap_cap_fifo_and_once_per_brand(self):
        text = self.prefix()
        encoder = wm.WordWindowEncoder(("Paradigm House", "Fort Mason Express"))
        ids = self.tokenizer.encode(text)
        first = encoder.at_step(len(ids), ids, text)
        self.assertEqual((first["start_word"], first["end_word"]), (0, 16))
        self.assertIn("Paradigm House", first["text"])
        text += " plan" * 48
        ids = self.tokenizer.encode(text)
        second = encoder.at_step(len(ids), ids, text)
        self.assertEqual((second["start_word"], second["end_word"]), (8, 24))
        self.assertIn("Fort Mason Express", second["text"])
        self.assertLess(second["start_word"], first["end_word"])
        text += " Paradigm House Fort Mason Express" + " plan" * 43
        ids = self.tokenizer.encode(text)
        self.assertIsNone(encoder.at_step(len(ids), ids, text))
        self.assertEqual(len(encoder.records), 2)
        self.assertTrue(all(len(r["text"].split()) <= 16 for r in encoder.records))
        fresh = wm.WordWindowEncoder(encoder.brands)
        self.assertIsNotNone(fresh.at_step(len(ids), ids, text))

    def test_identical_region_is_encoded_once(self):
        text = "Paradigm House Fort Mason Express" + " plan" * 43
        ids = self.tokenizer.encode(text)
        encoder = wm.WordWindowEncoder(("Paradigm House", "Fort Mason Express"))
        encoder.at_step(len(ids), ids, text)
        self.assertEqual(len(encoder.records), 1)
        self.assertFalse(encoder.queue)
        self.assertEqual(len(encoder.scheduled_brands), 2)

    def test_synthetic_recovery_with_local_token_merge_and_overlap(self):
        text = self.prefix()
        ids = self.tokenizer.encode(text)
        encoder = wm.WordWindowEncoder(("Paradigm House", "Fort Mason Express"))
        pool = [" city", " trip", " path", " place", " plan", " visit", " guide", " day",
                " road", " view", " coast", " town", " route", " space", " hotel", " bus",
                " train", " water", " stay", " walk", " travel", " time", " local", " park"]
        support = [self.tokenizer.encode(w)[0] for w in pool
                   if len(self.tokenizer.encode(w)) == 1]
        p = np.zeros(wm.VOCABULARY_SIZE)
        p[support] = 1 / len(support)
        rng = np.random.default_rng(921)
        forced = self.tokenizer.encode(" water") + self.tokenizer.encode("front")
        self.assertNotEqual(forced, self.tokenizer.encode(" waterfront"))
        for i in range(110):
            payload = encoder.at_step(len(ids), ids, text)
            if i in (15, 16):
                token = forced[i - 15]
            elif payload:
                allocation = wm.sample(
                    (p, p), (1, 1), RandomTape(*rng.random(4)), payload, len(ids))
                token = allocation.token_id
            else:
                token = int(rng.choice(support))
            ids.append(token)
            text = self.tokenizer.decode(ids)
        result = wm.decode_response(text)
        regions = [s for s in result["spans"] if s["protocol"] == ww.PROTOCOL]
        self.assertEqual([(r["start_word"], r["end_word"]) for r in regions], [(0, 16), (8, 24)])
        self.assertFalse(encoder.finish(text, ids)["canonical_roundtrip"])
        self.assertTrue(all(len(s["literal_text"].split()) <= 16 for s in result["spans"]))
        # Independent canonical prefixes recover byte positions after the merge.
        encoding = self.tokenizer.backend.encode(text, add_special_tokens=False)
        self.assertEqual(text.encode(), self.tokenizer.decode(encoding.ids).encode())
        self.assertEqual(result["families"][0]["alpha"], 0.005)

    @unittest.skipUnless(shutil.which("node"), "Requires node for browser formatting test")
    def test_overlap_renderer_preserves_text_and_blends_color(self):
        source = Path(__file__).parent / "walkthrough/decode.mjs"
        script = """
import {highlighted} from SOURCE;
const text='one <two> 😀 three four';
const spans=[{start_utf16:0,end_utf16:16,start_token:0,end_token:4},
             {start_utf16:4,end_utf16:text.length,start_token:1,end_token:6}];
console.log(JSON.stringify({text,html:highlighted(text,spans)}));
""".replace("SOURCE", json.dumps(source.as_uri()))
        process = subprocess.run(
            ["node", "--experimental-modules", "--input-type=module", "-e", script],
            capture_output=True, text=True, check=True)
        result = json.loads(process.stdout)
        self.assertIn('watermark-overlap', result['html'])
        self.assertIn('background:#dae3f0', result['html'])
        self.assertEqual(unescape(re.sub(r"<[^>]+>", "", result["html"])), result["text"])


if __name__ == "__main__":
    unittest.main()
