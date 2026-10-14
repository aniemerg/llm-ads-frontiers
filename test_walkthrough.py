"""Evidence fidelity and cross-language checks for the local walkthrough."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.walkthrough import ROOT, evidence_path, load_evidence


class WalkthroughTests(unittest.TestCase):
    def test_archive_keeps_failed_runs_and_original_tokens(self):
        data = load_evidence()
        self.assertEqual(len(data["runs"]), 8)
        self.assertEqual(len(data["diagnostics"]), 16)
        incomplete = [r for r in data["runs"] if r["result"] is None]
        self.assertEqual(len(incomplete), 2)
        for run in data["runs"]:
            original = [
                json.loads(line)
                for line in (ROOT / run["path"] / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(run["events"], [e for e in original if e["type"] == "token"])
            if run["result"]:
                self.assertEqual(run["events"][-1]["decoded_text_so_far"], run["result"]["text"])
                self.assertEqual([e["token_id"] for e in run["events"]], run["result"]["token_ids"])
        equal = next(r for r in data["runs"] if r["group"] == "validated" and r["result"]["bid_a"] == r["result"]["bid_b"] == 1)
        self.assertIn("free luggage storage", equal["result"]["text"])

    def test_evidence_routes_expose_only_archive_and_named_source(self):
        self.assertEqual(evidence_path("MECHANISM.md"), ROOT / "MECHANISM.md")
        self.assertIsNotNone(evidence_path("smoke_results/mps-2026-10-13/validated/audit.txt"))
        for name in (".env", ".git/config", "../../.ssh/id_rsa", "/etc/passwd", "walkthrough/app.mjs", "artifacts/private.json"):
            self.assertIsNone(evidence_path(name))

    @unittest.skipUnless(shutil.which("node"), "Node is only needed for the browser-math cross-check")
    def test_browser_counterfactual_matches_python_reference(self):
        rng = np.random.default_rng(583)
        fixtures = []
        auctions = [TwoBidderAuction([.6, .3, .1], [.1, .3, .6])]
        auctions.extend(TwoBidderAuction(rng.dirichlet(np.ones(17)), rng.dirichlet(np.ones(17))) for _ in range(12))
        for auction in auctions:
            for tape in [RandomTape(.9, .2, .7, .4), RandomTape(.1, .5, .5, .9), RandomTape.draw(rng)]:
                recorded = asdict(auction.resolve(1, 1, tape))
                recorded["tape"] = asdict(tape)
                for a, b in [(0, 1), (1, 0), (.6, 1), (.7, 1), (2 / 3, 1), (1, 1), (3, .05), (.05, 3)]:
                    fixtures.append({"event": recorded, "a": a, "b": b, "expected": asdict(auction.resolve(a, b, tape))})
        script = """
import fs from 'fs';
import assert from 'assert';
import {frozenOutcome, weights} from './walkthrough/mechanism.mjs';
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const f of fixtures) {
  const actual = frozenOutcome(f.event, f.a, f.b);
  assert.strictEqual(actual.token_id, f.expected.token_id);
  assert.strictEqual(actual.branch, f.expected.branch);
  for (const key of ['payment_a', 'payment_b']) {
    assert.ok(Math.abs(actual[key] - f.expected[key]) < 1e-12);
  }
}
for (const bids of [[0,0],[-1,1],[NaN,1],[1,Infinity]]) assert.throws(() => weights(...bids));
"""
        result = subprocess.run([shutil.which("node"), "--experimental-modules", "--input-type=module", "-e", script], input=json.dumps(fixtures), text=True, cwd=Path(__file__).parent, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
