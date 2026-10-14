"""Live API contracts, input isolation, stream fidelity, and concurrency guards."""

from copy import deepcopy
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.live import (
    BusyError, LiveRequest, LiveService, StaleRunError, default_inputs, live_messages,
)
from token_auction_lab.walkthrough import Handler


class HeldEngine:
    """Synthetic fixture: hold one worker until the test permits completion."""
    def __init__(self, fail=False, reference=False, fail_no_ad=False):
        self.release = threading.Event()
        self.started = threading.Event()
        self.fail = fail
        self.reference = reference
        self.fail_no_ad = fail_no_ad
        self.requests = []

    def load(self):
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("Fixture was not released")

    def decode(self, token):
        return ["shuttle", "weekend", "courtyard"][token]

    def run(self, request, output, emit, retain):
        self.requests.append(request)
        p_a, p_b = np.array([.6, .3, .1]), np.array([.1, .3, .6])
        tape = RandomTape(.9, .2, .7, .4)
        allocation = TwoBidderAuction(p_a, p_b).resolve(
            request.advertisers[0].bid, request.advertisers[1].bid, tape
        )
        event = {**asdict(allocation), "type": "token", "step": 0, "tape": asdict(tape),
            "bid_a": request.advertisers[0].bid, "bid_b": request.advertisers[1].bid}
        emit({"type": "round_start", "manifest": {"mode": "synthetic_fixture"}})
        retain(0, [p_a, p_b] + ([np.array([.2, .7, .1])] if self.reference else []))
        emit(event)
        emit({"type": "validation_start"})
        if self.fail:
            raise AssertionError("deliberately failed cache check")
        if self.reference:
            emit({"type": "round_end", "mode": "synthetic_fixture", "text": "ad result"})
            emit({"type": "no_ad_event", "event": {"type": "round_start", "manifest": {
                "mode": "synthetic_fixture", "prompts": [{"text": "user prompt only"}]}}})
            emit({"type": "no_ad_event", "event": {"type": "token", "step": 0,
                "token_id": 1, "decoded_text_so_far": "weekend", "payment_a": 0, "payment_b": 0}})
            if self.fail_no_ad:
                raise AssertionError("deliberately failed no-ad cache check")
            emit({"type": "no_ad_event", "event": {"type": "round_end",
                "text": "weekend", "mode": "synthetic_fixture", "payment_a": 0, "payment_b": 0}})
        return {"text": self.decode(allocation.token_id), "mode": "synthetic_fixture"}


class LiveContractTests(unittest.TestCase):
    def test_every_editable_prompt_field_is_used_but_bids_are_not(self):
        payload = default_inputs()
        original = LiveRequest.parse(payload)
        payload["advertisers"][0]["bid"] = 17
        payload["advertisers"][1]["bid"] = .0123
        changed = LiveRequest.parse(payload)
        for side in (0, 1):
            self.assertEqual(live_messages(original, side), live_messages(changed, side))
        for side in (0, 1):
            system = live_messages(original, side)[0]["content"]
            for ad in original.advertisers:
                self.assertIn(ad.facts, system)
            self.assertIn(original.advertisers[side].objective, system)
            self.assertIn(original.instructions, system)
            self.assertEqual(live_messages(original, side)[1]["content"], original.prompt)
        edited = deepcopy(payload)
        edited["advertisers"][0]["objective"] = "Write about the bicycles first."
        request = LiveRequest.parse(edited)
        self.assertNotEqual(live_messages(request, 0), live_messages(changed, 0))
        self.assertEqual(live_messages(request, 1), live_messages(changed, 1))
        edited["advertisers"][0]["facts"] = "New explicit bicycle fact."
        request = LiveRequest.parse(edited)
        self.assertTrue(all("New explicit bicycle fact." in live_messages(request, i)[0]["content"] for i in (0, 1)))

    def test_validation_and_separate_zero_bid_baseline(self):
        invalid = [
            {"advertisers": []}, {"seed": 1.2}, {"max_tokens": 97},
            {"prompt": ""}, {"temperature": float("nan")}, {"seed": True},
        ]
        for change in invalid:
            with self.assertRaises(ValueError):
                LiveRequest.parse({**default_inputs(), **change})
        payload = default_inputs()
        payload["advertisers"][0]["bid"] = -1
        with self.assertRaises(ValueError):
            LiveRequest.parse(payload)
        for ad in payload["advertisers"]:
            ad["bid"] = 0
        request = LiveRequest.parse(payload)
        self.assertTrue(request.baseline)
        baseline = live_messages(request, None)
        self.assertEqual(baseline, [{"role": "user", "content": request.prompt}])
        self.assertNotIn(request.advertisers[0].facts, baseline[0]["content"])

    def test_no_ad_prompt_is_independent_of_every_ad_input(self):
        payload = default_inputs()
        original = live_messages(LiveRequest.parse(payload), None)
        payload["instructions"] = "AD INSTRUCTIONS SENTINEL"
        for i, ad in enumerate(payload["advertisers"]):
            ad.update(brand=f"BRAND {i}", facts="FACT SENTINEL", objective="OBJECTIVE SENTINEL", bid=99)
        self.assertEqual(live_messages(LiveRequest.parse(payload), None), original)
        self.assertEqual(default_inputs()["advertisers"][0]["brand"], "Bayview Hotel")
        self.assertIn("Fort Mason", default_inputs()["prompt"])

    def test_no_ad_probabilities_and_delta_stay_fixed_under_preview_bids(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(reference=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            run_id = service.start(default_inputs())
            service.worker.join(2)
            original = service.inspect(run_id, 0, token_id=0)
            preview = service.inspect(run_id, 0, token_id=0, a=3, b=1)
            self.assertAlmostEqual(original["rows"][0]["p_no_ad"], .2)
            self.assertAlmostEqual(original["rows"][0]["delta_q_no_ad"], .15)
            self.assertAlmostEqual(original["no_ad_comparison"]["tv_q"], .4)
            self.assertAlmostEqual(preview["rows"][0]["q"], .475)
            self.assertEqual(preview["rows"][0]["p_no_ad"], original["rows"][0]["p_no_ad"])
            self.assertEqual(preview["rows"][0]["delta_q_no_ad"], original["rows"][0]["delta_q_no_ad"])
            self.assertEqual(preview["no_ad_comparison"]["tv_q"], original["no_ad_comparison"]["tv_q"])
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "complete")
            self.assertEqual(snapshot["no_ad"]["result"]["text"], "weekend")
            self.assertEqual(snapshot["no_ad"]["events"][0]["payment_a"], 0)

    def test_no_ad_failure_preserves_both_outputs_and_original_ad_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(reference=True, fail_no_ad=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            service.start(default_inputs())
            service.worker.join(2)
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertEqual(snapshot["result"]["text"], "ad result")
            self.assertEqual(len(snapshot["events"]), 1)
            self.assertEqual(snapshot["no_ad"]["events"][0]["decoded_text_so_far"], "weekend")
            self.assertIsNone(snapshot["no_ad"]["result"])
            self.assertIn("no-ad cache", snapshot["error"])

    def test_one_worker_immutable_snapshot_and_replaced_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            service = LiveService(Path(tmp), engine)
            payload = default_inputs()
            first = service.start(payload)
            self.assertTrue(engine.started.wait(1))
            payload["advertisers"][0]["bid"] = 50
            payload["prompt"] = "Edited after the request started"
            try:
                with self.assertRaises(BusyError):
                    service.start(payload)
                self.assertEqual(service.snapshot()["inputs"]["advertisers"][0]["bid"], 1)
            finally:
                engine.release.set()
                service.worker.join(2)
            self.assertEqual(service.snapshot()["status"], "complete")
            self.assertEqual(engine.requests[0].advertisers[0].bid, 1)
            second = service.start(payload)
            service.worker.join(2)
            self.assertNotEqual(first, second)
            with self.assertRaises(StaleRunError):
                service.inspect(first, 0)
            with self.assertRaises(StaleRunError):
                service.snapshot(first)

    def test_inspection_uses_full_vectors_and_does_not_change_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            run_id = service.start(default_inputs())
            service.worker.join(2)
            before = service.snapshot()
            preview = service.inspect(run_id, 0, a=.6, b=1, limit=1)
            self.assertEqual(preview["outcome"]["branch"], "B")
            self.assertAlmostEqual(preview["outcome"]["payment_b"], .9)
            self.assertEqual(preview["total"], 3)
            exact = service.inspect(run_id, 0, token_id=1)
            self.assertAlmostEqual(exact["rows"][0]["p_a"], .3)
            self.assertAlmostEqual(exact["rows"][0]["p_b"], .3)
            self.assertEqual(before, service.snapshot())
            self.assertEqual(service.snapshot(after=1)["events"], [])

    def test_failed_validation_preserves_tokens_and_unlocks_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine(fail=True)
            engine.release.set()
            service = LiveService(Path(tmp), engine)
            failed = service.start(default_inputs())
            service.worker.join(2)
            snapshot = service.snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertEqual(len(snapshot["events"]), 1)
            self.assertIsNone(snapshot["result"])
            self.assertIn("deliberately failed", snapshot["error"])
            self.assertTrue(service.file(failed, "error.json").is_file())
            engine.fail = False
            service.start(default_inputs())
            service.worker.join(2)
            self.assertEqual(service.snapshot()["status"], "complete")

    def test_http_rejects_cross_origin_and_busy_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = HeldEngine()
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.live = LiveService(Path(tmp), engine)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/api/live/run"

            def post(origin=None):
                headers = {"Content-Type": "application/json"}
                if origin:
                    headers["Origin"] = origin
                return urlopen(Request(url, data=json.dumps(default_inputs()).encode(), headers=headers), timeout=3)

            try:
                with self.assertRaises(HTTPError) as rejected:
                    post("https://example.com")
                self.assertEqual(rejected.exception.code, 403)
                with post() as response:
                    self.assertEqual(response.status, 202)
                with self.assertRaises(HTTPError) as busy:
                    post()
                self.assertEqual(busy.exception.code, 409)
            finally:
                engine.release.set()
                if server.live.worker:
                    server.live.worker.join(2)
                server.shutdown()
                server.server_close()
                thread.join(2)


if __name__ == "__main__":
    unittest.main()
