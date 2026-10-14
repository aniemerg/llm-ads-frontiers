"""Integration checks; replay fixtures are not new live model runs."""

import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab import watermark as wm
from token_auction_lab.live import LiveRequest, LiveService, default_inputs, live_messages
from token_auction_lab.walkthrough import Handler


class WatermarkIntegrationTests(unittest.TestCase):
    def test_option_stays_out_of_prompts_and_rejects_zero_bids(self):
        defaults = default_inputs()
        enabled = LiveRequest.parse({**defaults, "watermark": True})
        disabled = LiveRequest.parse({**defaults, "watermark": False})
        self.assertFalse(disabled.watermark)
        for side in (0, 1, None):
            self.assertEqual(live_messages(enabled, side), live_messages(disabled, side))
        with self.assertRaises(ValueError):
            LiveRequest.parse({**defaults, "watermark": "true"})
        for ad in defaults["advertisers"]:
            ad["bid"] = 0
        with self.assertRaises(ValueError):
            LiveRequest.parse({**defaults, "watermark": True})

    def test_coded_inspector_replays_candidate_field_and_charges_at_unequal_bids(self):
        raw = [np.array([0.6, 0.3, 0.1]), np.array([0.1, 0.3, 0.6])]
        payload = dict(key=1901, message_seed=1234567, start_token=2, end_token=3)
        tape = RandomTape(0.9, 0.2, 0.7, 0.4)
        actual = wm.sample(raw, (2, 1), tape, payload, 0)
        event = {
            **asdict(actual),
            "bid_a": 2,
            "bid_b": 1,
            "tape": asdict(tape),
            "watermark": payload,
            "selection_rule": "watermark",
        }

        class Engine:
            def decode(self, token):
                return str(token)

        service = LiveService(engine=Engine())
        service.current = {"run_id": "synthetic", "events": [event], "vectors": {0: raw}}
        for bids in ((2, 1), (1, 3), (0, 1), (1, 0)):
            result = service.inspect("synthetic", 0, *bids)
            expected = wm.sample(raw, bids, tape, payload, 0)
            for field in ("token_id", "branch", "payment_a", "payment_b"):
                self.assertEqual(result["outcome"][field], getattr(expected, field))
            coded = wm.coded_distributions(raw, payload, 0)
            rows = {r["token_id"]: r for r in result["rows"]}
            for token in rows:
                self.assertEqual(rows[token]["p_a"], raw[0][token])
                self.assertEqual(rows[token]["coded_p_a"], coded[0][token])
                self.assertAlmostEqual(
                    rows[token]["q"], TwoBidderAuction(*coded).mixture(*bids)[token]
                )
                self.assertAlmostEqual(
                    rows[token]["actual_q"], TwoBidderAuction(*coded).mixture(2, 1)[token]
                )
        self.assertEqual(service.current["events"][0], event)

    def test_decoder_endpoint_accepts_only_text_and_blocks_foreign_origin(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/watermark/decode"

            def send(payload, origin=None):
                headers = {"Content-Type": "application/json"}
                if origin:
                    headers["Origin"] = origin
                return urlopen(Request(url, json.dumps(payload).encode(), headers), timeout=5)

            with patch.object(
                wm, "decode_response", return_value={"status": "insufficient_evidence"}
            ) as decode:
                with send({"text": " exact text\n"}) as response:
                    self.assertEqual(json.load(response)["status"], "insufficient_evidence")
                decode.assert_called_once_with(" exact text\n")
                for payload, origin, code in [
                    ({"text": "x", "brands": ["Secret"]}, None, 400),
                    ({"text": "x"}, "https://elsewhere.example", 403),
                ]:
                    with self.assertRaises(HTTPError) as error:
                        send(payload, origin)
                    self.assertEqual(error.exception.code, code)
                self.assertEqual(decode.call_count, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_round_preserves_continuation_and_saves_watermark_even_on_cache_failure(self):
        try:
            import test_decoder
            from token_auction_lab.decoder import SharedPrefixDecoder
            from token_auction_lab.live import InferenceEngine
            from token_auction_lab.runner import RoundConfig, run_round
        except ImportError:
            self.skipTest("Install inference dependencies.")
        if test_decoder.torch is None:
            self.skipTest("Install inference dependencies.")
        fixture = test_decoder.DecoderTests()
        fixture.setUp()

        class ChatTokenizer(test_decoder.TestTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)

            def encode(self, text, **kwargs):
                return [1, 2]

        engine = InferenceEngine()
        engine.model, engine.tokenizer, engine.metadata = fixture.model, ChatTokenizer(), {}
        with patch("token_auction_lab.runner.run_round", return_value={}) as run:
            engine.run(
                LiveRequest.parse({**default_inputs(), "watermark": True}),
                Path("unused"),
                None,
                None,
            )
            config = run.call_args.args[3]
            self.assertEqual(config.continuation_tokens, 3)
            self.assertEqual(config.continuation_threshold, 0.98)
            self.assertTrue(config.watermark)
        self.assertEqual(
            RoundConfig(
                watermark=True, watermark_brands=("A", "B"), continuation_tokens=3
            ).continuation_tokens,
            3,
        )
        payload = dict(key=1901, message_seed=123, start_token=0, end_token=1, text="synthetic")

        class SyntheticEncoder:
            brands = ("A", "B")

            def __init__(self, brands):
                pass

            def at_step(self, step, generated, text):
                return payload

            def finish(self, text, visible):
                return {"canonical_roundtrip": True, "synthetic": True}

        for failed in (False, True):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(wm, "Encoder", SyntheticEncoder),
                patch.object(wm, "profile", return_value={}),
                patch.object(
                    wm,
                    "decode_response",
                    return_value={
                        "status": "insufficient_evidence",
                        "spans": [],
                        "coverage": "unknown",
                    },
                ) as decode,
            ):
                path = Path(tmp) / "run"
                events = []

                def emit(e):
                    if e["type"] == "token":
                        events.append(e)

                kwargs = dict(
                    mode="synthetic_model_test",
                    on_event=emit,
                    no_ad_prompt=("User only", [4, 5]),
                    should_stop=lambda: len(events) == 3,
                )
                args = (
                    fixture.model,
                    ChatTokenizer(),
                    [("A", [1]), ("B", [2, 3])],
                    RoundConfig(max_tokens=4, watermark=True, watermark_brands=("A", "B")),
                    path,
                    {},
                )
                if failed:
                    with patch.object(
                        SharedPrefixDecoder,
                        "check_cache",
                        side_effect=AssertionError("preserved failure"),
                    ):
                        with self.assertRaisesRegex(AssertionError, "preserved failure"):
                            run_round(*args, **kwargs)
                else:
                    result = run_round(*args, **kwargs)
                    self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])
                    self.assertEqual(result["token_ids"], [e["token_id"] for e in events])
                    self.assertIn(result["stop_reason"], ("user_stop", "eos"))
                self.assertTrue((path / "watermark-decoded.json").is_file())
                self.assertTrue((path / "watermark.json").is_file())
                decode.assert_called_once_with((path / "response.txt").read_text())
                self.assertTrue(all(e["selection_rule"] == "watermark" for e in events))


class CachedWatermarkReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = wm.get_tokenizer()
        except (ImportError, ValueError):
            raise unittest.SkipTest("Requires the cached pinned tokenizer, never downloads it.")

    def test_saved_success_false_positive_and_short_failure_remain_reproducible(self):
        expected = {"success": [(91, 94), (165, 168)], "false-positive": [(80, 102)], "short": []}
        for example in wm.examples():
            result = wm.decode_legacy_response(example["text"])
            self.assertEqual(result["coverage"], "unknown")
            self.assertEqual(
                [(s["start_token"], s["end_token"]) for s in result["spans"]],
                expected[example["id"]],
            )
            for span in result["spans"]:
                self.assertEqual(
                    example["text"][span["start_char"] : span["end_char"]], span["literal_text"]
                )
                self.assertEqual(span["literal_text"], span["text"])
            if example["id"] == "short":
                self.assertEqual(result["status"], "insufficient_evidence")

    def test_scheduler_matches_recorded_boundaries_and_repeated_mentions(self):
        example = wm.examples()[0]
        ids = self.tokenizer.encode(example["text"])
        encoder = wm.LegacyEncoder(("Paradigm House", "Fort Mason Express"))
        for step in range(0, len(ids), 64):
            encoder.at_step(step, ids[:step], self.tokenizer.decode(ids[:step]))
        audit = encoder.finish(example["text"], ids)
        self.assertTrue(audit["canonical_roundtrip"])
        self.assertEqual(
            [(r["start_token"], r["end_token"], r["carrier_start"]) for r in audit["records"]],
            [(91, 94, 128), (165, 168, 192)],
        )
        # The repeated mention no longer starts a third carrier block.
        self.assertTrue(all(r["complete"] for r in audit["records"]))
        self.assertIsNone(encoder.active)
        self.assertFalse(audit["pending_mentions"])
        self.assertEqual(audit["mention_policy"], wm.MENTION_POLICY)
        self.assertFalse(audit["prefix_alignment_failures"])
        # An observed mismatch disables encoding instead of rebuilding either context.
        encoder.at_step(320, [999], example["text"])
        self.assertIsNone(encoder.active)
        self.assertEqual(encoder.alignment_failures, [320])
        encoder.at_step(384, ids, example["text"])
        self.assertIsNone(encoder.active)
        partial = encoder.finish(self.tokenizer.decode(ids[:200]), ids[:200])
        self.assertFalse(partial["records"][1]["complete"])

    def test_repeats_do_not_delay_second_brand_or_requeue_pending_brands(self):
        text = "Paradigm House. Paradigm House. Fort Mason Express." + " Plan" * 200
        ids = self.tokenizer.encode(text)
        encoder = wm.LegacyEncoder(("Fort Mason Express", "Paradigm House", "Paradigm House"))
        first = encoder.at_step(64, ids[:64], self.tokenizer.decode(ids[:64]))
        self.assertEqual(first["text"].strip(), "Paradigm House")
        self.assertEqual([r["text"].strip() for r in encoder.queue], ["Fort Mason Express"])
        second = encoder.at_step(128, ids[:128], self.tokenizer.decode(ids[:128]))
        self.assertEqual(second["text"].strip(), "Fort Mason Express")
        self.assertFalse(encoder.queue)
        self.assertIsNone(encoder.at_step(192, ids[:192], self.tokenizer.decode(ids[:192])))
        self.assertEqual(len(encoder.records), 2)
        # Each new response gets its own allowance; no cross-run suppression.
        fresh = wm.LegacyEncoder(encoder.brands)
        self.assertEqual(fresh.at_step(64, ids[:64], self.tokenizer.decode(ids[:64])), first)

    def test_limit_never_silently_truncates(self):
        result = wm.decode_response(" hello" * 4097)
        self.assertEqual(result["status"], "unsupported_length")
        self.assertEqual(result["spans"], [])


if __name__ == "__main__":
    unittest.main()
