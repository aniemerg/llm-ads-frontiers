"""Synthetic combined-mode checks; tiny random weights do not measure text quality."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auction import RandomTape
from experiments.ad_stego import tilt
from token_auction_lab import watermark as wm
from token_auction_lab.live import LiveService


class WatermarkContinuationTests(unittest.TestCase):
    def test_text_only_decoder_keeps_carried_positions_in_fixed_blocks(self):
        from auction import TwoBidderAuction
        from experiments.ad_stego import tilt_race
        from experiments.ad_stego.prototype import Span, SyntheticTokenizer, decode
        from token_auction_lab.continuation import WinnerContinuation

        tokenizer = SyntheticTokenizer()
        rng = np.random.default_rng(7)
        continuation = WinnerContinuation(3, 0.98)
        ids, carried = [], []
        pa, pb = np.zeros(256), np.zeros(256)
        pa[5:], pb[5:] = np.linspace(1, 3, 251), np.linspace(3, 1, 251)
        pa /= pa.sum()
        pb /= pb.sum()
        # Scripted synthetic probabilities, including brand point masses and
        # repeated high-confidence spans; no actual-model quality claim.
        brands = {8: 0, 9: 1, 76: 2, 77: 3, 78: 4}
        for step in range(224):
            a, b = pa.copy(), pb.copy()
            if step % 16 in (0, 1, 2, 3):
                a[:] = b[:] = 0
                a[11], b[12] = 1, 1
            if step in brands:
                a[:] = 0
                a[brands[step]] = 1
                b = a.copy()
            span = Span(8, 10) if 64 <= step < 128 else Span(76, 79) if 128 <= step < 192 else None
            chosen = continuation.take(a, b)
            if chosen:
                allocation, _ = chosen
                carried.append(step)
            else:
                tape = RandomTape.draw(rng)
                if span:
                    message = tilt_race.message_seed(
                        span, tokenizer.decode(ids[span.start : span.end])
                    )
                    allocation = tilt_race.sample(
                        a, b, (1, 1), tape, key=1901, message=message, step=step
                    )
                else:
                    allocation = TwoBidderAuction(a, b).resolve(1, 1, tape)
                continuation.observe_auction(allocation, step)
            ids.append(allocation.token_id)
        text = tokenizer.decode(ids)
        self.assertEqual(tokenizer.encode(text), ids)
        self.assertEqual(len([s for s in carried if 64 <= s < 192]), 27)
        # No carry mask, auction trace, seed, or brand dictionary is passed.
        result = decode(text, tokenizer, key=1901, scheme="tilt-race")
        self.assertEqual(
            [(s["start_token"], s["end_token"]) for s in result["spans"]], [(8, 10), (76, 79)]
        )
        self.assertEqual(result["coverage"], "unknown")

    def test_raw_confidence_precedes_tilt_limit_miss_stop_and_eos(self):
        import test_decoder

        if test_decoder.torch is None:
            self.skipTest("Install inference dependencies.")
        from token_auction_lab.decoder import SharedPrefixDecoder
        from token_auction_lab.runner import RoundConfig, run_round

        # Choose a deterministic test pattern that would suppress a .98 raw token
        # at step 65 and boost a .97 raw token above .98 at step 69. Neither coded
        # confidence is allowed to decide continuation.
        message = next(
            m
            for m in range(10000)
            if not tilt.green(1901, m, 65, 5)
            and tilt.green(1901, m, 65, 0)
            and tilt.green(1901, m, 69, 5)
            and not tilt.green(1901, m, 69, 0)
        )
        payload = dict(
            key=1901,
            message_seed=message,
            start_token=8,
            end_token=10,
            text="synthetic span",
            carrier_start=64,
            carrier_end=128,
        )

        class SyntheticEncoder:
            brands = ("A", "B")
            calls = []

            def __init__(self, brands):
                pass

            def at_step(self, step, generated, text):
                self.calls.append(step)
                return payload if step >= 64 else None

            def finish(self, text, visible):
                return {
                    "canonical_roundtrip": True,
                    "synthetic": True,
                    "records": [{**payload, "complete": False}],
                }

        for eos in (False, True):
            with self.subTest(eos=eos):
                fixture = test_decoder.DecoderTests()
                fixture.setUp()
                events, vectors = [], {}
                SyntheticEncoder.calls = []
                original = SharedPrefixDecoder.distributions

                def distributions(decoder, temperature):
                    if len(decoder.contexts) == 1:
                        return original(decoder, temperature)
                    a, b = np.zeros(17), np.zeros(17)
                    step = len(decoder.generated)
                    if step < 64:
                        a[2] = b[2] = 1
                    elif step == 64:
                        a[2], b[3] = 1, 1
                    else:
                        token, confidence = (
                            (16, 0.99) if step == 71 else (5, 0.97 if step in (68, 69) else 0.98)
                        )
                        a[token], a[0], b[6] = confidence, 1 - confidence, 1
                    return [a, b]

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
                    ) as decoded,
                    patch.object(SharedPrefixDecoder, "distributions", distributions),
                    patch(
                        "token_auction_lab.runner.RandomTape.draw",
                        return_value=RandomTape(0.5, 0.5, 0.5, 0.1),
                    ) as draws,
                    patch.object(wm, "sample", wraps=wm.sample) as sampled,
                    patch.object(
                        wm, "coded_distributions", wraps=wm.coded_distributions
                    ) as transformed,
                ):
                    path = Path(tmp) / "combined"
                    result = run_round(
                        fixture.model,
                        test_decoder.TestTokenizer(),
                        [("A", [1]), ("B", [2, 3])],
                        RoundConfig(
                            max_tokens=80,
                            watermark=True,
                            watermark_brands=("A", "B"),
                            continuation_tokens=3,
                            checkpoint_steps=(64, 65, 69, 70),
                        ),
                        path,
                        {},
                        mode="synthetic_model_test",
                        no_ad_prompt=("User", [4, 5]),
                        on_event=lambda e: events.append(e) if e["type"] == "token" else None,
                        on_distributions=lambda step, ps: vectors.update({step: ps}),
                        should_stop=lambda: not eos and len(events) == 71,
                    )
                    expected_carried = [65, 66, 67, 70] + ([71] if eos else [])
                    self.assertEqual(
                        [e["step"] for e in events if e["selection_rule"] == "winner_continuation"],
                        expected_carried,
                    )
                    self.assertEqual([c.args[-1] for c in sampled.call_args_list], [64, 68, 69])
                    self.assertEqual([c.args[-1] for c in transformed.call_args_list], [64, 68, 69])
                    self.assertEqual(draws.call_count, 67)
                    self.assertEqual(SyntheticEncoder.calls, list(range(len(events))))
                    self.assertEqual(result["stop_reason"], "eos" if eos else "user_stop")
                    self.assertEqual(result["token_ids"], [e["token_id"] for e in events])
                    self.assertTrue(result["no_ad_reference_validation"]["shared_suffix_verified"])
                    self.assertEqual(
                        result["watermark"]["audit"]["carried_steps"], expected_carried
                    )
                    decoded.assert_called_once_with(result["text"])
                    self.assertFalse(result["watermark"]["audit"]["records"][0]["complete"])
                    self.assertEqual(events[68]["selection_rule"], "watermark")  # exhausted limit
                    self.assertEqual(
                        events[69]["selection_rule"], "watermark"
                    )  # raw confidence miss
                    self.assertGreater(events[69]["selected_probabilities"]["coded_p_a"], 0.98)
                    self.assertLess(
                        wm.coded_distributions(vectors[65][:2], payload, 65)[0][5], 0.98
                    )
                    for step in expected_carried:
                        e = events[step]
                        self.assertIsNone(e["watermark"])
                        self.assertEqual(e["watermark_block"], payload)
                        self.assertEqual(e["watermark_skip_reason"], "winner_continuation")
                        self.assertIsNone(e["tape"])
                        self.assertEqual(e["payment_a"] + e["payment_b"], 0)
                        self.assertEqual(e["selected_probabilities"]["p_output"], 1)
                        self.assertEqual(e["watermark_mixture_tv"], 0)
                    self.assertEqual(events[67]["continuation"]["remaining"], 0)
                    self.assertEqual(events[70]["continuation"]["source_step"], 69)
                    self.assertAlmostEqual(result["payment_a"], 3 / 9)
                    self.assertEqual(result["payment_b"], 0)
                    if eos:
                        self.assertEqual(result["token_ids"][-1], 16)
                    manifest = json.loads((path / "manifest.json").read_text())
                    self.assertEqual(
                        manifest["sampler"], "experimental_tilt_race_with_winner_continuation"
                    )

                    # Carried inspection is raw and deterministic; no watermark
                    # reconstruction or same-step bid counterfactual is allowed.
                    class Engine:
                        def decode(self, token):
                            return str(token)

                    service = LiveService(engine=Engine())
                    service.current = {"run_id": "fixture", "events": events, "vectors": vectors}
                    with patch.object(
                        wm, "coded_distributions", side_effect=AssertionError("No tilt on carry")
                    ):
                        view = service.inspect("fixture", 65)
                    row = next(r for r in view["rows"] if r["token_id"] == 5)
                    self.assertEqual(row["p_a"], 0.98)
                    self.assertEqual(row["p_output"], 1)
                    self.assertAlmostEqual(row["q"], 0.49)
                    self.assertEqual(view["watermark_skip_reason"], "winner_continuation")
                    self.assertEqual(view["outcome"]["payment_a"], 0)
                    with self.assertRaisesRegex(ValueError, "previous win"):
                        service.inspect("fixture", 65, a=2)
                    # Auction inspection still reproduces the recorded coded draw.
                    for step in (64, 68, 69):
                        view = service.inspect("fixture", step)
                        self.assertEqual(view["outcome"]["token_id"], events[step]["token_id"])


if __name__ == "__main__":
    unittest.main()
