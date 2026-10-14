"""Model-loop tests use random tiny Qwen weights: they are not quality evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from auction import Allocation, TwoBidderAuction
from token_auction_lab.prompts import messages, scenarios

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from token_auction_lab.decoder import SharedPrefixDecoder, probabilities
    from token_auction_lab.runner import RoundConfig, run_round, top_tokens
except ImportError:
    torch = None


class PromptTests(unittest.TestCase):
    def test_same_user_task_and_allowed_facts_in_both_contexts(self):
        for scenario in scenarios().values():
            a, b = (messages(scenario, i) for i in (0, 1))
            self.assertEqual(a[-1], b[-1])
            self.assertNotEqual(a[0], b[0])
            for ad in scenario["advertisers"]:
                for fact in ad["facts"]:
                    self.assertIn(fact, a[0]["content"])
                    self.assertIn(fact, b[0]["content"])
            self.assertNotIn("bid", a[0]["content"].lower())
            self.assertNotIn("bid", b[0]["content"].lower())


class TestTokenizer:
    eos_token_id = 16

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if not (kwargs.get("skip_special_tokens") and i == 16))


@unittest.skipIf(torch is None, "Install the inference extra to run model-loop tests.")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = Qwen3Config(
            vocab_size=17,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=16,
        )
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()

    def test_cache_matches_uncached_and_suffixes_remain_identical(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7, 8]])
        self.assertIsNot(decoder.contexts[0].cache, decoder.contexts[1].cache)
        for token in [9, 1, 12, 0, 16]:
            decoder.advance(token)
            errors = decoder.check_cache(atol=2e-5, rtol=1e-4)
            self.assertLess(max(errors), 2e-5)
        self.assertEqual(decoder.generated, [9, 1, 12, 0, 16])
        for context in decoder.contexts:
            self.assertEqual(context.token_ids[len(context.prefix) :], decoder.generated)
            self.assertEqual(context.attention_mask.shape[1], len(context.token_ids))
            self.assertEqual(context.cache.get_seq_length(), len(context.token_ids))
        self.assertNotEqual(decoder.contexts[0].prefix, decoder.contexts[1].prefix)

    def test_full_float64_distributions_are_accepted_by_reference(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2], [3, 4, 5]])
        a, b = decoder.distributions(0.8)
        self.assertEqual(a.dtype, np.float64)
        self.assertEqual(a.shape, (17,))
        self.assertAlmostEqual(float(a.sum()), 1, places=14)
        self.assertGreater(TwoBidderAuction(a, b).disagreement, 0)
        for invalid in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                probabilities(decoder.contexts[0].logits, invalid)

    def test_production_sdpa_cache_matches_uncached(self):
        self.model.config._attn_implementation = "sdpa"
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5, 6, 7, 8]])
        for token in [9, 1, 12, 16]:
            decoder.advance(token)
            self.assertLess(max(decoder.check_cache(atol=2e-5, rtol=1e-4)), 2e-5)

    def test_cache_probability_guard_detects_corruption(self):
        decoder = SharedPrefixDecoder(self.model, [[1, 2, 3], [4, 5]])
        decoder.advance(9)
        with torch.inference_mode():
            decoder.contexts[0].logits[0] += 5
        # Even when a caller allows a loose logit tolerance, a substantially
        # changed sampling distribution must fail the independent TV guard.
        with self.assertRaisesRegex(AssertionError, "probability TV"):
            decoder.check_cache(atol=10, rtol=0, max_probability_tv=1e-4)

    def test_top_table_is_union_and_is_not_renormalized(self):
        a = np.array([0.8, 0.1, 0.1])
        b = np.array([0.1, 0.1, 0.8])
        q = (a + b) / 2
        rows = top_tokens(TestTokenizer(), a, b, q, 1)
        self.assertEqual({row["token_id"] for row in rows}, {0, 2})
        self.assertAlmostEqual(sum(row["q"] for row in rows), 0.9)

    def test_round_logs_reference_payments_tape_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-round"
            result = run_round(
                self.model,
                TestTokenizer(),
                [("A", [1, 2]), ("B", [3, 4, 5])],
                RoundConfig(max_tokens=4, checkpoint_steps=(0,)),
                path,
                {"model_id": "random-tiny-qwen3-test"},
                mode="synthetic_model_test",
            )
            events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
            token_events = events[:-1]
            self.assertEqual(len(token_events), result["tokens"])
            self.assertTrue(result["shared_suffix_verified"])
            self.assertTrue(all(event["mode"] == "synthetic_model_test" for event in events))
            from auction import RandomTape

            with np.load(path / "checkpoint-000.npz", allow_pickle=False) as checkpoint:
                auction = TwoBidderAuction(checkpoint["p_a"], checkpoint["p_b"])
                first = token_events[0]
                replay = auction.resolve(1, 1, RandomTape(**first["tape"]))
                self.assertEqual(replay.token_id, first["token_id"])
                self.assertEqual(replay.payment_a, first["payment_a"])
            self.assertAlmostEqual(sum(e["payment_a"] for e in token_events), result["payment_a"])

    def test_eos_is_fed_to_both_contexts_and_stops_generation(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "token_auction_lab.runner.TwoBidderAuction.resolve",
                return_value=Allocation(16, "shared", 0, 0, None, None),
            ),
        ):
            result = run_round(
                self.model,
                TestTokenizer(),
                [("A", [1]), ("B", [2, 3])],
                RoundConfig(max_tokens=4),
                Path(tmp) / "eos",
                {},
                mode="synthetic_model_test",
            )
            self.assertEqual(result["token_ids"], [16])
            self.assertEqual(result["stop_reason"], "eos")
            self.assertTrue(result["shared_suffix_verified"])
            self.assertEqual(len(result["cache_max_absolute_errors"]), 2)

    def test_length_cap_and_baseline_without_auction(self):
        self.model.generation_config.eos_token_id = None
        tokenizer = TestTokenizer()
        tokenizer.eos_token_id = None
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "token_auction_lab.runner.TwoBidderAuction.resolve",
                side_effect=AssertionError("The no-ad path must not run an auction"),
            ),
        ):
            result = run_round(
                self.model,
                tokenizer,
                [("No ad", [1, 2])],
                RoundConfig(0, 0, baseline=True, max_tokens=3),
                Path(tmp) / "baseline",
                {},
                mode="synthetic_model_test",
            )
            self.assertEqual(result["tokens"], 3)
            self.assertEqual(result["stop_reason"], "length")
            self.assertEqual(result["payment_a"] + result["payment_b"], 0)

    def test_config_rejects_zero_pair_and_unbounded_lengths(self):
        for kwargs in [
            {"bid_a": 0, "bid_b": 0},
            {"max_tokens": 97},
            {"bid_a": -1},
            {"bid_b": float("nan")},
        ]:
            with self.assertRaises(ValueError):
                RoundConfig(**kwargs)
        self.assertTrue(RoundConfig(0, 0, baseline=True).baseline)

    def test_no_ad_observer_matches_full_prefix_without_changing_auction(self):
        self.model.generation_config.eos_token_id = None
        tokenizer = TestTokenizer()
        tokenizer.eos_token_id = None
        prompts = [("A", [1, 2]), ("B", [3, 4, 5])]
        reference_ids = [7, 8, 9, 10]
        captured = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = run_round(self.model, tokenizer, prompts, RoundConfig(max_tokens=4),
                root / "plain", {}, mode="synthetic_model_test")
            compared = run_round(self.model, tokenizer, prompts, RoundConfig(max_tokens=4),
                root / "compared", {}, mode="synthetic_model_test",
                no_ad_prompt=("User only", reference_ids),
                on_distributions=lambda step, vectors: captured.append(vectors))
            self.assertEqual(plain["token_ids"], compared["token_ids"])
            self.assertEqual(plain["payment_a"], compared["payment_a"])
            self.assertEqual(plain["payment_b"], compared["payment_b"])
            self.assertEqual(len(compared["cache_max_absolute_errors"]), 2)
            self.assertTrue(compared["no_ad_reference_validation"]["shared_suffix_verified"])
            original = [json.loads(line) for line in (root / "plain/events.jsonl").read_text().splitlines()][:-1]
            events = [json.loads(line) for line in (root / "compared/events.jsonl").read_text().splitlines()][:-1]
            for step, (vectors, event) in enumerate(zip(captured, events)):
                self.assertEqual(len(vectors), 3)
                self.assertEqual(event["tape"], original[step]["tape"])
                self.assertEqual(event["branch"], original[step]["branch"])
                self.assertEqual(event["candidate_a"], original[step]["candidate_a"])
                self.assertEqual(event["candidate_b"], original[step]["candidate_b"])
                ids = torch.tensor([reference_ids + compared["token_ids"][:step]])
                with torch.inference_mode():
                    logits = self.model(input_ids=ids, use_cache=False).logits[0, -1]
                expected = probabilities(logits, .8)
                np.testing.assert_allclose(vectors[2], expected, rtol=2e-6, atol=1e-8)
                selected = event["token_id"]
                self.assertAlmostEqual(event["selected_probabilities"]["p_no_ad"], vectors[2][selected])
                self.assertAlmostEqual(event["no_ad_comparison"]["tv_q"],
                    float(np.abs((vectors[0]+vectors[1])/2 - vectors[2]).sum()/2))
            with np.load(root / "compared/checkpoint-000.npz", allow_pickle=False) as checkpoint:
                np.testing.assert_array_equal(checkpoint["p_no_ad"], captured[0][2])
            manifest = json.loads((root / "compared/manifest.json").read_text())
            self.assertEqual(manifest["no_ad_reference"]["token_ids"], reference_ids)

    def test_top_table_includes_no_ad_candidates_without_renormalization(self):
        a = np.array([.8, .1, .1])
        b = np.array([.1, .1, .8])
        p_no_ad = np.array([.1, .8, .1])
        rows = top_tokens(TestTokenizer(), a, b, (a+b)/2, 1, p_no_ad)
        self.assertEqual({row["token_id"] for row in rows}, {0, 1, 2})
        middle = next(row for row in rows if row["token_id"] == 1)
        self.assertAlmostEqual(middle["p_no_ad"], .8)
        self.assertAlmostEqual(middle["delta_q_no_ad"], -.7)


if __name__ == "__main__":
    unittest.main()
