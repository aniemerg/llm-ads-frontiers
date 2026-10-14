"""Audit a complete CLI sweep: python -m scripts.audit_smoke RUN_DIRECTORY.

Requires the saved checkpoints and cached tokenizer; never runs model inference.
Run without Python's -O flag, which disables assertions.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from auction import RandomTape, TwoBidderAuction

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
print("REPLAY AUDIT of stored traces; no model inference is run.")
first_distributions = []
for result in summary:
    path = root / result["round_id"]
    manifest = json.loads((path / "manifest.json").read_text())
    assert json.loads((path / "result.json").read_text()) == result
    events = [json.loads(s) for s in (path / "events.jsonl").read_text().splitlines()]
    assert events[-1] == result
    tokens = events[:-1]
    assert [e["token_id"] for e in tokens] == result["token_ids"]
    assert len(tokens) == result["tokens"]
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"], revision=manifest["tokenizer_revision"], local_files_only=True
    )
    for i, event in enumerate(tokens):
        assert event["step"] == i
        decoded = tokenizer.decode(
            result["token_ids"][: i + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        assert decoded == event["decoded_text_so_far"]
        assert event["mode"] == manifest["mode"]
        assert 0 <= event["payment_a"] <= result["bid_a"] + 1e-10
        assert 0 <= event["payment_b"] <= result["bid_b"] + 1e-10
        assert not (event["payment_a"] and event["payment_b"])
        if "checkpoint" in event:
            with np.load(path / event["checkpoint"], allow_pickle=False) as cp:
                assert list(cp["prefix"]) == result["token_ids"][:i]
                a = TwoBidderAuction(cp["p_a"], cp["p_b"])
                tape = RandomTape(**event["tape"])
                replay = a.resolve(event["bid_a"], event["bid_b"], tape)
                assert replay.token_id == event["token_id"]
                assert replay.payment_a == event["payment_a"]
                assert replay.payment_b == event["payment_b"]
                q = a.mixture(event["bid_a"], event["bid_b"])
                for row in event["top_tokens"]:
                    j = row["token_id"]
                    np.testing.assert_allclose(
                        [row["p_a"], row["p_b"], row["q"]],
                        [cp["p_a"][j], cp["p_b"][j], q[j]],
                        atol=1e-14,
                        rtol=1e-12,
                    )
                if i == 0:
                    first_distributions.append((cp["p_a"].copy(), cp["p_b"].copy()))
        if manifest["mode"] == "live_model":
            assert (event["bid_a"], event["bid_b"]) == (result["bid_a"], result["bid_b"])
    assert tokens[-1]["decoded_text_so_far"] == result["text"]
    np.testing.assert_allclose(
        [sum(e["payment_a"] for e in tokens), sum(e["payment_b"] for e in tokens)],
        [result["payment_a"], result["payment_b"]],
        atol=1e-12,
    )
    print(
        json.dumps(
            {
                "round_id": result["round_id"],
                "bids": [result["bid_a"], result["bid_b"]],
                "mode": "replay_audit",
                "source_mode": result["mode"],
                "tokens": result["tokens"],
                "stop": result["stop_reason"],
                "seconds": result["seconds"],
                "ttft": result["time_to_first_token_seconds"],
                "s_per_token": result["seconds_per_token"],
                "payment_a": result["payment_a"],
                "payment_b": result["payment_b"],
                "mean_tv": result["mean_disagreement_mass"],
                "branches": dict(Counter(e["branch"] for e in tokens)),
                "cache_errors": result["cache_max_absolute_errors"],
                "memory": result["memory"],
                "text": result["text"],
            },
            ensure_ascii=False,
        )
    )
for pair in first_distributions[1:]:
    for actual, expected in zip(pair, first_distributions[0]):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)
print(
    "AUDIT PASSED: unchanged decoded text, fixed bids, token sequence, payment bounds and sums, "
    "checkpoint replay, displayed probabilities, and bid-independent first-prefix distributions."
)
