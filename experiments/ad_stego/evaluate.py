"""Reproducible synthetic evaluation. Never invokes a language model."""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.ad_stego.prototype import PROTOCOL, Config, SyntheticTokenizer, decode
from experiments.ad_stego.synthetic import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/ad-stego")
    parser.add_argument("--negative-count", type=int, default=1000)
    parser.add_argument("--positive-count", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()
    if args.negative_count < 1 or args.positive_count < 1:
        parser.error("Sample counts must be positive.")
    run = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    config = Config(alpha=args.alpha)
    tokenizer = SyntheticTokenizer()
    ratios = [(1, 0), (9, 1), (1, 1), (1, 9), (0, 1)]
    groups = [
        ("high_marked", True, "high", args.positive_count),
        ("medium_marked", True, "medium", args.positive_count),
        ("point_marked", True, "point", args.positive_count),
        ("high_unmarked", False, "high", args.negative_count),
        ("medium_unmarked", False, "medium", args.negative_count),
        ("high_unmarked_fixed_key", False, "high", args.negative_count),
        ("high_marked_fixed_key", True, "high", args.positive_count),
    ]
    sources = [Path(__file__), Path(__file__).with_name("prototype.py"),
               Path(__file__).with_name("synthetic.py"), Path("auction.py")]
    manifest = {
        "mode": "synthetic_probability_model", "protocol": PROTOCOL,
        "model": None, "model_revision": None, "prompts": None,
        "not_applicable_reason": "Scripted probability vectors; no language model or prompts.",
        "tokenizer": tokenizer.revision, "tokenizer_vocabulary": tokenizer.words,
        "dependencies": {"python": sys.version, "numpy": np.__version__},
        "platform": platform.platform(), "config": asdict(config),
        "generation": {"tokens": 224, "temperature": None, "continuation_tokens": 0,
                       "occurrences": [[8, [0, 1]], [76, [2, 3, 4]]]},
        "groups": groups, "bid_ratios_cycled": ratios,
        "seed_rule": "generation=10000+case_index; nonsecret key=20000+case_index, "
                     "except fixed_key groups use 1901",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": ["No LLM fluency or quality evidence", "Toy final-text tokenizer",
                        "Nominal p-values use an ideal independent-key null",
                        "Counter mixer is noncryptographic; no authentication"],
    }
    for source in sources:
        relative = source.resolve().relative_to(Path(__file__).resolve().parents[2])
        snapshot = run / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = {}
    t0 = time.monotonic()
    with (run / "cases.jsonl").open("w") as cases:
        for name, marked, entropy, count in groups:
            counts = {"responses": count, "both_correct": 0, "one_correct": 0,
                      "zero_correct": 0, "false_span_responses": 0, "any_detection": 0,
                      "correct_spans": 0, "wrong_spans": 0}
            for index in range(count):
                seed, key = 10000 + index, 20000 + index
                if name.endswith("fixed_key"):
                    key = 1901
                bids = ratios[index % len(ratios)]
                fixture = generate(seed=seed, key=key, bids=bids, marked=marked, entropy=entropy)
                decoded = decode(fixture["text"], tokenizer, key=key, config=config)
                expected = {(r["start"], r["end"]) for r in fixture["expected_spans"]}
                actual = {(r["start_token"], r["end_token"]) for r in decoded["spans"]}
                correct = actual & expected if marked else set()
                wrong = actual - correct
                counts[{0: "zero_correct", 1: "one_correct", 2: "both_correct"}[len(correct)]] += 1
                counts["false_span_responses"] += bool(wrong)
                counts["any_detection"] += bool(actual)
                counts["correct_spans"] += len(correct)
                counts["wrong_spans"] += len(wrong)
                row = {"group": name, "case_index": index, "seed": seed, "key": key,
                       "bids": bids, "text": fixture["text"], "expected": sorted(expected),
                       "decoded": decoded}
                cases.write(json.dumps(row) + "\n")
                if name == "high_marked" and index == 0:
                    (run / "example.json").write_text(json.dumps(row, indent=2) + "\n")
            summaries[name] = counts
            print(name, json.dumps(counts), flush=True)
    special = {}
    for name, kwargs in (
        ("truncated", {"tokens": 160}),
        ("late_brand", {"occurrences": ((8, (0, 1)), (220, (2, 3, 4)))}),
        ("repeated_brand", {"occurrences": ((8, (0, 1)), (76, (0, 1)))}),
    ):
        fixture = generate(**kwargs)
        special[name] = {"seed": 7, "nonsecret_key": 1901, "generation_overrides": kwargs,
                         "text": fixture["text"],
                         "decoded": decode(fixture["text"], tokenizer, key=1901)}
    summary = {"mode": "synthetic_probability_model", "groups": summaries,
               "special_cases": special, "elapsed_seconds": time.monotonic()-t0,
               "run_directory": str(run)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SYNTHETIC evidence:", run, flush=True)


if __name__ == "__main__":
    main()
