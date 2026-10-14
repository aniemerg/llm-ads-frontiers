"""Audit live evidence and diagnose carrier capacity; not the text-only decoder."""

import argparse
import json
from pathlib import Path

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego.prototype import Span, message_seed, sample, token_evidence
from experiments.ad_stego.tokenizer_check import QwenTokenizer


def inspect(path):
    manifest = json.loads((path / "manifest.json").read_text())
    seed_function, sample_function, evidence_function = message_seed, sample, token_evidence
    if manifest.get("scheme") == "race":
        from experiments.ad_stego import race
        seed_function, sample_function, evidence_function = (
            race.message_seed, race.sample, race.token_evidence)
    elif manifest.get("scheme") == "tilt":
        from experiments.ad_stego import tilt
        seed_function, sample_function, evidence_function = (
            tilt.message_seed, tilt.sample, tilt.token_evidence)
    elif manifest.get("scheme") == "tilt-race":
        from experiments.ad_stego import tilt_race
        seed_function, sample_function, evidence_function = (
            tilt_race.message_seed, tilt_race.sample, tilt_race.token_evidence)
    tokenizer = QwenTokenizer(manifest["tokenizer_file"])
    tokenizer.vocab_size = manifest["codebook_vocabulary_size"]
    key = manifest["nonsecret_fixed_key"]
    rows = []
    for case in manifest["plan"]:
        folder = path / case["name"]
        result = json.loads((folder / "result.json").read_text())
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        if [e["token_id"] for e in events] != result["generated_token_ids"]:
            raise AssertionError("Trace differs from selected token sequence.")
        if tokenizer.encode((folder / "response.txt").read_text()) != result["canonical_token_ids"]:
            raise AssertionError("Saved response differs from final canonical token sequence.")
        checkpoint_count = 0
        for checkpoint in sorted(folder.glob("probabilities-*.npz")):
            event = events[int(checkpoint.stem.split("-")[-1])]
            with np.load(checkpoint) as arrays:
                pa, pb = arrays["pa"], arrays["pb"]
            tape = RandomTape(**event["random_tape"])
            if event["message_seed"] is None:
                actual = TwoBidderAuction(pa, pb).resolve(*manifest["bids"], tape)
            else:
                actual = sample_function(pa, pb, manifest["bids"], tape, key=key,
                                         message=event["message_seed"], step=event["step"])
            for name in ("token_id", "branch", "payment_a", "payment_b",
                         "candidate_a", "candidate_b"):
                if getattr(actual, name) != event[name]:
                    raise AssertionError(f"Checkpoint {checkpoint} disagrees on {name}.")
            checkpoint_count += 1
        carriers = []
        for record in result["sent_records"]:
            start, end = record["carrier_start"], record["carrier_end"]
            if not record["complete"]:
                carriers.append({**record, "score": None})
                continue
            msg = seed_function(Span(record["start_token"], record["end_token"]), record["text"])
            score = sum(float(evidence_function(t, key, np.uint64(msg), step, tokenizer.vocab_size))
                        for step, t in enumerate(result["canonical_token_ids"][start:end], start))
            carriers.append({**record, "score": score,
                             "mean_entropy_nats": np.mean(
                                 [e["mixture_entropy_nats"] for e in events[start:end]]).item(),
                             "mean_max_probability": np.mean(
                                 [e["mixture_max_probability"] for e in events[start:end]]).item()})
        rows.append({"name": case["name"], "tokens": len(result["canonical_token_ids"]),
                     "stop": result["stop_reason"], "brands": result["brand_occurrences"],
                     "correct_spans": result["correct_spans"], "wrong_spans": result["wrong_spans"],
                     "decoded": result["decoded"], "carriers": carriers,
                     "canonical_roundtrip": result["canonical_roundtrip"],
                     "cache_passed": result["cache_validation"]["passed"],
                     "checkpoints_replayed": checkpoint_count})
    (path / "audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
