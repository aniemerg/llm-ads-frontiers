"""Isolated real-model watermark experiment; production inference is unchanged.

All output is actually sampled. Both private contexts receive every selected ID.
Only local cached weights are used. No text insertion, repair, or forced EOS.
"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego.prototype import Config, Span, message_seed, sample
from experiments.ad_stego.tokenizer_check import REVISION, QwenTokenizer
from token_auction_lab.prompts import render, scenarios

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parents[2]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def brand_spans(text, tokenizer, brands):
    """Encoder/evaluator helper. Decoder never calls this or sees brands."""
    encoding = tokenizer.backend.encode(text, add_special_tokens=False)
    found = []
    for brand in brands:
        for match in re.finditer(r"(?<!\w)" + re.escape(brand) + r"(?!\w)", text):
            touched = [i for i, (lo, hi) in enumerate(encoding.offsets)
                       if lo < match.end() and hi > match.start()]
            if touched and len(touched) <= 32:
                start, end = touched[0], touched[-1] + 1
                found.append({"start_token": start, "end_token": end, "brand": brand,
                              "text": tokenizer.decode(encoding.ids[start:end])})
    return sorted(found, key=lambda r: (r["start_token"], r["end_token"]))


def experiment_plan():
    short = scenarios()["transit-hotel"]
    long = {
        "title": "Longer fictional travel-planning response",
        "task": "Write a friendly travel-planning message of about 200 words for a visitor "
                "to a conference in the fictional Fort Mason district. Discuss arrival, "
                "choosing a place to stay, and what to check before making arrangements. "
                "Use only the supplied facts, and frame unknown details as things to check.",
        "advertisers": [
            {"brand": "Paradigm House", "facts": [
                "Paradigm House is a hotel in the fictional Fort Mason district.",
                "It has a quiet courtyard.", "It offers luggage storage."]},
            {"brand": "Fort Mason Express", "facts": [
                "Fort Mason Express is a shuttle between Central Station and the fictional "
                "Fort Mason district.", "Its shuttles have step-free boarding.",
                "Its shuttles have space for bicycles."]},
        ],
    }
    return [{"name": f"{label}-seed{seed}-{'marked' if marked else 'unmarked'}",
             "scenario": scenario, "seed": seed, "marked": marked,
             "max_tokens": 128 if label == "short" else 384}
            for label, scenario, seeds in (("short", short, (7,)), ("long", long, (7, 19)))
            for seed in seeds for marked in (False, True)]


def run_case(model, hf_tokenizer, tokenizer, case, directory, config, key, tokenizer_path,
             scheme, temperature):
    from token_auction_lab.decoder import SharedPrefixDecoder

    seed_function, sample_function = message_seed, sample
    if scheme == "race":
        from experiments.ad_stego import race
        seed_function, sample_function = race.message_seed, race.sample
    elif scheme in ("tilt", "tilt-race"):
        from experiments.ad_stego import tilt
        seed_function, sample_function = tilt.message_seed, tilt.sample
        if scheme == "tilt-race":
            from experiments.ad_stego import tilt_race
            seed_function, sample_function = tilt_race.message_seed, tilt_race.sample

    directory.mkdir()
    prompts = [render(hf_tokenizer, case["scenario"], i) for i in (0, 1)]
    dump(directory / "prompts.json", [{"text": p, "token_ids": ids} for p, ids in prompts])
    decoder = SharedPrefixDecoder(model, [ids for _, ids in prompts])
    rng = np.random.default_rng(case["seed"])
    brands = [ad["brand"] for ad in case["scenario"]["advertisers"]]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    queue, seen, records, canonical_failures = [], set(), [], []
    active, message, stop = None, None, "length"
    started = time.perf_counter()
    with (directory / "events.jsonl").open("w") as events:
        for step in range(case["max_tokens"]):
            if step % config.block_tokens == 0:
                active, message = None, None
                text = hf_tokenizer.decode(decoder.generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
                canonical = tokenizer.encode(text)
                if canonical != decoder.generated:
                    canonical_failures.append(step)
                else:
                    for record in brand_spans(text, tokenizer, brands):
                        identity = (record["start_token"], record["end_token"], record["text"])
                        # Wait for at least one subsequent token to stabilize the boundary.
                        if identity not in seen and record["end_token"] < step:
                            seen.add(identity)
                            queue.append(record)
                    if case["marked"] and queue:
                        active = {**queue.pop(0), "carrier_start": step,
                                  "carrier_end": step + config.block_tokens}
                        records.append(active)
                        message = seed_function(Span(active["start_token"], active["end_token"]),
                                                active["text"])
            pa, pb = decoder.distributions(temperature)
            tape = RandomTape.draw(rng)
            auction = TwoBidderAuction(pa, pb)
            baseline = auction.resolve(1., 1., tape)
            raw_baseline = baseline
            sampled_pa, sampled_pb = pa, pb
            if scheme in ("tilt", "tilt-race") and message is not None:
                sampled_pa, sampled_pb = tilt.transform(
                    pa, pb, key=key, message=message, step=step)
                baseline = TwoBidderAuction(sampled_pa, sampled_pb).resolve(1., 1., tape)
                chosen = baseline if scheme == "tilt" else sample_function(
                    pa, pb, (1., 1.), tape, key=key, message=message, step=step)
            else:
                chosen = baseline if message is None else sample_function(
                    pa, pb, (1., 1.), tape, key=key, message=message, step=step)
            if (chosen.branch, chosen.payment_a, chosen.payment_b) != (
                    baseline.branch, baseline.payment_a, baseline.payment_b):
                raise AssertionError("Frozen-prefix branch or payment changed.")
            q = (sampled_pa + sampled_pb) / 2
            positive = q[q > 0]
            event = {"step": step, **asdict(chosen), "random_tape": asdict(tape),
                     "payload": active, "message_seed": message,
                     "shared_mass": float(np.minimum(sampled_pa, sampled_pb).sum()),
                     "mixture_entropy_nats": float(-np.dot(positive, np.log(positive))),
                     "mixture_max_probability": float(q.max()),
                     "selected_pa": float(sampled_pa[chosen.token_id]),
                     "selected_pb": float(sampled_pb[chosen.token_id]),
                     "raw_reference_at_same_prefix": asdict(raw_baseline),
                     "raw_shared_mass": float(np.minimum(pa, pb).sum()),
                     "input_distribution_changed": scheme in ("tilt", "tilt-race")
                         and message is not None,
                     "mixture_tv_from_raw": float(np.abs(q-(pa+pb)/2).sum()/2)}
            if step in (0, 64, 128, 192, 256):
                np.savez_compressed(directory / f"probabilities-{step}.npz", pa=pa, pb=pb)
            decoder.advance(chosen.token_id)
            events.write(json.dumps(event) + "\n")
            events.flush()
            if (step + 1) % 64 == 0:
                print(json.dumps({"case": case["name"], "tokens": step + 1,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
            if chosen.token_id in eos:
                stop = "eos"
                break
    text = hf_tokenizer.decode(decoder.generated, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
    # Save exact text without adding a newline. It is the only per-run decoder input.
    (directory / "response.txt").write_text(text)
    duration = time.perf_counter() - started
    visible = decoder.generated[:-1] if decoder.generated[-1] in eos else decoder.generated
    canonical = tokenizer.encode(text)
    result = {
        "mode": "LIVE_MODEL", "case": case, "text": text,
        "generated_token_ids": decoder.generated, "canonical_token_ids": canonical,
        "canonical_roundtrip": visible == canonical,
        "prefix_canonical_failures": canonical_failures,
        "stop_reason": stop, "partial_generation": stop != "eos",
        "generation_seconds": duration, "sent_records": records,
        "brand_occurrences": brand_spans(text, tokenizer, brands),
        "shared_suffix_verified": True,
    }
    for record in records:
        record["complete"] = record["carrier_end"] <= len(canonical)
    dump(directory / "result.json", result)
    # The cache test uses the same float32 MPS tolerances as the existing runner.
    try:
        errors = decoder.check_cache(atol=3e-4, rtol=1e-4, temperature=temperature,
                                     max_probability_tv=1e-4)
        result["cache_validation"] = {"passed": True, "max_absolute_errors": errors,
                                      "probability_tvs": decoder.cache_probability_tvs}
    except AssertionError as error:
        result["cache_validation"] = {"passed": False, "error": str(error)}
    command = [sys.executable, "-m", "experiments.ad_stego.decode_text",
               str(directory / "response.txt"), "--tokenizer-json", str(tokenizer_path),
               "--vocabulary-size", str(tokenizer.vocab_size), "--key", str(key),
               "--block-tokens", str(config.block_tokens), "--alpha", str(config.alpha),
               "--scheme", scheme]
    decoded = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    (directory / "decoded.json").write_text(decoded.stdout)
    result["decoded"] = json.loads(decoded.stdout)
    truth = {(r["start_token"], r["end_token"], r["text"])
             for r in result["brand_occurrences"]} if case["marked"] else set()
    recovered = {(r["start_token"], r["end_token"], r["text"])
                 for r in result["decoded"]["spans"]}
    result["correct_spans"] = len(truth & recovered)
    result["wrong_spans"] = len(recovered - truth)
    dump(directory / "result.json", result)
    print(json.dumps({"case": case["name"], "tokens": len(canonical),
                      "brands": len(result["brand_occurrences"]),
                      "correct_spans": result["correct_spans"],
                      "wrong_spans": result["wrong_spans"],
                      "cache": result["cache_validation"]["passed"],
                      "canonical": result["canonical_roundtrip"]}), flush=True)
    return result


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--scheme", choices=("affine", "race", "tilt", "tilt-race"),
                        default="affine")
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--marked-only", action="store_true",
                        help="Run marked cases only; use preserved unmarked controls separately.")
    args = parser.parse_args()
    output = args.output or Path("artifacts/ad-stego-live") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = Config(block_tokens=args.block_tokens)
    plan = experiment_plan()
    if args.marked_only:
        plan = [case for case in plan if case["marked"]]
    tokenizer_path = (Path.home() / ".cache/huggingface/hub" /
                      "models--Qwen--Qwen3-4B-Instruct-2507" / "snapshots" / REVISION /
                      "tokenizer.json")
    sources = [*Path(__file__).parent.glob("*.py"), ROOT / "auction.py",
               ROOT / "token_auction_lab/decoder.py", ROOT / "token_auction_lab/prompts.py",
               ROOT / "token_auction_lab/fact_sheets.json"]
    source_hashes = {}
    for source in sources:
        relative = source.resolve().relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = {
        "mode": "LIVE_MODEL", "model_id": MODEL, "model_revision": REVISION,
        "tokenizer_revision": REVISION, "tokenizer_file": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "plan": plan, "watermark": asdict(config), "nonsecret_fixed_key": 1901,
        "bids": [1., 1.], "temperature": args.temperature, "continuation_tokens": 0,
        "scheme": args.scheme,
        "preauction_transform": ({"preferred_fraction": .25, "log_odds_boost": 4.,
                                  "note": "Auction input distributions change; raw-mixture "
                                          "and raw-payment claims do not apply."}
                                 if args.scheme in ("tilt", "tilt-race") else None),
        "checkpoint_vectors": "raw model probabilities before any watermark transform",
        "filters": None, "torch_seed": 7, "dtype": "float32", "device": device,
        "attention_implementation": "sdpa", "local_files_only": True,
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "python": sys.version, "platform": platform.platform(),
        "source_sha256": source_hashes,
        "encoder_policy": "All completed exact brand occurrences queued FIFO; one per block.",
        "prompt_policy": "Existing private advertiser briefs and common task; no bids in prompts.",
        "quality_review": "Unedited outputs retained; no automatic factual/coherence judge.",
    }
    dump(output / "manifest.json", manifest)
    print(json.dumps({"status": "loading", "output": str(output), "device": device}), flush=True)
    torch.manual_seed(7)
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=True).to(device).eval()
    tokenizer = QwenTokenizer(tokenizer_path)
    manifest["tokenizer_assigned_ids"] = tokenizer.vocab_size
    tokenizer.vocab_size = model.config.vocab_size
    manifest["codebook_vocabulary_size"] = tokenizer.vocab_size
    manifest["eos_token_ids"] = model.generation_config.eos_token_id
    dump(output / "manifest.json", manifest)
    results = []
    for case in plan:
        results.append(run_case(model, hf_tokenizer, tokenizer, case, output / case["name"],
                                config, 1901, tokenizer_path, args.scheme, args.temperature))
        dump(output / "summary.json", results)
    print(json.dumps({"status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
