"""Run a predeclared local batch and decode exact text without run metadata.

Usage: python -m scripts.evaluate_live_watermark OUTPUT_DIRECTORY
Uses the active app inputs; leaves the last planned run visible in the app.
"""

import argparse
import hashlib
import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[1]
BUSY = {"loading", "prefilling", "generating", "validating", "no-ad-generating", "no-ad-validating"}


def api(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def record_id(record):
    if "start_word" in record:
        return (record["protocol"], record["start_word"], record["end_word"],
                record["carrier_start_word"])
    return record["start_token"], record["end_token"], record["carrier_start"]


def score(text, brands, audit, decoded):
    records = audit.get("records", [])
    complete = [r for r in records if r["complete"]]
    spans = decoded.get("spans", [])
    correct = [s for s in spans if record_id(s) in {record_id(r) for r in complete}]
    missed = [r for r in complete if record_id(r) not in {record_id(s) for s in spans}]
    extra = [s for s in spans if record_id(s) not in {record_id(r) for r in complete}]
    mentioned = [b for b in brands if re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", text)]
    recovered = [b for b in brands if any(b in s["literal_text"] for s in correct)]
    return {
        "mentioned_brands": mentioned,
        "correctly_recovered_brands": recovered,
        "missing_brands_in_text": [b for b in brands if b not in mentioned],
        "complete_records": len(complete),
        "incomplete_records": len(records) - len(complete),
        "pending_mentions": len(audit.get("pending_mentions", [])),
        "correct_records": len(correct),
        "missed_complete_records": missed,
        "extra_detections": extra,
        "both_brands_correct_no_extras": len(recovered) == len(brands) and not extra,
        "alignment_failures": audit.get("prefix_alignment_failures", []),
        "canonical_roundtrip": audit.get("canonical_roundtrip"),
        "regions_over_word_limit": [s for s in spans if len(s["literal_text"].split()) > 16],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-seed", type=int, default=21)
    parser.add_argument("--marked-runs", type=int, default=8)
    parser.add_argument("--control-runs", type=int, default=4)
    parser.add_argument("--control-first-seed", type=int)
    parser.add_argument("--marked-first", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = api("/api/live")
    if initial["status"] in BUSY:
        raise RuntimeError("An existing generation is active; batch was not started.")
    write(args.output / "before-batch.json", initial)
    base = initial.get("inputs", initial.get("defaults"))
    control_seed = args.first_seed if args.control_first_seed is None else args.control_first_seed
    controls = [{"watermark": False, "seed": s}
                for s in range(control_seed, control_seed + args.control_runs)]
    marked = [{"watermark": True, "seed": s}
              for s in range(args.first_seed, args.first_seed + args.marked_runs)]
    cases = marked + controls if args.marked_first else controls + marked
    plan = {
        "mode": "LIVE_MODEL_WATERMARK_EVALUATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_inputs": base,
        "cases": cases,
        "protocol": "Current app protocol with fixed public key; no tuning during batch. "
        "Each run manifest records exact code hashes and protocol parameters.",
        "decoding_inputs": "Only exact response text sent to /api/watermark/decode.",
        "controls": "Same ad contexts and bids with watermark disabled; not ad-free responses.",
        "scoring": "Exact region coordinates AND carrier start must match an eligible "
        "encoder record (word coordinates for v2; token coordinates for legacy). "
        "V2 eligibility means at least 32 scored carrier tokens, including partial final blocks. "
        "A brand is recovered only when fully inside that matching region. "
        "Scoring reads the audit only after independent text decoding returns.",
        "limitations": "Small single-task fixed-key sample; "
        "not general calibration or authentication.",
    }
    write(args.output / "plan.json", plan)
    report = {**plan, "runs": []}
    expected_run = initial.get("run_id")
    for index, case in enumerate(cases, 1):
        current = api("/api/live?after=4096")
        if current.get("run_id") != expected_run or current["status"] in BUSY:
            raise RuntimeError(
                "Active run changed externally; stopping batch without replacing it."
            )
        inputs = {**deepcopy(base), **case}
        started = time.monotonic()
        run_id = api("/api/live/run", inputs)["run_id"]
        expected_run = run_id
        label = f"{index:02d}-{'marked' if case['watermark'] else 'control'}-seed-{case['seed']}"
        print(f"START {label} {run_id}", flush=True)
        folder = args.output / label
        folder.mkdir()
        write(folder / "request.json", inputs)
        after, latest, last_progress = 0, None, started
        while True:
            snapshot = api(f"/api/live?run_id={run_id}&after={after}")
            if snapshot["events"]:
                latest = snapshot["events"][-1]
            after = snapshot["event_count"]
            if snapshot["status"] not in BUSY:
                break
            if time.monotonic() - last_progress > 30:
                print(f"PROGRESS {label} {snapshot['status']} tokens={after}", flush=True)
                last_progress = time.monotonic()
            time.sleep(0.5)
        # Fetch the final snapshot so failures retain every emitted token and input.
        snapshot = api(f"/api/live?run_id={run_id}")
        write(folder / "snapshot.json", snapshot)
        source = ROOT / "artifacts" / "live-lab" / run_id
        response_path = source / "response.txt"
        text = (
            response_path.read_text()
            if response_path.exists()
            else (latest["decoded_text_so_far"] if latest else "")
        )
        (folder / "response.txt").write_text(text)
        decoded = (
            api("/api/watermark/decode", {"text": text})
            if text
            else {"status": "no_output", "spans": []}
        )
        write(folder / "decoded.json", decoded)
        # Decoder call above receives no audit, brands, token IDs, or run identifier.
        audit_path = source / "watermark.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        write(folder / "encoding-audit.json", audit)
        brands = [a["brand"] for a in inputs["advertisers"]]
        metrics = score(text, brands, audit, decoded)
        result = snapshot.get("result") or {}
        error = snapshot.get("error")
        stop = result.get("stop_reason")
        if stop is None and latest:
            eos = snapshot["manifest"].get("eos_token_ids", [])
            eos = [eos] if isinstance(eos, int) else eos
            stop = (
                "eos"
                if latest["token_id"] in eos
                else ("length" if after == inputs["max_tokens"] else "unknown_after_failure")
            )
        record = {
            **case,
            "label": label,
            "run_id": run_id,
            "status": snapshot["status"],
            "error": error,
            "cache_check_failed": bool(error and error.startswith("AssertionError:")),
            "stop_reason": stop,
            "generated_tokens_including_eos": after,
            "wall_seconds_including_validation": time.monotonic() - started,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "words": len(text.split()),
            "charges": {
                k: latest.get(k) if latest else None for k in ["cumulative_a", "cumulative_b"]
            },
            "carried_tokens": sum(
                e.get("selection_rule") == "winner_continuation" for e in snapshot["events"]
            ),
            "encoded": audit,
            "decoded": decoded,
            "metrics": metrics,
            "raw_artifacts": str(source.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        }
        if not report["runs"]:
            report["model_environment"] = {
                k: snapshot["manifest"].get(k)
                for k in (
                    "model_id",
                    "model_revision",
                    "tokenizer",
                    "tokenizer_revision",
                    "device",
                    "dtype",
                    "hardware",
                    "platform",
                    "python",
                    "dependencies",
                    "source_sha256",
                    "attention_implementation",
                )
            }
        report["runs"].append(record)
        write(args.output / "report.json", report)
        print(
            f"DONE {label} status={snapshot['status']} tokens={after} "
            f"complete={metrics['complete_records']} correct={metrics['correct_records']} "
            f"extra={len(metrics['extra_detections'])} "
            f"both={metrics['both_brands_correct_no_extras']}",
            flush=True,
        )
        if snapshot.get("stop_requested"):
            raise RuntimeError("User stopped generation; remaining batch cancelled.")
    print(f"FINISHED {len(report['runs'])} runs: {args.output}/report.json", flush=True)


if __name__ == "__main__":
    main()
