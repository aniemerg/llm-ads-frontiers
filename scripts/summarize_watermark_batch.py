"""Archive every run and failure from a completed, predeclared live batch."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from experiments.ad_stego import tilt_race, word_redundant, word_window
from experiments.ad_stego.prototype import score_threshold
from token_auction_lab import watermark as wm


def summarize(folder):
    report = json.loads((folder / "report.json").read_text())
    assert len(report["runs"]) == len(report["cases"]), "Batch is not complete"
    tokenizer = wm.get_tokenizer()
    for run in report["runs"]:
        encoding = tokenizer.backend.encode(run["text"], add_special_tokens=False)
        groups = word_window.carrier_groups(run["text"], encoding)
        diagnostics = []
        for record in run["encoded"].get("records", []):
            if record.get("protocol") == word_redundant.PROTOCOL:
                chunks, byte_offsets = word_redundant.positions(run["text"], encoding)
                family = next(f for f in run["decoded"]["families"]
                              if f["protocol"] == word_redundant.PROTOCOL)
                lanes = []
                for lane in (0, 1, "all"):
                    indices = [i for i, chunk in enumerate(chunks)
                               if chunk * word_redundant.CHUNK_WORDS >= record["carrier_start_word"]
                               and (lane == "all" or (chunk - 1) % 2 == lane)]
                    evidence = sum(float(tilt_race.token_evidence(
                        encoding.ids[i], wm.KEY, np.uint64(record["message_seed"]),
                        byte_offsets[i], tokenizer.vocab_size)) for i in indices)
                    threshold = (score_threshold(len(indices), family["hypotheses_tested"], 0.005)
                                 if len(indices) >= word_redundant.MIN_EVIDENCE_TOKENS else None)
                    lanes.append({"lane": lane, "score": evidence, "threshold": threshold,
                                  "scored_tokens": len(indices)})
                diagnostics.append({"region": record["text"], "assigned_lane": record["lane"],
                                    "carried_steps": len(record.get("carried_steps", [])),
                                    "carrier_chunks": len(record["carriers"]), "lanes": lanes})
                continue
            if record.get("protocol") != word_window.PROTOCOL:
                continue
            indices = groups.get(record["carrier_start_word"], [])
            evidence = sum(float(tilt_race.token_evidence(
                encoding.ids[i], wm.KEY, np.uint64(record["message_seed"]),
                len(run["text"][:encoding.offsets[i][0]].encode()), tokenizer.vocab_size))
                for i in indices)
            family = next(f for f in run["decoded"]["families"]
                          if f["protocol"] == word_window.PROTOCOL)
            threshold = (score_threshold(len(indices), family["hypotheses_tested"], 0.005)
                         if len(indices) >= word_window.MIN_EVIDENCE_TOKENS else None)
            diagnostics.append({"region": record["text"], "score": evidence,
                                "threshold": threshold, "scored_tokens": len(indices),
                                "carried_steps": len(record.get("carried_steps", []))})
        run["post_decode_score_diagnostics"] = diagnostics
    marked = [r for r in report["runs"] if r["watermark"]]
    controls = [r for r in report["runs"] if not r["watermark"]]
    report["summary"] = {
        "marked_runs": len(marked), "unmarked_controls": len(controls),
        "both_brands_no_extras": sum(r["metrics"]["both_brands_correct_no_extras"] for r in marked),
        "one_brand": sum(len(r["metrics"]["correctly_recovered_brands"]) == 1 for r in marked),
        "neither_brand": sum(not r["metrics"]["correctly_recovered_brands"] for r in marked),
        "brands_recovered": sum(len(r["metrics"]["correctly_recovered_brands"]) for r in marked),
        "brand_opportunities": 2 * len(marked),
        "controls_with_false_detections": sum(bool(r["decoded"]["spans"]) for r in controls),
        "marked_runs_with_extra_detections": sum(bool(r["metrics"]["extra_detections"])
                                                 for r in marked),
        "regions_exceeding_16_words": sum(len(r["metrics"]["regions_over_word_limit"])
                                          for r in report["runs"]),
        "cache_or_generation_errors": sum(r["status"] == "error" for r in report["runs"]),
        "canonical_mismatches": sum(r["encoded"].get("canonical_roundtrip") is False
                                    for r in marked),
    }
    report["post_decode_analysis_note"] = (
        "Audit-based diagnostics run after text-only detection; they never change its results. "
        "Hallucinations were not scored, per user preference. All planned runs are retained. "
        "Small fixed-key single-task sample; not a calibrated failure rate."
    )
    report["plan_sha256"] = hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest()
    isolation = folder / "isolation-checks.json"
    if isolation.is_file():
        report["text_only_isolation_checks"] = json.loads(isolation.read_text())
    replays = folder / "replayed-controls.json"
    if replays.is_file():
        report["replayed_unmarked_controls"] = json.loads(replays.read_text())
        report["replay_summary"] = {
            "mode": "REPLAY; these are not new inference trials or unseen calibration data",
            "controls": len(report["replayed_unmarked_controls"]),
            "controls_with_detections": sum(bool(r["decoded"]["spans"])
                                            for r in report["replayed_unmarked_controls"]),
        }
    report["batch_source_note"] = (
        "Protocol, sampling parameters, and thresholds stayed fixed throughout this batch. "
        "Each run retains its filesystem source hashes."
    )
    return report


def render(report):
    esc = html.escape
    rows, details = [], []
    for run in report["runs"]:
        metrics = run["metrics"]
        rows.append(f'<tr><td><a href="#{run["label"]}">{run["label"]}</a></td>'
                    f'<td>{len(metrics["correctly_recovered_brands"])} / 2</td>'
                    f'<td>{len(metrics["extra_detections"])}</td><td>{run["status"]}</td></tr>')
        details.append(
            f'<details id="{run["label"]}"><summary>{run["label"]}</summary>'
            f'<p>{esc(run["run_id"])} · {run["words"]} words · '
            f'{run["generated_tokens_including_eos"]} tokens · stop: {run["stop_reason"]}</p>'
            f'<p class="response">{esc(run["text"])}</p>'
            f'<h3>Detected regions (text-only decoder)</h3><pre>'
            f'{esc(json.dumps(run["decoded"]["spans"], indent=2, ensure_ascii=False))}</pre>'
            f'<h3>Post-decode diagnostics</h3><pre>'
            f'{esc(json.dumps(run["post_decode_score_diagnostics"], indent=2))}</pre>'
            f'<h3>Run error</h3><pre>{esc(run["error"] or "None")}</pre></details>')
    return ('<!doctype html><meta charset="utf-8"><title>AdReveal word-region evaluation</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;'
            'color:#183b35;background:#faf9f5}table{border-collapse:collapse;width:100%}'
            'td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}'
            'details{margin:16px 0;padding:18px;background:white;border:1px solid #ddd}'
            'pre,.response{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}'
            '</style><h1>AdReveal: regions up to 16 words</h1>'
            '<p>LIVE MODEL EVALUATION · Qwen3-4B on local MPS · public key 1901</p>'
            f'<p>{esc(report["post_decode_analysis_note"])}</p>'
            f'<p>{esc(report.get("comparison_note", "All planned runs retained."))}</p>'
            f'<pre>{esc(json.dumps(report["summary"], indent=2))}</pre>'
            f'<pre>{esc(json.dumps(report.get("replay_summary", {}), indent=2))}</pre>'
            '<table><tr><th>Run</th><th>Brands recovered</th><th>Extra detections</th>'
            '<th>Run status</th></tr>' + ''.join(rows) + '</table>' + ''.join(details))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("archive", type=Path, help="JSON archive path; HTML uses the same stem")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--demo", type=Path,
                        help="Separate repeat used to restore a marked app result")
    args = parser.parse_args()
    report = summarize(args.folder)
    if args.demo:
        demo = json.loads(args.demo.read_text())["runs"][0]
        report["demo_repeat"] = {
            "note": ("Additional repeat to leave a marked result in the app; "
                     "excluded from batch counts"),
            **{k: demo[k] for k in ("seed", "run_id", "status", "text_sha256", "metrics")},
        }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        old = {r["seed"]: r for r in baseline["runs"] if r["watermark"]}
        report["paired_comparison"] = [
            {"seed": r["seed"], "before": old[r["seed"]]["metrics"]["correctly_recovered_brands"],
             "after": r["metrics"]["correctly_recovered_brands"]}
            for r in report["runs"] if r["watermark"] and r["seed"] in old]
        old_marked = list(old.values())
        both = sum(r["metrics"]["both_brands_correct_no_extras"] for r in old_marked)
        report["comparison_note"] = (
            f"Previous single-block region protocol: both brands in {both}/{len(old_marked)} runs. "
            "Marked runs use the same seeds and inputs; changed sampling can change later text. "
            "The two current controls use new seeds. Small sample; not a calibrated failure rate."
        )
    args.archive.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.archive.with_suffix(".html").write_text(render(report))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
