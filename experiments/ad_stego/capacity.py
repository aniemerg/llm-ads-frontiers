"""REPLAY diagnostics on saved actual-model distributions, not new generation."""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.ad_stego import tilt
from experiments.ad_stego.prototype import Config, possible_spans


def entropy(weights):
    mass = float(weights.sum())
    if mass == 0:
        return 0.
    p = weights[weights > 0] / mass
    return float(-np.dot(p, np.log2(p)))


def entropy_split(pa, pb, lam=.5):
    c = np.minimum(pa, pb)
    mass = float(c.sum())
    q = lam * pa + (1-lam) * pb
    conditional = (mass * entropy(c) + (1-mass) * (
        lam * entropy(pa-c) + (1-lam) * entropy(pb-c)))
    full = entropy(q)
    return {"mixture_entropy_bits": full, "candidate_entropy_bits": conditional,
            "branch_information_bits": full - conditional, "shared_mass": mass}


def main():
    runs = ["20261014T185631.151952Z", "20261014T190203.717954Z", "20261014T190535.193785Z"]
    output = Path("artifacts/ad-stego-capacity") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True)
    rows, seen = [], set()
    for run in runs:
        root = Path("artifacts/ad-stego-live") / run
        for file in sorted(root.glob("long-*-marked/probabilities-*.npz")):
            step = int(file.stem.split("-")[-1])
            if step == 0:
                continue
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            with np.load(file) as saved:
                pa, pb = saved["pa"], saved["pb"]
            trials = {d: [] for d in (0., 2., 4.)}
            for key in range(9000, 9032):
                mask = tilt.green(key, np.uint64(71), step,
                                  np.arange(len(pa), dtype=np.uint64))
                ga, gb = float(pa[mask].sum()), float(pb[mask].sum())
                for delta in trials:
                    za, zb = 1 + math.expm1(delta)*ga, 1 + math.expm1(delta)*gb
                    qa, qb = math.exp(delta)*ga/za, math.exp(delta)*gb/zb
                    kl = ((delta*qa - math.log(za)) + (delta*qb - math.log(zb))) / (2*math.log(2))
                    multiplier = np.where(mask, math.exp(delta), 1.)
                    ca, cb = pa*multiplier/za, pb*multiplier/zb
                    trials[delta].append({"hit_probability": (qa+qb)/2,
                                          "mean_bidder_kl_bits": kl,
                                          "mixture_tv": float(np.abs((ca+cb-pa-pb)/2).sum()/2),
                                          "shared_mass": float(np.minimum(ca, cb).sum())})
            rows.append({"checkpoint": str(file), "sha256": digest, **entropy_split(pa, pb),
                         "tilt": {str(d): {name: float(np.mean([t[name] for t in values]))
                                           for name in values[0]} for d, values in trials.items()}})
    summary = {"mode": "REPLAY_FIXED_PREFIX_CAPACITY_PROBE", "checkpoints": len(rows),
               "key_range": [9000, 9032], "message_seed": 71,
               "numpy": np.__version__, "sampling_config": asdict(Config()),
               "entropy": {k: float(np.mean([r[k] for r in rows])) for k in (
                   "mixture_entropy_bits", "candidate_entropy_bits", "branch_information_bits")},
               "tilt": {d: {k: float(np.mean([r["tilt"][d][k] for r in rows]))
                            for k in rows[0]["tilt"][d]} for d in rows[0]["tilt"]},
               "note": "Unweighted checkpoint averages; entropy is a ceiling, not achievable "
                       "payload rate. Fixed prefixes do not predict changed future generation."}
    config = Config()
    tests = sum(len(possible_spans(i, 32)) for i in range(64, 300-64+1, 64))
    summary["example_decoder"] = {"response_tokens": 300, "hypotheses": tests,
                                  "log2_hypotheses_over_alpha": math.log2(tests/config.alpha),
                                  "green_hits_required": tilt.score_threshold(64, tests, .01)}
    for source in (Path(__file__), Path(__file__).with_name("tilt.py"),
                   Path(__file__).with_name("race.py"), Path(__file__).with_name("prototype.py")):
        target = output / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "checkpoints.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
