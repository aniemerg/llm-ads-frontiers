"""Recompute recorded probability checkpoints without KV caches; never rewrites a run."""

import argparse
import hashlib
import json
from importlib.metadata import distributions
from pathlib import Path

import numpy as np


def main():
    import torch
    from transformers import AutoModelForCausalLM

    from token_auction_lab.decoder import probabilities

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.case.parent / "manifest.json").read_text())
    result = json.loads((args.case / "result.json").read_text())
    prompts = json.loads((args.case / "prompts.json").read_text())
    torch.manual_seed(manifest["torch_seed"])
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_id"], revision=manifest["model_revision"],
        dtype=getattr(torch, manifest["dtype"]),
        attn_implementation=manifest["attention_implementation"],
        local_files_only=True).to(manifest["device"]).eval()
    checks = []
    with torch.inference_mode():
        for checkpoint in sorted(args.case.glob("probabilities-*.npz")):
            step = int(checkpoint.stem.split("-")[-1])
            with np.load(checkpoint) as saved:
                for i, name in enumerate(("pa", "pb")):
                    ids = torch.tensor([prompts[i]["token_ids"] +
                                        result["generated_token_ids"][:step]],
                                       dtype=torch.long, device=manifest["device"])
                    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   position_ids=positions, use_cache=False,
                                   return_dict=True).logits[0, -1]
                    actual = probabilities(logits, manifest["temperature"])
                    delta = np.abs(actual - saved[name])
                    tv = float(delta.sum() / 2)
                    checks.append({"step": step, "bidder": i, "probability_tv": tv,
                                   "max_absolute_probability_error": float(delta.max()),
                                   "passes_existing_tv_guard": tv <= 1e-4})
    report = {
        "mode": "REPLAY_UNCACHED_PROBABILITY_CHECK", "case": str(args.case),
        "model_revision": manifest["model_revision"], "checks": checks,
        "original_cache_validation": result["cache_validation"],
        "note": "Additional diagnostics only; original failed logit check remains failed.",
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.case / "uncached_probability_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (args.case / "uncached_probability_diagnostic.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
