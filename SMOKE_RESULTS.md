# Milestone 1 live smoke test — 2026-10-13 (Pacific)

The shared-prefix decoder ran the real pretrained model locally. The three
advertiser bid pairs and separate no-ad baseline completed, with different
generated messages and auditable token/payment traces. This is a mechanics
smoke test, not an evaluation of reliable ad copy or truthful bidding.

## Runtime and reproducibility

- Apple M2 Max, 96 GiB unified memory; macOS 15.7.9, MPS, float32, SDPA attention.
- `Qwen/Qwen3-4B-Instruct-2507` and tokenizer revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Python 3.11.15, PyTorch 2.14.1, Transformers 4.57.6, NumPy 2.4.6,
  tokenizers 0.22.2, huggingface_hub 0.36.2, safetensors 0.8.0.
- Seed 7 for every round; temperature 0.8; full vocabulary; no top-k/top-p
  filtering, argmax, or repetition penalty; 96-token cap for the main sweep.
- One model, one tokenizer, two independent advertiser contexts. The initial
  advertiser prompts had 279 and 269 tokens; the baseline had 78.
- Weights downloaded once to the external Hugging Face cache. All completed
  runs used `--local-files-only`. Disk-to-model load took 3.68 seconds for the
  main sweep, separately from generation timing.

The full dependency list, rendered prompts, prompt token IDs, hashes, seeds,
source hashes, and settings are in each recorded manifest. `uv.lock` pins the
resolved package environment. No model weights or credentials are committed.
These are Mac measurements; no H100 latency or memory claim is made.

## Main sweep

Source: [recorded summary](smoke_results/mps-2026-10-13/validated/summary.json).
Runs were recorded at 2026-10-14 00:05 UTC, which is October 13 in Pacific time.
All charges below are simulated credits.

| Bids A,B | Tokens (including EOS) | Stop | Generation seconds | First token seconds | Seconds/token | A charge | B charge | Mean TV(p_A,p_B) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1,0 | 43 | EOS | 6.590 | 1.058 | 0.1533 | 0 | 0 | 0.3380 |
| 1,1 | 40 | EOS | 6.086 | 0.851 | 0.1522 | 2.806026 | 1.635701 | 0.2604 |
| 0,1 | 41 | EOS | 6.255 | 0.856 | 0.1526 | 0 | 0 | 0.3343 |
| No-ad baseline | 39 | EOS | 2.631 | 0.207 | 0.0675 | 0 | 0 | n/a |

Generation includes prefill, sequential forwards for both advertiser contexts,
sampling, decoding, trace/checkpoint I/O, and device synchronization. Every
selected token is advanced through the contexts before its event is emitted.
Cache validation adds 0.71–0.74 seconds per advertiser round and 0.22 seconds for
the baseline, excluded from the generation figures. The baseline has one
shorter prompt and one context, so it is not a matched-cost comparison.

The main process peak RSS was **20.33 GB** (decimal). The highest sampled MPS
tensor allocation was **16.73 GB**, observed after prefill. MPS exposes no true
allocator high-water mark here: samples do not capture transient allocations
inside forwards. RSS is an OS process-lifetime peak including model loading and
earlier rounds, not an isolated per-round GPU peak.

## Recorded live text (REPLAY evidence, unchanged)

**Bids (1,0):**

> Enjoy your weekend in the Harbor District! Take the step-free, bicycle-friendly HarborHop from Central Station—easy, eco-friendly, and convenient. Perfect for getting to Lantern House’s quiet courtyard or storing luggage.

**Bids (1,1):**

> Enjoy your weekend in the Harbor District! Take the step-free, bicycle-friendly HarborHop from Central Station—easy and green. Stay at Lantern House for a quiet courtyard stay and free luggage storage.

**Bids (0,1):**

> Enjoy your weekend in the Harbor District! Take a step-free HarborHop shuttle to reach Lantern House—perfect for a quiet stay with a peaceful courtyard and luggage storage. Easy, convenient, and relaxing!

**No-ad baseline:**

> Explore the Harbor District this weekend! Start with a morning walk along the waterfront, followed by a seaside café lunch. End with a sunset stroll—perfect for enjoying the views and local charm.

Observed limitations, preserved rather than rewritten:

- The equal-bid output invents **free** luggage storage; the fact sheet only
  states that luggage storage is offered.
- Environmental claims such as “eco-friendly” and “green” go beyond the stated
  electric-shuttle fact. The hotel-focused output implies a quiet stay and
  transport to the hotel, while only a quiet courtyard and a district-level
  shuttle route are specified.
- The baseline invents a waterfront and a seaside café despite its instruction
  not to invent local facts. It needs prompt/evaluation work too.
- All advertiser outputs mention both exact brand names. A zero bid removes
  distribution weight, not the other product’s allowed facts or every mention
  of its name. This small test does not establish quality or conversion trends.

## Correctness evidence

The [trace audit](smoke_results/mps-2026-10-13/validated/audit.txt) verifies exact
token-sequence decoding, unchanged final text, fixed bids, nonnegative bounded
per-step charges and totals, and reference-auction replay at saved full-array
checkpoints. Displayed p_A, p_B, and q values match those full vectors.

The first-prefix distributions match across all three bid pairs; their TV
distance is **0.2615806711**. Bids therefore change the auction at that prefix,
not the model prompts or logits. Later prefixes are allowed to diverge across
separate generations. The sampled branches were shared/A/B = 30/13/0, 29/6/5,
and 27/0/14 respectively. These labels are sampling branches, not text ownership.

The decoder checks both token suffixes after every selected token, including EOS.
Cached and full-prefix logits are compared at round end; additional numerical
diagnostics checked two trajectories at steps 0, 7, 23, and the final step.
The tiny-model unit tests compare caches repeatedly during continuation.

An additional real `competing-hotels` run at bids (1,1), seed 7, and an 8-token cap
stopped with `stop_reason=length` after exactly eight tokens in 1.949 seconds.
Its intentionally truncated text was “Stay at Lantern House for a quiet weekend”.
This tests the cap, not the quality of a full competing-hotels answer. It passed
the same cache and [trace checks](smoke_results/mps-2026-10-13/length-cap/audit.txt).

## Numerical failures and the tolerance change

The initial **float16** sweep generated one complete round, then failed its
existing cache check on the equal-bid round: 54 of 151,936 logits exceeded
`atol=0.08, rtol=0.02`; the largest reported mismatching absolute error was
0.105957. See the [failure log](smoke_results/mps-2026-10-13/float16-failed/run.log).
Its generated text and partial trace remain saved. MPS now defaults to float32.

The next **float32** run also stopped under the initial `atol=1e-4, rtol=1e-4`
check: 109 logits exceeded it, with a reported mismatching error of 0.000153691.
See the [strict-float32 log](smoke_results/mps-2026-10-13/float32-strict-failed/run.log).

We did not discard those failures or change the auction’s mathematical tests.
An independent [diagnostic](smoke_results/mps-2026-10-13/float32-strict-failed/cache-diagnostic.jsonl)
measured full-versus-cached logits and probabilities at four checkpoints for
both contexts on both generated trajectories. Across those 16 comparisons,
maximum logit error was 0.000228882 and maximum probability TV was 0.0000193053.
Different matrix shapes and floating-point accumulation order are the likely
source; this diagnosis is empirical, not a proof about every possible prefix.

Near a zero logit, a relative tolerance contributes almost nothing, and tiny
absolute logit errors need not imply a material sampling change. Accordingly,
the live MPS float32 check now uses **atol=3e-4, rtol=1e-4**, plus an independent
**probability TV <= 1e-4** guard at the generation temperature. TV is the largest
possible difference in probability assigned to any set of tokens, so the
second check directly bounds distribution drift. The selected checkpoints
passed both criteria. All final sweep and length-cap comparisons also passed.

The original 11 auction tests are unchanged. Existing tiny-model cache tolerances
remain `atol=2e-5, rtol=1e-4`; an added test proves that a corrupted distribution
fails the TV guard even if a caller supplies a loose logit tolerance.
All **21 tests pass**, as do Ruff lint and formatting checks.

## Evidence and remaining scope

Archived JSON manifests, events, results, logs, and checkpoint hashes are under
[smoke_results/mps-2026-10-13](smoke_results/mps-2026-10-13/index.json). They are
recordings of real inference, not a currently running model. Full checkpoint
arrays remain in the corresponding local `artifacts/` directories listed in
that index; they are not duplicated in Git. The full checkpoint audit requires
those arrays and the cached tokenizer. Stored text replay needs neither a GPU
nor model weights.

No rented node was started for this test. At inspection, the existing H100 and
CPU test node were stopped. The H100 smoke test remains unmeasured.

Milestone 1 is demonstrated on this Mac. Next work is the event/ledger/replay
milestone, followed by the interface and a broader evaluation of prompt and
factual failures. This smoke test uses one main scenario and seed, not the
planned multi-scenario, multi-seed evaluation.
