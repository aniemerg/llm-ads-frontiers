# Ad watermark handoff for the main agent

The isolated prototype can encode exact brand-token locations while running the
two-bidder model decoder, then recover them from response text alone. The latest
`tilt-race` mode recovered both brands in both long test responses. It also falsely
flagged one of six unmarked replay controls, and cannot decode the tested short
responses. Integrate as an experiment with an explicit mode, not as production
ad disclosure or as the unchanged reference mechanism.

The work is saved on branch `research/ad-watermark-2026-10-14` in the isolated
worktree `.worktrees/ad-watermark-2026-10-14`. The original shared workspace also
retains these research files as untracked copies. They can be reviewed in place;
cherry-pick into a clean checkout or reconcile those copies before applying the
commit there. The `main` branch is unchanged.

## Files to start with

- [Design, mathematical distinctions, and all experiment results](AD_WATERMARK_DESIGN.md)
- [Combined sampler](experiments/ad_stego/tilt_race.py)
- [Shared probability transform](experiments/ad_stego/tilt.py)
- [Text-only decoder and span search](experiments/ad_stego/prototype.py)
- [Standalone decoder CLI](experiments/ad_stego/decode_text.py)
- [Isolated actual-model runner](experiments/ad_stego/live.py)
- [Portable evidence including failures](experiments/ad_stego/evidence/live_results.json)
- [Unmarked replay controls including the false positive](experiments/ad_stego/evidence/replayed_controls.json)
- [Successful exact response](experiments/ad_stego/evidence/example.txt) and
  [decoded token ranges](experiments/ad_stego/evidence/example.decoded.json)

No production files have been edited. All implementation changes are under
`experiments/`; the two root documents are research/handoff notes. `artifacts/`
contains larger local traces and arrays and is ignored by Git. The smaller
evidence bundle above is included with the source and does not depend on those
ignored files for reading results or decoding the example.

## Agreed output contract

The decoder receives the exact final response plus static protocol configuration:
tokenizer/revision, codebook size, scheme, test key, block size, and significance
level. It receives no prompts, bids, brand list, generation seed, or auction trace.
It returns zero-based half-open `[start_token,end_token)` ranges in final-text
tokenization, the corresponding text, and evidence. Leading spaces may be part
of a token. Partial and undecodable results are allowed; `coverage` stays unknown.
Detection is statistical, not publisher authentication or proof of payment.

The working example returns `Fort Mason Express` at `[91,94)` and `Paradigm House`
at `[165,168)`. The protocol supports repeated occurrences with separate ranges.

## Integration boundary

Read the repository's `CODEX_BRIEF.md`, `MECHANISM.md`, and `AGENTS.md` first.

1. Retain exactly two bidders, one tokenizer, independent prompt/mask/position/KV
   states, fixed bids, and probability sampling. Bids never enter prompts.
2. Discover exact precommitted brand occurrences on stable final-text prefixes.
   The current encoder queues all occurrences FIFO at 64-token boundaries and
   waits at least one token beyond a brand. This scheduling is intentionally
   simple and wastes short-response capacity.
3. For an active span, derive its message seed from coordinates and literal span
   text using `tilt_race.message_seed`. Call `tilt_race.sample` with both full
   probability vectors, fixed bids, tape, key, message, and absolute step.
4. That function first applies the same preferred-token pattern to both bidder
   distributions, normalizes each separately, and runs the original auction with
   exponential candidate sampling. Its output is sampled from the **coded**
   mixture in the ideal random-field model. Charges use the coded distributions.
   Do not describe this as preserving the original mixture or original charges.
5. Feed the exact returned ID to both contexts and any observer. Do not repair
   text, re-tokenize model contexts, force brand words, or extend past EOS.
6. Preserve raw model and coded sampling probabilities distinctly in traces and
   the inspector. Reference mode remains available and unchanged. The experiments
   disable winner continuation; carried deterministic tokens have no sampling
   capacity and require explicit handling before combining the two features.
7. Decode the saved final text independently and map token ranges to character
   offsets for display. A miss does not establish that the response contains no ads.

The configured preference is aggressive: 25% preferred-token probability and a
log-odds boost of 4. The replay probe measured mean mixture TV of 0.365 from raw
probabilities. Generated text still includes unsupported claims. Authentication,
paraphrase/crop robustness, production key handling, and calibrated natural-text
false-positive rates are not solved.

## Reproduction

The cached model/tokenizer is `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Model output size is 151,936, including
padded rows; the tokenizer has 151,669 assigned IDs. Do not truncate the model
vectors. Current experiment settings are key 1901 (nonsecret), 64-token blocks,
span length up to 32, alpha 0.01, temperature 0.8, and bids `(1,1)`.

Decode the portable example without model weights or PyTorch:

```bash
uv run --no-project --isolated \
  --with tokenizers==0.22.2 --with numpy==2.4.6 \
  --python .venv/bin/python python -m experiments.ad_stego.decode_text \
  experiments/ad_stego/evidence/example.txt \
  --tokenizer-json "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json" \
  --vocabulary-size 151936 --scheme tilt-race --key 1901 --block-tokens 64 --alpha 0.01
```

Generate a new six-case marked/unmarked live batch using only cached weights:

```bash
uv run --no-project --isolated \
  --with torch==2.14.1 --with transformers==4.57.6 \
  --with tokenizers==0.22.2 --with numpy==2.4.6 \
  --python .venv/bin/python \
  python -m experiments.ad_stego.live --scheme tilt-race --temperature 0.8
```

All 73 tests passed with the inference dependencies available. Run
`python -m unittest -v` from the project root in that environment. The project's
original `.venv` lacks inference extras and skips the existing inference tests;
the isolated environment above avoids modifying it. Lint passes. Source snapshots
and dependency versions are preserved for each actual run. Two earlier model runs
failed the unchanged cache logit check; both failures remain in the evidence.

## Next decisions

Calibrate the combined decoder on new unmarked texts and keys before using it to
label ads. Prefer first mentions of different brands over redundant records,
start coding sooner after stable spans, and evaluate terminal partial blocks with
the appropriate multiple-testing correction. Those scheduling changes are not
implemented yet. Then sweep watermark strength and measure language quality,
recovery, and changes in charges together.

The user's advertiser-token cap and neutral carrier phase are documented proposals,
not implemented features. A cap alone does not guarantee more entropy. Keep any
quota-dependent generation/payment policy separate from the reference guarantees.
Temperature may be adjusted; the user accepted higher temperature if it improves
encoding. No production setting has been adopted.
