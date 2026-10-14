# Token Auction Lab

A two-bidder token-auction reference and a shared-prefix LLM decoding CLI.
Exactly two fictional advertisers use one model and tokenizer, with independent
prompt contexts and KV caches. Every selected token ID is fed to both contexts.
Bids stay fixed for a round and never enter either prompt. All charges are
simulated credits.

The mathematical sampler is unchanged from the supplied reference. The decoder
uses full-vocabulary probabilities at a fixed temperature, without argmax,
logit averaging, or top-k/top-p filtering. The top-token table is for inspection
only. See [MECHANISM.md](MECHANISM.md) for the payment derivation and its limits.

## Install and test

The inference package requires Python 3.11 or newer. With `uv`:

```bash
uv sync --frozen --extra inference --extra dev
source .venv/bin/activate
python -m unittest -v
ruff check .
```

Alternatively, install into a Python 3.11+ virtual environment with
`python -m pip install -e '.[inference,dev]'`. `uv.lock` records exact resolved
versions; pip resolves within the ranges in `pyproject.toml`.

The CPU auction reference only needs `pip install -r requirements.txt`.
Without the inference dependencies, decoder tests are explicitly skipped.
The complete suite has 21 tests: all 11 original auction tests, a prompt test,
and nine decoder tests using tiny randomly initialized Qwen models. These
synthetic models test mechanics and cache correctness, not text quality.

## Run a live smoke test

The proposed model is
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
This command freezes its revision and runs the three bid pairs `(1,0)`, `(1,1)`,
and `(0,1)`, followed by a separate no-ad baseline:

```bash
python -m token_auction_lab.cli \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --max-tokens 96 --temperature 0.8 --seed 7 --baseline \
  --output artifacts/smoke
```

The default device selection is CUDA, then Apple MPS, then CPU. MPS defaults to
float32 after the live float16 run failed cache validation. Override with
`--device` and `--dtype`. The model is loaded once for the bid-pair sweep.
Use `--scenario competing-hotels` for the second preset, and repeat `--bids A,B`
to specify a different sweep. The token cap must be between 1 and 96.
Zero/zero bids are only valid through the no-ad baseline.

Once the model is cached, add `--local-files-only` with the full revision SHA
to run without contacting Hugging Face. Model downloads stay in the normal
Hugging Face cache, outside the repository.

Each uniquely named round directory contains:

- `manifest.json`: model and tokenizer revision, dependency versions, prompts
  and prompt token IDs, source hashes, seeds, fixed bids, and generation settings.
- `events.jsonl`: exact token IDs and cumulative decoded text, auction branches,
  probabilities for displayed candidates, random tapes, and simulated charges.
- `checkpoint-*.npz`: full probability vectors and the frozen generated prefix
  for at most four selected steps. These are inspection data, not model weights.
- `result.json`: unchanged generated text, EOS/length stop reason, timing,
  cache-comparison errors, suffix verification, payments, and memory measurements.

The enclosing `summary.json` contains all completed rounds in the invocation.
An exception leaves any existing manifest and partial trace intact; a directory
without `result.json` is incomplete and must not be presented as a successful run.
Live inference is labeled `live_model` or `live_baseline`; synthetic test runs
use `synthetic_model_test`. Reading a stored trace for display must be labeled
REPLAY. A seed alone does not guarantee identical regeneration on other hardware.

Generation timing includes prefill, sequential forwards for both contexts,
sampling, synchronization, decoding, and trace/checkpoint I/O. The selected token
is fed to both contexts before its event is emitted. Cache validation is measured
separately. Process peak RSS is a process-lifetime high-water mark; CUDA allocator
peak is reset per round. MPS reports current and sampled allocator memory, not a
true transient device peak.

Measured live output and preserved failures are in [SMOKE_RESULTS.md](SMOKE_RESULTS.md).
The completed sweep and an eight-token length-cap run passed trace auditing:

```bash
python -m scripts.audit_smoke artifacts/smoke
```

This audit uses the recorded full checkpoints and cached tokenizer without
model inference. It prints a REPLAY AUDIT label. The source prompts and seeds
remain in each manifest; archived JSON evidence is under `smoke_results/`.

## Scope

This is milestone 1 of [CODEX_BRIEF.md](CODEX_BRIEF.md). There is no UI, credit
reservation ledger, inference service, or blockchain integration. The trace
fields prepare for later inspection and replay work.

This independent implementation covers a narrow two-bidder construction. It does
not establish arbitrary-utility truthful bidding, incentives over complete
answers, advertiser conversion, or production advertising economics. Generated
text is never repaired by a third model. Inspect and preserve failures.
