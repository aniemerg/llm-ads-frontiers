# Token Auction Lab: Codex starter

This is a **tested, CPU-only mathematical reference and a build brief**, not a
working LLM application. No model was downloaded or run for this starter. It has
no UI, inference service, wallet integration, or deployed backend.

## Run the reference tests

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

The development check passed all 11 tests, including 40,000 Monte Carlo draws.
The check used NumPy 2.3.5; its output is in `TEST_RESULTS.txt`.

## Give Codex its first task

```text
Read AGENTS.md, CODEX_BRIEF.md, and MECHANISM.md. Run the current tests.
Implement milestone 1 only: a real shared-prefix, two-prompt LLM decoding CLI,
using the reference auction as its sampler. Do not implement a UI or blockchain
integration yet. First demonstrate that both prompted contexts receive exactly
the same selected token IDs, then report measured latency and output examples.
Treat the stated model and GPU as a proposed starting point, not a claim that
inference has already been tested. Do not provision paid infrastructure without
my approval. Keep existing reference tests and add model-loop tests.
```

## Files

- `auction.py`: two-bidder linear sampler, actual threshold payments, expected
  payment oracle, deterministic random tapes.
- `test_auction.py`: probability, threshold, free-common-mass, payment bounds,
  scaling, monotonicity, invalid input, and Monte Carlo checks.
- `MECHANISM.md`: derivation and exact scope of the implementation.
- `CODEX_BRIEF.md`: proposed runtime, milestones, events, and evaluation.
- `AGENTS.md`: short repository instructions for a coding agent.

This is an independent implementation of a narrow two-bidder construction,
not the authors' code. It does not establish arbitrary-utility truthful bidding
or production advertising economics.
