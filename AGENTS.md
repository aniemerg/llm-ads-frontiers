# Project instructions

Read CODEX_BRIEF.md and MECHANISM.md before changing the mechanism.

- Preserve exactly two bidders and one common tokenizer for the initial demo.
- Keep bids out of model prompts and fixed for each generation round.
- Sample probabilities, not averaged logits; do not use argmax or post-mixture
  top-k/top-p filtering in the reference mode.
- Append the exact auction-selected token ID to BOTH advertiser contexts.
- Keep each context's prompt, attention mask, positions, and KV cache separate.
- Retain and extend existing tests; never replace failed tests with new targets
  without explaining the mathematical reason to the human.
- The current test command is `python -m unittest -v` at the project root.
- Use simulated credits. Do not provision paid infrastructure, send transactions,
  or access real wallets without explicit approval.
- Label replay, synthetic data, and live model output distinctly.
- Do not silently repair generated text with another model or claim untested
  latency, model quality, truthful bidding, or on-chain inference verification.
- Keep dependencies, seeds, model revision, tokenizer, prompts, and generation
  parameters in each run manifest. Never commit secrets or model weights.
