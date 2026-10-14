"""Scripted synthetic probability vectors, not a language model."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from auction import RandomTape, TwoBidderAuction
from experiments.ad_stego.prototype import (
    Config,
    Span,
    SyntheticTokenizer,
    message_seed,
    sample,
)


def generate(*, seed=7, key=1901, bids=(1.0, 1.0), marked=True, tokens=224,
             entropy="high", occurrences=((8, (0, 1)), (76, (2, 3, 4))),
             config=Config(), tokenizer=None, background_ids=None):
    """Two bidders, one tokenizer, fixed bids; every ID advances both contexts.

    Scheduled names are synthetic point-mass distributions for BOTH bidders.
    No text is inserted or repaired after the auction. Context lists here are
    mechanics fixtures, not KV caches or real prompt-conditioned LLM states.
    """
    tokenizer = tokenizer or SyntheticTokenizer()
    rng = np.random.default_rng(seed)
    size = tokenizer.vocab_size
    active = np.arange(5, size) if background_ids is None else np.asarray(background_ids)
    if entropy == "medium":
        active = np.array([11, 41, 97, 149, 199, 233])
    elif entropy == "point":
        active = np.array([11])
    elif entropy != "high":
        raise ValueError("Unknown entropy regime.")
    pa, pb = np.zeros(size), np.zeros(size)
    pa[active] = np.linspace(1.0, 3.0, len(active))
    pb[active] = np.linspace(3.0, 1.0, len(active))
    pa /= pa.sum()
    pb /= pb.sum()
    schedule = {start+j: token for start, phrase in occurrences
                for j, token in enumerate(phrase)}
    target_spans = [Span(start, start + len(phrase)) for start, phrase in occurrences
                    if start + len(phrase) <= tokens]
    phrases = tuple(tuple(phrase) for _, phrase in occurrences)
    contexts = [[], []]
    queue, seen, events, sent = [], set(), [], []
    current_message = None
    for step in range(tokens):
        if step % config.block_tokens == 0:
            current_message = None
            if marked and queue:
                span = queue.pop(0)
                surface = tokenizer.decode(contexts[0][span.start:span.end])
                current_message = message_seed(span, surface)
                sent.append({"span": asdict(span), "carrier_start": step,
                             "complete": step + config.block_tokens <= tokens})
        p_a, p_b = pa, pb
        if step in schedule:
            p_a = np.zeros(size)
            p_a[schedule[step]] = 1
            p_b = p_a.copy()
        tape = RandomTape.draw(rng)
        if current_message is None:
            result = TwoBidderAuction(p_a, p_b).resolve(*bids, tape)
        else:
            result = sample(p_a, p_b, bids, tape, key=key,
                            message=current_message, step=step)
        for context in contexts:
            context.append(result.token_id)
        events.append(asdict(result))
        for phrase in phrases:
            if len(contexts[0]) >= len(phrase) and tuple(contexts[0][-len(phrase):]) == phrase:
                span = Span(step + 1 - len(phrase), step + 1)
                if span not in seen:
                    seen.add(span)
                    queue.append(span)
    assert contexts[0] == contexts[1]
    text = tokenizer.decode(contexts[0])
    assert tokenizer.encode(text) == contexts[0]
    return {
        "mode": "synthetic_probability_model", "text": text, "token_ids": contexts[0],
        "expected_spans": [asdict(span) for span in target_spans], "sent": sent,
        "events": events, "shared_suffix_verified": True,
    }
