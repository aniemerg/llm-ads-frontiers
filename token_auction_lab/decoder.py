"""One model, independent KV states, and an identical generated token suffix."""

import math
from dataclasses import dataclass

import numpy as np
import torch


def probabilities(logits: torch.Tensor, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("Expected a finite full-vocabulary logit vector.")
    # Compute softmax in float32, then explicitly cast and renormalize for the
    # strict float64 NumPy reference. Never filter or truncate the vocabulary.
    p = torch.softmax(logits.float() / temperature, dim=-1).cpu().numpy().astype(np.float64)
    p /= p.sum(dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite next-token probabilities.")
    return p


@dataclass
class Context:
    prefix: tuple[int, ...]
    token_ids: list[int]
    attention_mask: torch.Tensor
    cache: object
    logits: torch.Tensor


class SharedPrefixDecoder:
    """Sequential forwards are deliberate; correctness precedes batching."""

    @torch.inference_mode()
    def __init__(self, model, prefixes: list[list[int]]):
        if len(prefixes) not in (1, 2) or any(not p for p in prefixes):
            raise ValueError("Provide two nonempty prefixes, or one for the no-ad baseline.")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.generated: list[int] = []
        self.contexts: list[Context] = []
        for prefix in prefixes:
            ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=self.device)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=True,
                return_dict=True,
            )
            self.contexts.append(
                Context(
                    tuple(prefix),
                    list(prefix),
                    mask,
                    output.past_key_values,
                    output.logits[0, -1].detach(),
                )
            )
        if len(self.contexts) == 2 and self.contexts[0].cache is self.contexts[1].cache:
            raise RuntimeError("Advertiser contexts unexpectedly share a cache.")

    def distributions(self, temperature: float) -> list[np.ndarray]:
        return [probabilities(c.logits, temperature) for c in self.contexts]

    @torch.inference_mode()
    def advance(self, token_id: int) -> None:
        if not 0 <= token_id < self.contexts[0].logits.numel():
            raise ValueError("Token ID is outside the common vocabulary.")
        # The selected integer is fed directly to every context. There is no
        # tokenization, chat-template application, or per-token chat turn here.
        for context in self.contexts:
            length = len(context.token_ids)
            ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            mask = torch.ones((1, length + 1), dtype=torch.long, device=self.device)
            position = torch.tensor([length], dtype=torch.long, device=self.device)
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position.unsqueeze(0),
                cache_position=position,
                past_key_values=context.cache,
                use_cache=True,
                return_dict=True,
            )
            context.token_ids.append(token_id)
            context.attention_mask = mask
            context.cache = output.past_key_values
            context.logits = output.logits[0, -1].detach()
        self.generated.append(token_id)
        self.assert_shared_suffix()

    def assert_shared_suffix(self) -> None:
        for context in self.contexts:
            if context.token_ids[len(context.prefix) :] != self.generated:
                raise RuntimeError("Advertisers received different generated token IDs.")

    @torch.inference_mode()
    def check_cache(
        self,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        temperature: float = 1.0,
        max_probability_tv: float | None = None,
    ) -> list[float]:
        """Compare cached logits against full-prefix recomputation, without a cache."""
        errors = []
        self.cache_probability_tvs = []
        for context in self.contexts:
            ids = torch.tensor([context.token_ids], dtype=torch.long, device=self.device)
            positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
            actual = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            ).logits[0, -1]
            error = float((actual.float() - context.logits.float()).abs().max().item())
            torch.testing.assert_close(actual, context.logits, atol=atol, rtol=rtol)
            tv = float(
                np.abs(
                    probabilities(actual, temperature) - probabilities(context.logits, temperature)
                ).sum()
                / 2
            )
            if max_probability_tv is not None and tv > max_probability_tv:
                raise AssertionError(f"Cached probability TV {tv} exceeds {max_probability_tv}.")
            self.cache_probability_tvs.append(tv)
            errors.append(error)
        return errors
