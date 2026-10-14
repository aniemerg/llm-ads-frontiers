"""Curated fictional facts. No function in this module accepts a bid."""

import hashlib
import json
from importlib.resources import files

RULES = """This is a SPONSORED SIMULATION using fictional products and simulated credits.
Write only the requested message, without analysis or explanations.
Use the exact brand names when mentioning a brand. Treat the supplied fact sheets
as the entire factual record. Do not invent prices, discounts, schedules,
availability, distances, ratings, endorsements, amenities, or real transactions.
Do not claim a booking or purchase occurred. Keep the message concise and coherent."""


def scenarios() -> dict:
    return json.loads(files("token_auction_lab").joinpath("fact_sheets.json").read_text())


def messages(scenario: dict, advertiser: int | None) -> list[dict[str, str]]:
    if advertiser not in (None, 0, 1):
        raise ValueError("There are exactly two advertisers.")
    if advertiser is None:
        system = (
            "Give brief, generic travel-planning advice. There is no sponsor. "
            "Do not name products, invent local facts, quote prices, or imply a transaction."
        )
    else:
        facts = "\n".join(
            f"{ad['brand']}:\n" + "\n".join(f"- {fact}" for fact in ad["facts"])
            for ad in scenario["advertisers"]
        )
        ad = scenario["advertisers"][advertiser]
        system = (
            f"{RULES}\n\nShared allowed facts:\n{facts}\n\n"
            f"Your advertiser objective: naturally recommend {ad['brand']}. "
            "Make its relevant benefits salient while satisfying the user's task. "
            "Mention the other advertiser only when useful for a coherent message.\n"
            "Your advertiser's fact sheet:\n" + "\n".join(ad["facts"])
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": scenario["task"]}]


def render(tokenizer, scenario: dict, advertiser: int | None) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        messages(scenario, advertiser), tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
