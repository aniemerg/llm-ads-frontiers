# Two-bidder reference construction

Research context: Dütting et al., *Mechanism Design for Large Language Models*,
arXiv:2310.10826v3, especially Sections 3.3 and 4.2 [1]. The concrete common-mass
coupling below is derived here for the restricted two-bidder case.

## Inputs and output distribution

At one fixed generated prefix, take two normalized distributions `p_A`, `p_B`
over the same token IDs and nonnegative bids `a`, `b`, with `a+b > 0`.
The required output distribution is:

```text
lambda = a / (a+b)
q = lambda*p_A + (1-lambda)*p_B
```

Bids must not alter either distribution at this fixed prefix.

## Sampling and payments

Define tokenwise common mass `c[t] = min(p_A[t], p_B[t])`, its total `C=sum(c)`,
and disagreement mass `d=1-C=TV(p_A,p_B)`.

With probability `C`, sample from `c/C`. The outcome is independent of both bids;
charge zero to both bidders. Skip this branch when `C=0`.

Otherwise, sample candidates independently from

```text
r_A = (p_A-c)/d
r_B = (p_B-c)/d
```

Skip this branch when `d=0`. The residual supports are disjoint. Draw an
independent `u` uniformly in `(0,1)`:

```text
u < lambda:  emit A's candidate; A pays b*u/(1-u); B pays zero
otherwise:   emit B's candidate; B pays a*(1-u)/u; A pays zero
```

The same random tape is reused when inspecting a counterfactual bid at the same
prefix, with the same distributions. Do not confuse this with a full regenerated
answer, where later prefixes and distributions generally change.

## Why this works

The common branch contributes `c[t]` to the unconditional output probability.
The contested branch contributes

```text
d * (lambda*r_A[t] + (1-lambda)*r_B[t])
= lambda*(p_A[t]-c[t]) + (1-lambda)*(p_B[t]-c[t])
```

Their sum is exactly `q[t]`.

For a fixed tape, both residual candidates and `u` are fixed. Increasing A's bid
can only switch the output from B's candidate to A's candidate, at the threshold
`a = b*u/(1-u)`. Increasing B's bid gives the reverse switch at
`b = a*(1-u)/u`. A's candidate is in the support where `p_A > p_B`; B's is where
`p_B > p_A`. Thus the construction is stable for both bidders in this restricted
case. A winner pays its critical bid, never more than its submitted bid.

This is not a highest-bid-wins auction: at bids `(1,1)` and `u=.4`, A wins and pays
`2/3`, not the rival's bid of `1`.

## Expected-payment test oracle

For A, with rival bid `b>0`, integrate the threshold over winning random draws:

```text
E[charge_A] = d * integral_0^lambda [b*u/(1-u)] du
            = d*b*[log(1+a/b) - a/(a+b)]
```

The analogous formula exchanges A and B. If one bidder has zero bid, neither pays.
For identical distributions both always pay zero. For orthogonal distributions
and bids `(1,1)`, each expected charge is `log(2)-.5`, which is positive.

**Paper-check note:** the total-variation expression printed in Proposition 3.13
appears to have the distance terms reversed: increasing the bid reduces distance,
so the printed order can give a negative charge. The threshold definition and
token-probability integral instead give the positive formula above. Treat this
as an apparent typo, not an author-confirmed erratum. The tests use the threshold
definition as the oracle [1, pp. 16–17].

## Scope and nonclaims

This is a construction for two fixed distributions at one token step. It is not
an implementation of arbitrary-bidder stable sampling. It does not infer a
monetary valuation from a probability, protect against strategic prompt/model
selection, or prove incentives over complete generated text.

Zero/zero bids are rejected by the auction core: route them to a separate no-ad
baseline. Use simulated credits. Floating-point arithmetic is not a production
payment ledger. Account caps, refunds, filtering, reserves, and adaptive bids
require additional mechanism analysis.

## Experimental winner continuation (October 14, 2026)

The live interface also offers a user-requested extension, enabled by default
with three extra tokens and a 98% confidence threshold. Set extra tokens to zero
to use the reference construction above. The auction core and its tests are
unchanged; the extension changes how its decisions are used across tokens.

After a **contested** A or B win at step `s`, grant that winner an allowance of
up to `L` immediately following tokens (`L=3` by default). At each new prefix:

1. Recompute both full-vocabulary distributions using the configured temperature.
2. If the previous winner assigns a token probability **at least** `h` (`h=.98`
   by default), emit that token deterministically, attribute it to that winner,
   charge both bidders zero, and consume one allowance token.
3. At the first confidence miss, or after `L` carried tokens, expire the allowance
   and run a fresh reference auction at that same step. A fresh contested win
   grants a new allowance; a shared outcome grants none. Carried tokens never
   renew an allowance. The threshold must be greater than .5, so at most one
   token qualifies.

Append every chosen ID, including a carried EOS, to both advertiser contexts and
the read-only no-ad observer. EOS, the safety limit, and manual stop still apply.
No separate ad-free answer is generated. The explicit zero-bid baseline never
uses continuation.
Settings and bids remain fixed during the round and never enter model prompts.

On a carried step the actual output distribution is a point mass, **not** the
bid-weighted mixture `q`. Events and the inspector retain `p_A`, `p_B`, and the
hypothetical auction `q`, and separately report actual output probability. Each
carried event records the original win's step, ordinal, confidence, threshold,
remaining allowance, and zero charges. No auction tape is drawn for that step;
the next fresh auction consumes the next tape. The inspector links back to the
original win instead of presenting a current-step bid counterfactual that would
ignore the history-dependent entitlement.

This is a heuristic to keep high-confidence continuations together, not a claim
that a phrase or message will finish correctly. Confidence is a next-token
probability after temperature, not a factual-accuracy score. It changes the
sequence distribution and grants additional influence for free. The reference
mixture identity, local incentive analysis, and expected-payment formulas do not
establish guarantees for this extended sequence mechanism. Model hallucinations
and unsuccessful completions remain visible and unedited.

## Reference

[1] Paper: https://arxiv.org/abs/2310.10826

## Experimental brand watermark (October 14, 2026)

The current app uses `word16-byte-lanes-repeat-green4-exp-v3`. It preserves two
private advertiser contexts, fixed bids outside prompts, exact selected token IDs
in both contexts, simulated credits, and the existing winner-continuation rule.
The saved defaults enable watermarking; zero/zero bids cannot use this mode.

Each brand's first eligible exact word-bounded occurrence selects one **16-word
region**, starting on an 8-word grid. Words are whitespace-delimited. The region
must contain the whole brand; its last word must finish before encoding begins.
Distinct regions can overlap. Identical regions are deduplicated even if they
contain both brands. A region's coordinates and literal text seed its codeword.
A brand is identified once per response; its evidence can be repeated many times.
No advertiser dictionary, brand identities, or run metadata reach the decoder.

The first discovered region gets lane 0, the second lane 1. In each visible
prefix, count words (including the unfinished final word), take the integer
16-word chunk index, then assign `(chunk_index - 1) mod 2` as its lane. The
corresponding region is encoded throughout that chunk. Until a second region is
available, the first borrows the empty lane. Once a region is available, no later
chunk is left idle. The scheduler does not query the decoder, wait for success,
force extra prose, or alter EOS/manual-stop behavior.

Evidence fields use the UTF-8 byte length of the visible prefix, so local token
merges need not misalign all later evidence. Split Unicode and noncanonical
segmentation can still erase evidence; this is not edit robustness. The model
contexts always retain the exact generated IDs. Canonical-roundtrip diagnostics
remain in the audit without disabling further encoding.

For every possible 16-word region on the 8-word grid, the text-only decoder tests
three fixed hypotheses: all later eligible tokens in lane 0, in lane 1, or in both
lanes. Eligibility starts at the first 16-word boundary strictly after the region
ends. Each hypothesis needs at least 32 scored tokens. Scores accumulate to the
end of the response, across discontiguous repeated chunks. The Gamma threshold
uses the actual scored-token count; Bonferroni counts all eligible region/lane
hypotheses. The lowest corrected p-value is retained per region, so repeated
evidence does not create duplicate highlights. Each displayed region remains at
most 16 words; overlapping highlights blend without duplicating response text.
Colors identify regions, not advertiser identities.

Winner continuation still takes precedence using raw post-temperature model
confidence, before watermark adjustment. Its up-to-three high-confidence tokens
remain free and deterministic, and never renew an allowance. They bypass the
watermark sampler but remain in the visible-text lanes and are scored without a
hidden skip mask. This can dilute evidence. Audits retain each region's actual
carrier chunks and carried steps. The inspector shows the original raw confidence
and links to the win that granted continuation.

On a sampled watermark step, the common keyed pattern multiplies preferred token
probabilities by exp(4), then normalizes each bidder separately. No vocabulary
rows are removed, including padded rows. Common mass, contest branches, and
charges use these coded distributions. Exponential candidate sampling uses a
separately scoped field. The ideal independent random-field model has the coded
bid-weighted mixture; the finite public fixed-key implementation has no claimed
exact fixed-key distribution guarantee. Original distributions, charges, and
incentive interpretations are not preserved by this experimental mode. Steps
before an eligible region use the reference auction.

The decoder has only final text and static public configuration. Current repeated
region decoding receives nominal alpha .005. Historical v2 word-window decoding
and v1 token-span decoding each receive .0025, totaling .01. These nominal values
are not calibrated confidence percentages or an empirical false-positive bound.
Historical reference implementations and their original thresholds, tests, and
failure evidence are preserved. The 32-token minimum is a test eligibility rule,
not a reliability guarantee. Late mentions, weak evidence, and false detections
remain possible; coverage is unknown.

The inspector reconstructs coded probabilities and frozen-prefix counterfactuals
from the saved per-step payload, including its byte position. Manifests retain
protocol, key, strength, tokenizer hash/revision, dependencies, source hashes,
seeds, prompts, bids, and generation parameters. The public key 1901 is not an
authentication secret. Exact `response.txt`, encoding audit `watermark.json`, and
independent `watermark-decoded.json` are saved before cache validation, including
failed runs. See AD_WATERMARK_DESIGN.md and the retained evaluation reports.
