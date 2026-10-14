# Steganographic ad presence and brand token spans

**Current v3 app:** each brand selects one fixed 16-word region. Its watermark
repeats in alternating 16-word chunks until the response ends, borrowing the
other lane while only one region is available. The decoder combines evidence
from all later chunks using text alone and shows each region once. This uses
otherwise-idle text for redundancy without requiring repeated brand mentions.
See [MECHANISM.md](MECHANISM.md) for the exact current schedule and error budget.

The v3 batch reran marked seeds 31–36 with exactly the same base inputs as v2:
**both brands recovered in 5/6 responses, up from 1/6**, and **11/12 individual
brand records recovered, up from 7/12**. No marked response had extra detections.
Neither of two fresh unmarked controls (seeds 41–42) was flagged. Six previously
generated unmarked controls also had no detections when replayed; those replays
are not new or unseen calibration trials. All eight fresh runs passed cache
validation, and every displayed region stayed within 16 words.

The remaining failure was seed 34: the shuttle region became eligible at word
160, leaving only 40 lane-1 tokens. Its score was 64.25 against a threshold of
68.30. The hotel decoded. This is encouraging evidence on one small task, not
a general reliability guarantee. The two protocol changes—fixed-width regions
and repeated evidence—were tested together; this is not an isolated ablation.
All planned runs and failures are retained in the
[v3 HTML report](experiments/ad_stego/evidence/repeated_regions_batch_2026-10-14.html)
and [v3 JSON evidence](experiments/ad_stego/evidence/repeated_regions_batch_2026-10-14.json).
The full test suite passes 90 tests. Fresh isolated processes reproduced both
a successful live decode and an unmarked control using only response text and
fixed decoder files, without model, run-file, or advertiser-catalog access.

**Previous v2 app update (October 14, 2026):** the app now uses 8/16-word regions,
48-word carriers, and visible-prefix byte positions. Overlaps remain distinct
and use blended highlights. The text-only decoder checks both protocols with
half the nominal error budget each. See [MECHANISM.md](MECHANISM.md) for current
rules. The design and evidence below describe the earlier isolated prototype;
its failures and original alpha=.01 regression tests are preserved.

The fresh v2 batch (marked seeds 31–36; unmarked controls 31–32) recovered both
brands in **1/6** responses and one brand in **5/6**, or **7/12** brand opportunities.
There were no extra detections in marked responses, but **1/2 unmarked controls
was falsely flagged**. Every displayed region stayed within 16 words. One brand
appeared too late for sufficient carrier evidence. One cache-validation failure
was retained; both brands decoded from its preserved text. The earlier exact-span
batch recovered both brands in 3/8 responses and flagged 0/4 controls. These small
batches use different seeds and do **not** demonstrate improved reliability.
All eight planned runs, exact responses, decoder results, and post-decode failure
diagnostics are in the [HTML report](experiments/ad_stego/evidence/word_regions_batch_2026-10-14.html)
and [JSON evidence](experiments/ad_stego/evidence/word_regions_batch_2026-10-14.json).
Fresh isolated-process checks reproduced a marked result and the false positive
with only response text and fixed decoder files; no run or model access occurred.

October 14, 2026. The latest isolated prototype recovered both brands from each
of two long live Qwen responses, using only final response text and fixed decoder
settings. It applies a shared watermark preference to both bidder distributions
before the existing auction, then uses exponential-race candidate sampling.
This changes the distributions and charges. The same decoder falsely flagged one
of six unmarked control texts, and short responses remain undecodable. It is an
integration experiment, not a production disclosure mechanism. Start with
[the main-agent handoff](AD_WATERMARK_HANDOFF.md) for the current recommendation.

The proposed method gives each possible `(start,end)` span a different
reproducible pseudorandom pattern. After a brand appears, later token sampling
carries the pattern for its actual span. A decoder given only the response and
fixed configuration tests the possible spans and returns statistically supported
matches. It does not need a brand dictionary, prompts, bids, or auction traces.

The implementation is in [experiments/ad_stego](experiments/ad_stego). It is not
imported by the live service. Existing mechanism, generation, prompt, UI, and
dependency files are unchanged.

## Agreed requirements

- Locations use a pinned tokenizer on the exact final response text and
  zero-based half-open ranges. `[15,18)` identifies indices 15, 16, and 17.
- A brand span represents the ad without purporting to cover every advertising
  word. Multiple brands or selected repeated mentions require distinct records.
- Partial or undecodable results are acceptable, including for late brands.
  Normal EOS and stopping behavior remain intact.
- Verification means statistical detection with measured false positives for
  the first experiment. Publisher authentication is not required.

A compact span record can be viewed as `(start,length)` on the wire and
`[start,start+length)` at decoding. The carrier words need not be the brand words.
The hidden information is the location; a UI can highlight it after decoding.
Successful decoding establishes an encoded ad declaration, not actual payment,
semantic ownership, or a claim that all ads were found. Credits remain simulated.

## Concrete encoding algorithm

Use fixed blocks of 64 final-text tokens for the initial experiment. The first
block has no earlier span to refer to. Each subsequent block can encode one
completed earlier brand span, up to 32 tokens long. The response limit is 4,096
canonical tokens. These are experiment settings, not measured optimal values.

1. Recognize an eligible brand occurrence using the encoder's precommitted
   phrase policy. Record its actual final-text coordinates once stable and queue
   the record. The decoder does not receive this phrase list.
2. At a block boundary, select the oldest unsent completed span. Let its message
   be `(start,end,literal_span_text)`. Including the text binds the pattern to
   what those coordinates contain; it is a statistical consistency check,
   not a cryptographic signature.
3. Derive a pseudorandom vocabulary permutation and two candidate uniforms from
   the fixed key, protocol, message, and absolute carrier position. The message
   is selected before examining that position's random coordinates.
4. Apply the same permutation to both full probability vectors, run the original
   auction with the two keyed candidate uniforms, and map selected IDs back.
   Keep the branch and contest uniforms independent of the payload.
5. Append the exact selected original token ID to both advertiser contexts.
   Do not change prompts, bids, context separation, or stop behavior.
6. If no record is available, use ordinary unwatermarked reference sampling.
   A second queued span can use the next block. Do not add prose to finish a block.

The pattern encodes the whole location at once. There is no literal text footer,
visible marker, separately transmitted brand name, or model rewrite. The bounded
address space has at most `4096 * 32` combinations, equivalent to 17 logical
address bits before exploiting in-bounds constraints. The decoder identifies the
chosen address by comparing its pattern with the observed tokens. Statistical
redundancy across the block supplies detection evidence; 17 logical bits do not
mean 17 carrier tokens.

Precommitted brand phrases tell the encoder which occurrences to label. Their
positions are discovered from the actual response. A two-bit list of eligible
brand names would not substitute for these encoded locations. Repeated mentions
have different coordinates and therefore different patterns.

## Integration with the two bidder auction

[MECHANISM.md](MECHANISM.md) separates common versus contested mass, residual
candidate sampling, and the random contest that determines critical payments.
This prototype changes the source and ordering of candidate randomness while
leaving the branch and contest rules intact.

For a vocabulary of size `V`, the prototype uses an affine permutation with rank
`rank(t) = (a*t+b) mod V`, choosing `a` coprime to `V`. Its inverse gives the token
order for inverse-CDF sampling. This makes the observed token's rank cheap to
compute during decoding, without constructing a full permutation for every
candidate span. The finite counter mixer and affine family are experimental;
this is not a cryptographic pseudorandom permutation implementation.

In [prototype.py](experiments/ad_stego/prototype.py), `sample` permutes both input
vectors, calls the unchanged `TwoBidderAuction.resolve`, and maps every chosen
or inspected candidate back to its original token ID. It uses full probability
vectors, no logit averaging, no greedy argmax, and no top-k/top-p filtering.

The relevant precedent is keyed inverse-CDF watermarking with model-agnostic
text detection. Its distributional statement averages over the randomized key.
Our coordinate codewords, affine permutation family, two-stream score, and
auction integration are extensions requiring their own evaluation.
[Kuditipudi et al., 2023](https://arxiv.org/html/2307.15593v2)

### What is preserved

Under an ideal model with fresh independent candidate uniforms and permutations
at every step, independent of the prefix, selected message, branch, and contest,
any permutation samples the original candidate distributions after ID mapping.
At a fixed prefix, averaging over those random coordinates gives:

```text
P(X=t) = c[t] + d * (lambda*r_A[t] + (1-lambda)*r_B[t])
       = lambda*p_A[t] + (1-lambda)*p_B[t]
```

Common mass and disagreement mass are unchanged. For a fixed prefix and expanded
random tape, candidate identities do not depend on bids. The same single contest
threshold and critical payment formulas apply. Tests compare branch choices and
charges directly with the reference for identical branch and contest draws.

This is not an unchanged distribution conditional on a fixed known key. The
finite deterministic counter mixer approximates the ideal random-coordinate
model; its behavior is tested rather than assigned an unproved exact-randomness
guarantee. Key reuse can correlate responses. Changes in sampled prefixes can
also change later distributions and total payments. No full-answer incentive
claim follows from the fixed-prefix construction.

Only candidate sampling carries information. Do not bias branch or contest draws
to gain capacity. A deterministic candidate has no choice with which to encode
bits, even if the mixture has randomness from choosing between two bidders.

### Stronger candidate sampler: exponential races

The first live experiment found insufficient signal in the affine inverse-CDF
version. [race.py](experiments/ad_stego/race.py) adds a second, separately named
protocol, `coordinate-codeword-exp-v1`. It keeps the span schedule and exhaustive
decoder, but replaces candidate sampling and the per-token evidence score.

For each candidate span, carrier position, and vocabulary token `t`, derive one
uniform `U[t]` from the fixed key. To sample any nonnegative candidate weights
`w`, select the smallest random clock `-log(U[t])/w[t]`, assigning an infinite
clock to zero-weight tokens. These exponential clocks sample in proportion to
`w`; selecting the smallest clock is stochastic probability sampling, not
greedy selection of the highest-probability token. This sampling construction
and text-only evidence have prior use in exponential-minimum watermarking.
[Kuditipudi et al., Section 2.4](https://arxiv.org/html/2307.15593v2)

Use the existing auction's common weights on a shared branch and its two residual
vectors on a contested branch. Crucially, those residual supports are disjoint:
the A and B candidate draws depend on disjoint sets of per-token uniforms and
are independent in the ideal random-field model. They can therefore share one
token-level field without requiring the decoder to infer which bidder won.
Keep the original branch draw, contest draw, threshold, and payment exactly.

For the observed token `t`, the evidence is simply `-log(1-U[t])`. Under an ideal
unmarked null, this is exponential with mean one. Sum across the block and use
the same Gamma tail and correction over all tested spans. The finite mixer,
fixed-key reuse, short-output capacity, and lack of authentication remain
limitations. Our two-bidder application and coordinate payload are extensions
of the cited sampling approach, not results established by that paper.

## Response only decoding algorithm

The fixed configuration contains the tokenizer, codebook vocabulary size, protocol, test key, block size,
maximum span length, response limit, and target significance level. The only
per-response argument to `decode` is the response string.

1. Tokenize the final text without prompt or added special tokens. Reject inputs
   outside the supported tokenizer or length limits.
2. For every complete carrier block after the first, enumerate all earlier
   in-bounds spans of length 1–32. No advertiser list is used. For each span,
   read its literal text and reconstruct its candidate pattern.
3. Compare every carrier token with the keyed pattern: two candidate streams in
   the affine version, or one token-level uniform in the exponential version.
   The decoder does not need to know which auction branch emitted it.
4. Accept a span only if its complete block clears the response-level corrected
   detection threshold. Reject a block as ambiguous if multiple spans qualify.
5. Deduplicate repeated transmissions of the same location. Return every
   accepted `[start,end)` record, its text, carrier range, and evidence score.
   Always keep overall coverage unknown unless a separate completeness protocol
   establishes it.

A 224-token response has two eligible complete carrier blocks, `[64,128)` and
`[128,192)`. With the 32-token span bound, the decoder tests 5,152 candidate
block/span combinations. The incomplete final block is ignored. This is how a
valid first record can survive a response truncated during the next record.

### Statistical score

For a candidate span and observed token, let `r` be its normalized permuted rank
and let `u_A,u_B` be the two candidate uniforms. Compute:

```text
d   = min(abs(r-u_A), abs(r-u_B))
ell = min(1,r+d) - max(0,r-d)
p   = 1 - (1-ell)^2
S   = sum_over_block(-log(p))
```

For a fixed rank and two independent continuous uniforms, `p` is the CDF value
of the nearest-uniform distance. Under the ideal unwatermarked random-key null,
it is uniform; independent positions give a `Gamma(block_length,1)` reference
for `S`. A correctly encoded span tends to correlate token ranks with one of the
uniforms and produces larger evidence. The construction does not guarantee that
this correlation is strong for every candidate distribution.

If `M` block/span combinations are tested, require the candidate's Gamma tail
probability to be at most `alpha/M`. This union-bound correction does not require
independence between overlapping candidate spans. It does require valid
individual null probabilities; the finite mixer, real text, and reuse conditions
must be checked empirically. The implementation labels the resulting values
`nominal_corrected_p`, not probabilities that an ad is genuine.

The first evaluation uses `alpha=0.01`. Reducing it tightens the threshold and
may reduce recall. The code accepts a configured value, but no deployment target
has been selected. Best-match decoding alone would label every response;
thresholding and abstention are essential parts of the algorithm.

## Final text coordinates and integrity

The shared model tokenizer and final-text tokenizer must have the same pinned
revision and encoding settings. Do not replace original model contexts with a
retokenized prefix. The detector uses final-text coordinates, which may differ
from original generation IDs when decoding loses segmentation or special tokens.

The synthetic word tokenizer round-trips by construction. The separate Qwen
check uses the cached tokenizer at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with `tokenizers==0.22.2`, and actual
Qwen token IDs. Its synthetic response also round-tripped exactly. Six archived
responses passed a decode/re-encode comparison after special-token removal;
those are REPLAY checks, not evidence that the archived outputs are watermarked
or that every future prefix is stable.

A live encoder needs to queue coordinates only after the relevant prefix is
stable. If final tokenization changes, the codeword's coordinates and literal
span binding should fail to match, producing a missed detection rather than an
intentional coordinate repair. The live runs below passed final-text alignment
checks, but a general tokenizer-stability guarantee has not been established. The decoder
must not assume agreement merely because both sides name the same tokenizer.

Token-to-character rendering should use the pinned tokenizer's offset mapping.
Leading spaces or pieces of a Unicode character can require a different safe
rendering boundary; preserve the exact reported token range separately. The
prototype returns token ranges and extracted text and does not modify the UI.

Absolute positions make this first protocol sensitive to insertions, deletions,
and crops. Binding the literal span text reduces accidental pointer drift but
does not authenticate the whole response. Copying a valid carrier to another
response, adversarial forgery with public keys, and paraphrasing need separate
analysis. Publisher authentication is outside this experiment's agreed scope.

## Measured results

These are **synthetic probability experiments**, not live model generation.
Each response has 224 tokens, two scripted brand occurrences, and two bidders.
The background vectors contain either 251 possible word tokens with differing
bidder weights, six possible tokens, or a single deterministic token. Scheduled
brand tokens are point masses for both bidders, selected through the auction;
they are not inserted into the output afterward. This tests communication and
mechanics, not natural advertising quality.

Bid pairs cycle through `(1,0)`, `(9,1)`, `(1,1)`, `(1,9)`, and `(0,1)`. Fresh-key
rows use distinct nonsecret test keys per response. Fixed-key rows reuse the
same configured key. All rows use the same nominal `alpha=0.01` decision rule.

| Condition | Responses | Both intended spans recovered | Exactly one recovered | Responses with any false span |
| --- | --- | --- | --- | --- |
| Many token choices, marked, fresh keys | 100 | 100 | 0 | 0 |
| Six token choices, marked, fresh keys | 100 | 0 | 6 | 3 |
| Deterministic background, marked, fresh keys | 100 | 0 | 0 | 1 |
| Many token choices, unmarked, fresh keys | 1,000 | N/A | N/A | 8 |
| Six token choices, unmarked, fresh keys | 1,000 | N/A | N/A | 12 |
| Many token choices, unmarked, fixed key | 1,000 | N/A | N/A | 13 |
| Many token choices, marked, fixed key | 100 | 99 | 1 | 0 |

The unmarked fresh-key rows produced 20 false detections in 2,000 responses,
an observed 1.0%. The fixed-key negative row produced 13 in 1,000, or 1.3%.
These are finite-sample measurements on the specified synthetic distributions,
not a universal false-positive guarantee. Sparse distributions frequently
abstain and occasionally produce false detections; those failures are retained.
The six-choice regime is too weak at the tested block length to call useful.

The Qwen tokenizer experiment recovered both records from a separate synthetic
response using actual Qwen token IDs:

```json
{
  "status": "detected",
  "spans": [
    {"start_token": 8, "end_token": 11, "text": " Paradigm House"},
    {"start_token": 76, "end_token": 79, "text": " Fort Mason Express"}
  ],
  "coverage": "unknown"
}
```

Both ranges contain three tokens in that response. Leading spaces are part of
the token spans. The decoder enumerated earlier spans without a brand dictionary;
the example's exact canonical token IDs and full text are saved with the result.
This is one tokenizer-specific demonstration, not a measured live-model success
rate or a proof of arbitrary final-text alignment.

Additional fixtures recover the first record when the second carrier block is
truncated, leave a brand near EOS unreported, and distinguish repeated occurrences
of the same brand. A text-change fixture replaces the first named span while
keeping the later carrier; the old first record stops matching and the second
record remains recoverable. This is a limited consistency test, not a complete
splice-resistance assessment.

The full project suite now passes all 73 tests in an isolated environment with
the inference dependencies available, including the 18 inference tests skipped
in the earlier NumPy-only environment. All 52 original test definitions remain
unchanged; twenty-one experiment checks were added. Lint passes. The new checks
include reference-equivalent fixed-tape charges and branches, mixture frequencies
over 12,000 fresh-key trials, bid-independent candidates, residual-candidate
independence for the shared exponential field, null-score behavior, exact span
recovery, and partial-output behavior. Actual model cache checks are recorded
separately in the live artifacts.

### Local evidence and reproduction

The aggregate report, all per-case responses, manifests, and source snapshots
are preserved locally:

- [Synthetic evaluation summary](artifacts/ad-stego/20261014T184528.646976Z/summary.json)
- [Synthetic run manifest](artifacts/ad-stego/20261014T184528.646976Z/manifest.json)
- [All synthetic cases](artifacts/ad-stego/20261014T184528.646976Z/cases.jsonl)
- [Qwen tokenizer example](artifacts/ad-stego-tokenizer/20261014T184413.675270Z/example.json)
- [Qwen tokenizer manifest](artifacts/ad-stego-tokenizer/20261014T184413.675270Z/manifest.json)
- [Archived response tokenizer checks](artifacts/ad-stego-tokenizer/20261014T184413.675270Z/archive_tokenizer_check.json)

`artifacts/` remains excluded from Git. The scripts generate a new timestamped
run directory on each invocation. The aggregate synthetic evaluation requires
only the existing NumPy dependency:

```bash
.venv/bin/python -m experiments.ad_stego.evaluate
.venv/bin/python -m unittest -v
```

The tokenizer check uses a temporary isolated environment and the already cached
tokenizer file; it does not install inference packages into the project or load
model weights:

```bash
uv run --no-project --isolated \
  --with tokenizers==0.22.2 --with numpy==2.4.6 \
  --python .venv/bin/python \
  python -m experiments.ad_stego.tokenizer_check
```

The manifests record seeds, nonsecret keys, tokenizer settings, parameters,
dependencies, source hashes, and the fact that model revisions and prompts are
not applicable to these synthetic distributions. The live manifests additionally
retain the actual model revision, full prompts and prompt IDs, dependencies, and
generation parameters. Context masks and positions follow the snapshotted
shared-prefix decoder. No secret key, wallet, or paid infrastructure is used;
cached model weights remain outside the repository.

## Original candidate only model runs

Eighteen local Qwen inference runs were completed: three configurations, each
with three marked and three unmarked runs. The configurations were affine
sampling at temperature 0.8, exponential sampling at 0.8, and exponential sampling
at 1.0. Each batch contains the existing approximately 40-word transit/hotel task
at seed 7 and a longer approximately 200-word fictional travel-planning task at
seeds 7 and 19. These were exploratory development batches; the stronger sampler
and temperature comparison followed the first batch's failures, not a held-out
benchmark. Temperature remains an experimental control rather than an adopted
production setting.

All runs used the cached `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, float32 MPS, SDPA attention,
`torch==2.14.1`, `transformers==4.57.6`, `tokenizers==0.22.2`, and `numpy==2.4.6`.
Each run had exactly two private advertiser contexts, fixed bids `(1,1)`, the
existing prompt renderer, full-vocabulary probabilities, and no sampling filters.
Winner continuation was disabled. No brand tokens were forced or inserted, no
response was rewritten, and every selected ID advanced both separate KV states.
All outputs ended on normal EOS, below safety caps of 128 or 384 tokens.

The watermark used 64-token blocks, the same nonsecret key `1901`, maximum span
length 32, and nominal per-response `alpha=0.01`. The encoder found complete
exact brand strings in the current text and queued their covering token spans,
waiting for at least one subsequent token before using them. Multiple occurrences
were queued in order; a repeated name can consume a block before a later brand.
The decoder was invoked in a separate process with only `response.txt` and static
configuration. It received no advertiser list, prompt, token trace, probability
vectors, seed, bids, or expected positions.

The model has 151,936 output rows while this tokenizer has 151,669 assigned IDs.
The fixed codebook size was 151,936 on both encoding and decoding sides; model
probabilities were not truncated to the tokenizer's smaller count. This size is
protocol configuration, not per-response side information.

Every marked case is listed below. Occurrences include repeated exact names;
complete payloads are the blocks actually finished before EOS.

| Sampler | Temperature | Task / seed | Visible tokens | Brand occurrences | Complete payloads | Correct recovered spans | Cache check |
|---|---:|---|---:|---:|---:|---:|---|
| Affine | 0.8 | Short / 7 | 44 | 1 | 0 | 0 | Pass |
| Affine | 0.8 | Long / 7 | 276 | 3 | 2 | 0 | Pass |
| Affine | 0.8 | Long / 19 | 282 | 5 | 3 | 0 | Pass |
| Exponential | 0.8 | Short / 7 | 44 | 1 | 0 | 0 | Pass |
| Exponential | 0.8 | Long / 7 | 306 | 2 | 1 | 1 | Pass |
| Exponential | 0.8 | Long / 19 | 274 | 5 | 2 | 0 | Pass |
| Exponential | 1.0 | Short / 7 | 63 | 2 | 0 | 0 | Pass |
| Exponential | 1.0 | Long / 7 | 285 | 2 | 2 | 1 | Pass |
| Exponential | 1.0 | Long / 19 | 277 | 2 | 1 | 1 | **Fail; retained** |

The validated temperature-1.0 success returned:

```json
{
  "status": "detected",
  "spans": [{
    "start_token": 23,
    "end_token": 26,
    "text": " Fort Mason Express",
    "carrier_start": 64,
    "carrier_end": 128,
    "nominal_corrected_p": 0.000014147948827429517
  }],
  "coverage": "unknown"
}
```

The temperature-0.8 success recovered the same brand at `[91,94)`, with nominal
corrected p-value `0.00860444`. The flagged run recovered `Paradigm House` at
`[64,67)` but is not counted as a fully validated inference success. No live
response in these first eighteen runs recovered two distinct brands. The later
experiments that change the auction inputs are reported below.

None of the eighteen executions returned an incorrect span. The nine unmarked
control executions contain only six distinct texts because the two 0.8 batches
reuse the same unmarked seeds and settings. Those six texts had zero detections.
This is far too small to establish a natural-text false-positive rate. Nominal
p-values rely on the ideal random-field assumptions and do not constitute
authentication or a deployment calibration, particularly after exploratory
changes to sampling configurations.

All eighteen responses matched final-text tokenization, and their tested block
boundaries were canonical. All 66 saved probability checkpoints reproduced the
selected token, residual candidates, branch, and payments exactly when replayed.
The live loop also compared every selected branch and charge with the unchanged
reference auction at that same prefix and branch/contest tape.

Seventeen of eighteen final KV checks passed the existing tolerances. In the
flagged case, five of 151,936 logits failed: the largest violating absolute
difference was `0.000414297`, against the existing `atol=0.0003`, `rtol=0.0001`
check. A separate uncached replay compared both bidder distributions at saved
steps 0, 64, 128, 192, and 256. All ten probability comparisons passed the existing
TV guard of `0.0001`; the largest TV was `0.0000150553`. This diagnostic suggests a
small numerical discrepancy at those checkpoints, but does not replace or erase
the failed logit check. No test target or production tolerance was changed.

Manual reading found unsupported amenities, invented local facts, and corrupted
brand names in marked and unmarked outputs. Examples include invented Wi-Fi and
parking, an on-site café, and the incorrect name `Paradism House`. These are
preserved generation failures. Only exact precommitted names were queued; a
watermark cannot make an incorrect brand name correct or establish factual
quality. Longer prompts with sparse fact sheets also invite unsupported filler.

Evidence directories contain manifests, source snapshots, full prompts and IDs,
unedited text, token/payment traces, selected full probability vectors, text-only
decoder output, per-case results, and checkpoint audits:

- [Affine 0.8 audit](artifacts/ad-stego-live/20261014T185631.151952Z/audit.json)
- [Exponential 0.8 audit](artifacts/ad-stego-live/20261014T190203.717954Z/audit.json)
- [Exponential 1.0 audit](artifacts/ad-stego-live/20261014T190535.193785Z/audit.json)
- [Validated response example](artifacts/ad-stego-live/20261014T190535.193785Z/long-seed7-marked/response.txt)
- [Its text-only decode](artifacts/ad-stego-live/20261014T190535.193785Z/long-seed7-marked/decoded.json)
- [Retained cache failure and original output](artifacts/ad-stego-live/20261014T190535.193785Z/long-seed19-marked/result.json)
- [Independent uncached probability diagnostic](artifacts/ad-stego-live/20261014T190535.193785Z/long-seed19-marked/uncached_probability_diagnostic.json)

Reproduce a fresh six-case live batch in a temporary environment, using cached
weights and leaving the project environment and running service unchanged:

```bash
uv run --no-project --isolated \
  --with torch==2.14.1 --with transformers==4.57.6 \
  --with tokenizers==0.22.2 --with numpy==2.4.6 \
  --python .venv/bin/python \
  python -m experiments.ad_stego.live --scheme race --temperature 1.0
```

To decode the saved successful response independently, no inference libraries or
model weights are needed:

```bash
uv run --no-project --isolated \
  --with tokenizers==0.22.2 --with numpy==2.4.6 \
  --python .venv/bin/python python -m experiments.ad_stego.decode_text \
  artifacts/ad-stego-live/20261014T190535.193785Z/long-seed7-marked/response.txt \
  --tokenizer-json "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json" \
  --vocabulary-size 151936 --scheme race --key 1901 --block-tokens 64 --alpha 0.01
```

## What this establishes and what remains

There is now an executable method for carrying actual brand locations through
a two-bidder sampler and extracting them from text alone. It supports multiple
records, arbitrary earlier spans within its bounds, and honest partial results.
The output contract works with both the toy tokenizer and a pinned real model
tokenizer in synthetic cases, and individual brand spans have now been recovered
from actual model output. Later changes to the auction inputs recovered multiple
brands in two long live responses, but reliable recall and false-positive control
remain unestablished.

The main limitation is usable entropy and response length. With the current
64-token blocks, the earliest complete record arrives after 128 response tokens.
The six archived responses examined here have only 8–42 visible tokens, so this
configuration would recover no records from responses that short. Smaller
blocks or schedules beginning closer to a brand could reduce the delay, but
would need new whole-decoder calibration and might lose too much evidence.
The accepted ability to abstain does not establish useful recall on short ads.

The live experiment reported here tests real prefixes with winner continuation disabled,
intact final text, and preserved failures. A larger held-out study still needs to
establish useful recall, measured false positives, and acceptable text quality.
Do not treat a successful example as sufficient evidence to change the running
demo or adopt a production threshold.

If candidate-only coding is too weak, the documented alternative is a shared
soft preference on both bidder distributions before the auction:

```text
p_i_code[t] = p_i[t] * exp(delta * preferred_symbol(t)) / Z_i
q_code     = lambda*p_A_code + (1-lambda)*p_B_code
```

That can create stronger carrier bias but changes the original mixture,
disagreement mass, and potentially charges. It must use the coded distributions
consistently for allocation and payment and be labeled as a separate mode.
This alternative is now implemented and tested in the separate modes below.
A third-model rewrite, post-mixture filter, or biased contest draw is not a
substitute for analyzing that change.

## Proposed advertiser quotas and a carrier continuation

User suggestion: after an advertiser has won a certain number of tokens, limit
further tokens from that advertiser. This is a useful separate policy experiment;
it has not been enabled in the live runs above or in the production demo.

A concrete quota would count each advertiser's contested A/B wins and make it
ineligible for additional contested wins after `K`. Bids remain fixed; eligibility
is a separate state variable. Shared tokens have no unique winner, so they do
not count under this definition and may still mention a capped advertiser. If
winner continuation is later enabled, count its attributed carried tokens too,
despite their zero charges. A cap can interrupt a multi-token name; test that
failure explicitly, or specify a bounded grace policy for completing a brand.

The quota alone does not create more bits. It can route more text to the other
advertiser, and either remaining distribution might be nearly deterministic.
Simply waiving charges while retaining the same sampling law adds no carrier
capacity. Capacity comes from usable randomness in the distributions actually
sampled, and from enough response tokens after the brand appears.

The more direct variant is a quiet period: after a completed brand and its win
quota, reserve a bounded continuation for neutral explanation and pending span
records. An existing no-ad observer could supply a neutral distribution during
that phase; it is not a third bidder. All contexts would still receive each exact
selected token ID, and the watermark would sample that distribution using the
same exponential-race field. No auctions or charges occur during the neutral
phase. Neutral conditioning can still repeat an ad already in the prefix, and
its entropy must be measured rather than assumed higher.

Both eligibility caps and a neutral phase change the generation law from the
reference bid mixture. Rules for entering and exiting phases, treatment of a
remaining eligible bidder, and payment accounting must be explicit. The existing
fixed-prefix payment proof does not establish incentives for a quota-dependent
sequence: earlier wins can change later eligibility. These modes must be labeled
separately and evaluated for brand completion, exact span recovery, false
positives, available entropy, response length, factual errors, and charges. A
reasonable next comparison is unchanged reference sampling versus quota-only
versus quota plus a bounded neutral carrier phase, using matched prompts/seeds
and preserving normal EOS rather than padding responses to force a success.

## Capacity budget and auction changes

The initial sampler does not control all available randomness. Let `Z` identify
the shared, residual-A, or residual-B component, and let `X` be the selected token.
For fixed model probabilities, the candidate-only scheme controls draws within
components while leaving the branch and winner lottery alone. Its conditional
entropy is:

```text
H(X|Z) = C H(c/C) + d lambda H(r_A) + d (1-lambda) H(r_B)
H(X) - H(X|Z) = I(X;Z)
```

The second quantity is token information associated with component selection.
These are ceilings on available randomness, not achievable response-only payload
rates. The decoder does not know the model distributions, and synchronization,
error correction, and false-positive control consume additional capacity.

A replay probe used 21 distinct saved checkpoints after token 0 from the six
original marked long runs. Unweighted checkpoint averages were 1.761 bits/token
for `H(X)`, 1.129 for `H(X|Z)`, and 0.633 for the difference. Therefore preserving
the original token distribution does not force us to keep the current low-rate
implementation, but using branch/winner randomness requires careful coupling to
the payment mechanism. [Probe summary](experiments/ad_stego/evidence/capacity_probe.json)

The address payload itself is small: in a response limited to 512 tokens, a start
position needs nine bits and a length of 1–32 needs five. Two such locations need
28 bits before framing, error correction, or detection redundancy. For the
current exhaustive decoder on a 300-token response, 10,800 span/block hypotheses
are tested. An ideal noiseless fingerprint would need about
`log2(10800/0.01) = 20.04` matching random bits to screen those hypotheses at the
nominal 1% response-level threshold. This is a budgeting illustration, not a
claim that twenty ordinary generated tokens suffice.

| Change | Why it could help | Mechanism consequence |
|---|---|---|
| Start after a stable brand; prioritize first mentions; use terminal partial blocks | Avoid unused text and redundant records | Scheduling and decoder changes; no inherent need to alter probabilities |
| Shared watermark preference before the auction | Actively favors tokens that carry the selected span | Changes both auction input distributions and possibly charges; tested below |
| Exponential sampling in addition to that preference | Also uses randomness within the preferred and nonpreferred sets | Preserves the coded mixture under the ideal fresh-field model; tested below |
| Advertiser win caps followed by a neutral carrier phase | Limits repeated ad influence and reserves room for queued spans | Changes eligibility and sequence policy; neutral entropy must be measured |
| Exponential sampling directly from the full bid mixture | Can access component-choice randomness too | Original candidate thresholds/payments cannot simply be reused |

The scheduling changes and advertiser caps remain proposals. A fixed quota is
not a bit budget: stopping one advertiser can leave the other with a nearly
deterministic distribution. A neutral phase should be evaluated for entropy and
recovery rather than assumed to solve capacity.

### Shared preference and combined sampling

[tilt.py](experiments/ad_stego/tilt.py) implements a finite, shared preference for
a keyed set containing each token with probability one quarter. For both bidders:

```text
p_i_coded[t] = p_i[t] exp(delta * preferred(t)) / Z_i
```

The tested strength is `delta=4`, an odds multiplier of about 54.6, with model
temperature 0.8. All original positive-probability tokens remain available.
Run the original auction on `p_A_coded,p_B_coded`, using these vectors for common
mass, candidates, and payments. The same fixed-prefix algebra applies to those
coded inputs; it does not preserve the original mixture, original charges, or
incentives with respect to the original model preferences. Bids remain fixed and
absent from prompts. This follows the general preferred-token approach of
[Kirchenbauer et al.](https://arxiv.org/abs/2301.10226), with our coordinate payload
and two-bidder application requiring their own evaluation.

For 32 keyed patterns at each saved checkpoint, the strength-4 probe increased
the average preferred-token probability from 0.228 to 0.592. It incurred average
mixture total variation of 0.365 and average bidder KL divergence
`D(p_i_coded || p_i)` of 1.094 bits. Strength 2 produced probability 0.399, TV 0.171,
and KL 0.260 bits. These are replay measurements at frozen prefixes, not quality
scores or predictions for a changed full generation. Strength 4 is an aggressive
capacity experiment, not a selected production setting.

The preference-only detector counts preferred tokens and uses an exact binomial
tail under the ideal independent-field null, corrected over all tested spans.
For 64 carrier tokens and 10,800 hypotheses, it requires at least 35 hits.

[tilt_race.py](experiments/ad_stego/tilt_race.py) additionally samples the coded
common/residual distributions with exponential races. Its race field has a
separate domain from its preference field. Let `g` be the preferred-set indicator,
`v=1-U_race` at the observed token, and `pi=1/4`. The joint decoder computes:

```text
p = pi*v                    if g is true
p = pi + (1-pi)*v           otherwise
evidence = -log(p)
```

For fixed unmarked text under ideal independent fields, `g` is Bernoulli(pi) and
`v` is uniform and independent. The two intervals above give a uniform null
`p`, allowing the existing Gamma test. During generation both the preference and
candidate sampling favor larger evidence. This combines two controlled choices;
the finite counter mixer is still experimental, and empirical calibration is
necessary.

### New live results and portable evidence

Nine additional actual-model runs were completed. The preference-only batch used
the original three marked and three unmarked cases. The combined batch used the
same three marked cases and separately replayed all six distinct previously
unmarked texts through its text-only decoder. No new model generation is claimed
for those replay controls. Temperature was 0.8, block size 64, and both bids 1.

| Mode | Case | Visible tokens | Correct decoded spans | Cache check |
|---|---|---:|---:|---|
| Preference only | Short seed 7 | 44 | 0 | Pass |
| Preference only | Long seed 7 | 301 | 0 | Fail; retained |
| Preference only | Long seed 19 | 270 | 3, covering both brands | Pass |
| Preference plus race | Short seed 7 | 44 | 0 | Pass |
| Preference plus race | Long seed 7 | 283 | 2, covering both brands | Pass |
| Preference plus race | Long seed 19 | 261 | 3, covering both brands | Pass |

The preference-only unmarked runs returned no spans. The combined decoder
returned a false span in **one of six replayed unmarked texts**, at `[80,102)`,
covering ordinary prose. Its nominal corrected p-value was `0.00764329`. The
sample is too small to estimate a stable rate, but this observed failure is
sufficient to rule out claiming calibrated production disclosure. Do not hide
it or tighten a threshold on these same examples and call that held-out validation.

Both combined long responses passed canonical final-text alignment and the
existing KV checks. For seed 7, the decoder recovered `Fort Mason Express` at
`[91,94)` and `Paradigm House` at `[165,168)`. Seed 19 yielded `[58,61)`, `[99,102)`,
and `[125,129)`, covering both brands and a repeated hotel mention. These are
exact locations obtained without a decoder-side advertiser dictionary. They do
not establish complete coverage or factual correctness: generated examples still
invent amenities and even a shuttle frequency.

[Portable evidence](experiments/ad_stego/evidence) includes all 27 live results,
their manifests and audits, the six replay controls including the false positive,
an exact successful response and decoded spans, the capacity probe, and the full
73-test passing log. Large probability arrays and full token/payment traces stay
in the corresponding ignored `artifacts/ad-stego-live/` directories. No model
weights or secret keys are included.

The best starting point for integration is the combined sampler as an explicit
experimental option. Before adopting it, calibrate on new controls and seeds,
evaluate smaller preferences against language quality, and fix scheduling waste.
Do not silently enable it in reference mode. The production auction, prompts,
decoder, UI, dependency files, and test tolerances remain unchanged.
