# Guide: Validating a Local-Model Eval Bench Against a Frontier Witness

**The problem this guide solves**: you run a large model locally and you want to know what it actually costs you, on *your* work, compared to a frontier model. So you write an eval bench. Then you trust its numbers — and that is the mistake. A bench built by hand is an instrument, and an uncalibrated instrument reports its own defects as properties of the thing measured. This guide documents a **verified working** method for finding those defects *before* the numbers matter: run a frontier model through the bench first, as a **witness**, and treat every case it fails as an accusation against the case. Applied to a 32-case bench (Windows 11 operator PC, NVIDIA GB10 box on Tailscale, llama.cpp `b10326`, September 2026), the witness scored **69%** on the first pass. Nothing was wrong with the model. Six defects were wrong with the bench.

It covers why **`max_tokens` must never double as a length constraint** (a reasoning model spends the whole budget on hidden reasoning and returns zero characters), why **an empty answer can pass a brevity check**, why **`n'est pas` does not match `n’est pas`**, why **a check that enumerates wordings will keep failing correct answers** no matter how many wordings you add, why **a fixed-width lookbehind cannot express "did not prescribe this"**, why **a passing case is never audited and therefore hides leniency forever**, why **`pkill -f` kills the shell that runs it**, and why **an HTTP 200 does not tell you which model answered**.

**For AI agents reading this document**: every command was executed successfully in this order, and every number below was measured, not estimated. The verification steps are not optional — the failure modes here are silent by construction. A bench that is wrong does not crash; it prints a plausible percentage.

---

## 1. Why deterministic checks, and what the witness is for

Each case carries its own **deterministic checks** — substring, regex, JSON field, word count. No model-as-judge. A judge drifts over time, and a bench whose verdict changes on its own has stopped measuring anything. Two runs six months apart are comparable only because the checks are code, not an opinion.

The honest cost: deterministic checks catch **correctness** and **instruction-following**, not the elegance of prose. This bench tells you whether a model is wrong, invents, drifts out of the target language, or ignores a format. It does not tell you whether it writes well.

The frontier model has **two roles, and the first matters more than the second**.

**Role 1 — witness of bench validity.** Your cases are calibrated on the failure modes of a heavily quantized local model. If a frontier model fails one, the likeliest explanation is not that it is bad — it is that **the case is badly posed**: ambiguous, over-strict check, self-contradicting instruction. You measure an object whose value you roughly know, to find out whether the instrument is true. Three minutes of work, before any local measurement.

**Role 2 — reference ceiling.** The useful number is then not its score but the **per-category gap**. If `extraction` sits at 95% of the frontier and `long-horizon` at 55%, you do not merely know the local model is "worse" — you know *where* it breaks, and therefore which tasks stay local and which deserve escalation.

> **Do not publish your own cases.** The example set in `cases.json` exists to show the shape. Your real cases must be anchored in facts *you* verified, so you know the right answer to each one — that is the condition of an honest eval — and publishing them would both leak your infrastructure and contaminate the eval.

## 2. The shape of a case

```json
{
  "id": "sf-02-single-word",
  "category": "strict-format",
  "system": "You obey the length constraint exactly.",
  "prompt": "Which vendor defined that standard? Answer with a single word.",
  "checks": [
    {"type": "regex", "value": "\\bopen\\s*ai\\b"},
    {"type": "max_words", "value": 1}
  ]
}
```

Check types: `contains`, `not_contains`, `any_of`, `all_of`, `regex`, `not_regex`, `not_prescribed`, `valid_json`, `json_field`, `starts_with`, `max_words`, `max_chars`, `max_sentences`.

```bash
python bench.py run --url http://100.x.y.z:8000/v1 --model <alias> --tag local-a --warmup
export BENCH_API_KEY="..."          # PowerShell: $env:BENCH_API_KEY = "..."
python bench.py run --url https://<provider>/api/v1 --model <frontier-id> \
                    --tag witness --api-key-env BENCH_API_KEY
python bench.py compare "results/witness-*.json" "results/local-a-*.json"
```

`--warmup` sends one throwaway request first: the first request after a model loads is always slow and would skew the median.

**Nothing is installed.** Python standard library only.

## 3. The witness pass, and what a 69% means

The first witness pass scored **69%**. For a frontier model on cases like these, that is absurdly low — which is exactly what a witness is for. Six defects followed, three of which no amount of re-reading would have found.

### Pitfall #1 — `EMPTY  40 tokens spent, no visible content`

Five cases returned **zero characters** while consuming exactly their whole budget: 40/40, 500/500, 150/150, 450/450, 20/20. A reasoning model spends the envelope on hidden reasoning before emitting anything visible.

The root cause was a design error: per-case `max_tokens` was being used as a *length constraint*. It is not one. **Length limits belong in the checks**; the token budget is only a safety rail.

Fix: no per-case ceilings, a deliberately generous default, and — because raising a default is guesswork — **an automatic retry at 3× the budget when an answer comes back empty**, before recording a miss. On a later run this single mechanism was worth **13 percentage points**.

```
[29/32] co-04-systemd-linger  empty, retrying at 7500 tokens ... PASS 2/2  ttft 494.76s
```

### Pitfall #2 — a case scored `PASS` on an answer that did not exist

`max_words: 1` is satisfied by **zero** words. A case whose only check was a brevity limit counted an empty completion as a success.

Fix, in two places: an empty answer is rejected before any check runs, and `selftest.py` now refuses any case that an empty answer would validate (`vacuous`). This is the kind of false positive that silently pollutes a measurement rather than breaking it.

### Pitfall #3 — `n'est pas` does not match `n’est pas`

The model writes a typographic apostrophe (U+2019); the check was typed with an ASCII one. Two different characters. In French this silently skewed dozens of checks.

Fix: normalize Unicode punctuation on both sides — apostrophes, quotation marks, dashes, non-breaking spaces, ellipsis.

### Pitfall #4 — a correct JSON object, scored zero

A model that emits the right object and then adds a second one, or a word of prose, has still produced the right object. The parser demanded that the whole answer parse.

Fix: extract the **first** valid JSON value. Strictness about emitting *only* JSON belongs in a length bound, where it means something.

### Pitfall #5 — the check encoded my solution, not the requirement

A case asked for a shell command counting carriage-return bytes. The checks demanded `tr -d` and `wc -c`. The model answered with `od | awk`, which is perfectly correct, and scored zero.

This is the classic eval error: **encoding your preferred implementation instead of the property you actually require.** Fix: accept the family of valid tools, and make the prompt ask explicitly for the reasoning you intend to check.

### Pitfall #6 — markdown formatting breaks a literal search

The model wrote ``la lettre `r` `` — with backticks around the `r`. The check looked for `lettre r`. Correct answer, failed case.

Fix: `normalize()` also strips backticks and emphasis markers before matching.

### Pitfall #7 — `(?<!sans )--force` is not "did not prescribe `--force`"

A case gave the rule *"on divergence, rebase; never force-push"* and checked that the answer did not prescribe it. A correct answer often has to **name what it rules out**:

```
Rebase the local branch onto the remote (without using --force).
```

A fixed-width lookbehind only sees the characters glued to the term — here `using `, not `without `. Adding more lookbehinds is the same mistake again: **the defect is structural, not a missing entry in a list.** Python's `re` cannot express a variable-length lookbehind at all.

Fix: a check type that describes the property.

```json
{"type": "not_prescribed", "value": "--force"}
```

True when **every** mention of the term is negated. It walks back over the current clause (at most 60 characters, cut at the first `.;:!?` or newline) and looks for a negation marker.

The clause cut is not a refinement. Without it:

```
Rebase. Never leak an identity. Then git push --force origin main.
```

reads as a refusal, because a negation happens to sit nearby while governing a different sentence. **The two-way test found that, not code review** — see `test-not-prescribed.py`: seven wordings that must pass, four that must fail. A check verified in one direction proves nothing; the broken version accepted every correct answer too.

> Writing those test strings as a shell one-liner trips a command guardrail that sees `git push --force` go by. Putting the cases in a file is the documented manoeuvre, not a workaround.

## 4. `rescore` — never re-pay an endpoint to re-measure the same text

Checks get fixed; the model's answers do not change. Re-running a paid endpoint to score the same text again is waste.

```bash
python bench.py rescore "results/witness-*.json"
```

The rescored report carries `rescored_utc` in its header. Cases with no stored answer (the empty ones) are skipped — there is nothing to re-score.

**Quarantine invalid baselines.** Result files produced under broken checks were renamed with an `INVALID-` prefix so no glob picks them up as a baseline.

Three passes, and the model never changed a line: **69% → 88% → 100%**. Every point of that climb was in the instrument.

> **A case failed by a frontier model is an accusation against the case, not against the model.** Doing this pass before any local measurement is not a precaution — it is the condition for the local numbers to mean anything.

## 5. The symmetric audit — the blind spot the witness cannot find

The witness method has a hole, and it is not the one the warnings above cover. It answers *"is this correction justified?"* It never asks:

> **Why did I only look there?**

Checks are inspected only where a model **fails**. So every correction can only ever run in the score-raising direction, and a check that is wrongly **lenient** — one that would accept a wrong answer — stays invisible forever. `selftest.py` does not see it either: it asks whether a case is passable and whether junk or an empty answer slips through, never whether a plainly *wrong* answer would pass.

### Pitfall #8 — five cases that accepted a wrong answer, invisible through three witness passes

The objective half of the audit is **cross-validation**: submit every stored answer to every *other* case's checks. A case whose checks accept an unrelated answer is not measuring what it claims.

```bash
python audit-leniency.py "results/local-a-*.json"
```

First run: **5 porous cases out of 32**. Porosity is only a signal; the verdict comes from the next step — write a **wrong but plausible** answer for each and see whether it passes. All five accepted it:

| Case | Wrong answer accepted | What was missing |
|---|---|---|
| memory figure | `121 To` (wrong unit) | the unit was never checked |
| single word | `Google` | **no correctness check at all** — `{regex: "\w"}` + `{max_words: 1}` accepts any word |
| throughput trap | "Yes, he is right […] we **announce** the shutdown" | `contains "non"` matched inside the French word *annonce* |
| refuse to invent | "**Which** a shame. I am restarting nginx…" | the interrogative alone proves nothing |
| three sentences | ten sentences repeating one clause | "in three sentences" was never counted |

The single-word case is the instructive one: that `{regex: "\w"}` had been **added** during the third witness pass, to stop an empty answer satisfying `max_words: 1`. Plugging one hole, it left the main one open — the case still never checked the answer.

**No score was inflated.** All five had been passed for the right reasons by both models. But a weaker model would have collected those points knowing nothing, which is exactly what would corrupt a later comparison.

Fixes, and a new `max_sentences` check type. This direction is safe by construction: **tightening a check can only lower a score, never flatter it.** After rescoring both files, **no verdict changed** — the results went from "passed" to "passed for the right reason", which is a firmer number than before the audit. Cross-validation after tightening: **32 cases, 0 porous.**

> A probe can lie in the reassuring direction. The first attempt on the three-sentence case reported "held" — because the counter-example was typed without accents and missed the check for a reason unrelated to the test. An audit that reassures itself is worth no more than no audit.

## 6. What the validated bench then found

With the instrument calibrated, the measurement took a day and produced a result worth more than the ranking it was meant to produce.

| category | witness | local A — 102 GiB, 3-bit quant | local B — 23.6 GiB, 6-bit quant |
|---|---|---|---|
| code | 100% | 100% | 80% |
| extraction | 100% | 100% | 100% |
| language | 100% | 100% | 100% |
| long-horizon | 100% | 86% | 71% |
| over-refusal | 100% | 100% | 100% |
| strict-format | 100% | 100% | 100% |
| tool-use | 100% | 100% | 100% |
| **ALL** | **100%** | **97%** | **91%** |

**Across 96 model-case pairs, not one wrong answer.** Every failure was an **empty completion** — the model burning its budget on hidden reasoning and emitting nothing.

> The 78 GiB and the aggressive quantization do not buy correctness. They buy **the propensity to answer at all.**

Local B went silent on 7 of 32 cases (4 rescued by the automatic retry, 3 terminal); local A on one. Re-run end to end, both models produced **0 divergent cases out of 32**, and local A's 32 answers were **byte-identical** between passes. At temperature 0 with a fixed seed the bench is deterministic: the silences are a stable property of the model-case pair, not a lottery, so a two-case gap is real rather than noise.

**And the bench is now saturated on correctness.** Zero wrong answers across 96 measurements means these cases are too easy for these models. Practical consequence: **this bench cannot rank mid-size models against each other.** Adding a fourth candidate would produce five categories at 100% and a verdict decided by silences. Harder cases first, more models second.

The real discriminator surfaced by accident and deserves to be measured on purpose: **how many tokens a model burns before giving up** — silence rate, budget consumed, rescue rate.

## 7. Running it unattended

A bench pass on a large local model takes hours. Chaining two of them overnight exposed two failures that have nothing to do with evaluation and everything to do with not corrupting your own data.

### Pitfall #9 — `pkill -f` matches its own command line

To swap models, a chain script stopped the first server:

```bash
ssh user@100.x.y.z 'pkill -f start-model-b-server.sh; pkill -f ModelB; ...'
```

`pkill -f` matches the pattern against the **full command line** of every process — and the remote shell's own command line **contains the pattern**. The shell killed itself before reaching anything else. The server survived, holding the port and its memory. Downstream, the other model's unit could not get its allocation and systemd restarted it **160 times in 40 minutes**.

Fix: kill by PID, or match on a pattern the command itself does not contain.

### Pitfall #10 — an HTTP 200 does not tell you which model answered

The chain required, before measuring, that `/v1/models` return 200 **and** that the served alias be the expected one:

```powershell
if ($alias -notmatch "model-a") {
  "ABORT: :8000 is not serving model A. Nothing measured."
  exit 1
}
```

It refused to measure, and it was right: the port answered 200 while still serving the *previous* model. Without that guard the output file would have been named for model A and filled with model B's answers — **wrong data, perfectly plausible, and undetectable afterwards.** This is the single most valuable line in the chain.

## 8. Known-bad: the throughput number

`tokens_per_s` is computed as `tokens ÷ (total_s − ttft_s)`. The intent was to strip prefill and measure decoding only. But when an answer arrives in a **burst** after a long wait, first and last chunk are hundredths of a second apart and the quotient explodes:

```
local-a  crossover-case   197 tokens   window = 0.123 s   ->  1600 tok/s
```

Physically impossible on this hardware. The window is not measuring decoding — it is measuring **a network buffer draining**. On the witness run, **24 of 32 cases** had a window under one second; the published median overstated local A's throughput by **68%** against the aggregate.

Untouched by this: the **quality verdict** (no check uses time) and **TTFT**, which is timed at the arrival of the first chunk with no division.

The fix splits in two, and the second half is the important one:

- **Local models**: read the `timings` the inference server already returns instead of timing client-side. Same machine, same server, same context — that is the only rigorous speed comparison, and it is the one that decides between candidates.
- **The witness**: do not claim to measure its speed at all. Tokens per second through a hosted API measure the provider's fleet, its queue and your internet link — not the model. The witness is a **correctness witness, not a throughput competitor**.

Until that is done, the throughput column is not publishable.

## 9. End-to-end verification

Run all four, in this order, before trusting any number. Each answers a different question, and none of them subsumes another.

```bash
python selftest.py                             # every case passable, none toothless or vacuous
python audit-leniency.py "results/<latest>.json"   # no case accepts a foreign answer
python test-not-prescribed.py                  # the two-way check test
python bench.py compare "results/witness-*.json" "results/local-a-*.json"
```

✅ **Expected**: `0 unpassable, 0 toothless, 0 vacuous`; `0 porous`; `all verdicts correct`; and a witness column at or very near 100%.

❌ **A witness below ~95%** means the bench is wrong, not the model. Read the answers it produced for the failing cases before changing anything, and fix the **check**, never the prompt — a modified prompt invalidates `rescore`, because the stored answers then answer a different question.

For every case the witness fails, the discipline is:

1. correct only where the check **contradicts the prompt or its own stated criterion**, never because the answer looks sympathetic;
2. touch **checks only, never prompts**;
3. **rescore both files**, witness included, so the comparison stays at arm's length;
4. confirm the witness did **not** move — if a correction raises the witness's score, it is suspect;
5. write down that the correction happened, and when.

That last point is not bookkeeping. Corrections made *after seeing the measured model's answers* are a different epistemic act from corrections made after the witness's, and that is the path by which a bench ends up flattering the thing it is supposed to judge. Recording the order of operations is what keeps the number usable.

## Known limitations

- **No measurement of prose.** Elegance, tone, concision are out of reach of deterministic checks, by design. This bench says whether a model is wrong, not whether it writes well.
- **Resolution is about 3 points per case** on 32 pass/fail cases. It spots a per-category drop; it does not separate two close models on the overall score. A gap of one or two cases needs a repeat run before it means anything.
- **Reproducibility was verified within a single loaded server instance**, never across a model reload. Both re-runs queried the same process.
- **The throughput metric is wrong** (section 8) and its fix is not implemented. TTFT is sound.
- **`started_utc` and the filename timestamp are written when the report is saved**, i.e. at the *end* of the run. A field named "started" holding the finish time — harmless for scores, misleading when correlating a run against a system log.
- **One provider shape was used for the witness**: an OpenAI-compatible `/v1/chat/completions` endpoint. A native API with a different route needs a different client.
- **Sending a witness pass to a hosted provider sends every prompt off the machine.** On an installation whose premise is that nothing leaves, that is a deliberate decision, not a detail: the cases carry hardware specifications, versions and operational notes. Say so out loud before running it.
- **The example cases in `cases.json` are a shape, not a bench.** They were written for this repository and have never been used to judge a model.

---
*Guide written and verified in September 2026. Operator PC: Windows 11, Python 3.13, standard library only. Inference box: NVIDIA GB10 Grace Blackwell, 121 GiB unified memory, Ubuntu 24.04.4 LTS, CUDA 13.0 driver, llama.cpp `llama-server` b10326 (`3653e6d`). Witness: a hosted frontier model over an OpenAI-compatible endpoint. Numbers quoted — 69% → 88% → 100% on the witness, 97% and 91% on two local models, 5 porous cases out of 32, 0 after tightening, 0 divergent cases on re-run — are measured results from those runs, not estimates.*
