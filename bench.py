#!/usr/bin/env python3
"""Quality + throughput bench for OpenAI-compatible endpoints (llama.cpp).

Standard library only -- no pip install, runs on Windows and on the Spark.

Two things are measured on every case:
  * quality  -- deterministic checks declared with the case (no judge model,
                so the verdict is reproducible and cannot drift)
  * speed    -- time to first token, decode rate, token counts

Usage
  python bench.py run  --url http://100.x.y.z:8000/v1 \
                       --model <local-model-alias> --tag local-a
  python bench.py run  --url http://100.x.y.z:8001/v1 \
                       --model <second-model-alias> --tag local-b --category long-horizon
  python bench.py diff results/local-a-*.json results/local-b-*.json

Exit status is 0 unless the run itself failed (transport errors on every case).
A low score is a result, not an error.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES = os.path.join(HERE, "cases.json")
DEFAULT_OUT = os.path.join(HERE, "results")

# Windows consoles still ship legacy code pages; force UTF-8 so accented
# prompts and French output do not explode mid-run.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def call_model(url, model, messages, max_tokens, temperature, seed, timeout,
               api_key=None):
    """Stream one completion. Returns a dict; never raises for HTTP errors.

    Frontier endpoints (OpenRouter, OpenAI, ...) need a bearer token and are
    pickier than llama.cpp about optional fields: a 400/422 is retried once
    without `seed` and `stream_options`, which some providers reject outright.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed

    result = _attempt(url, payload, timeout, api_key)
    if result.get("error", "").startswith(("HTTP 400", "HTTP 422")):
        trimmed = {k: v for k, v in payload.items()
                   if k not in ("seed", "stream_options")}
        retry = _attempt(url, trimmed, timeout, api_key)
        if "error" not in retry:
            retry["degraded"] = "provider rejected seed/stream_options"
            return retry
    return result


def _attempt(url, payload, timeout, api_key):
    endpoint = url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    ttft = None
    chunks = []
    deltas = 0
    usage = {}
    finish = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    # Why the model stopped matters as much as what it said. A
                    # "length" stop is a truncated answer, not a wrong one, and
                    # scoring it as wrong punishes the model for the budget.
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        chunks.append(piece)
                        deltas += 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return {"error": "HTTP %s: %s" % (exc.code, detail)}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"error": "transport: %s" % exc}

    elapsed = time.perf_counter() - started
    text = "".join(chunks)
    out_tokens = usage.get("completion_tokens") or deltas
    decode_window = max(elapsed - (ttft or 0.0), 1e-9)

    return {
        "text": text,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": out_tokens,
        "tokens_per_s": round(out_tokens / decode_window, 1) if out_tokens else None,
        "token_source": "usage" if usage.get("completion_tokens") else "delta-count",
        "finish_reason": finish,
        # Some providers omit finish_reason on a stream. Hitting the budget
        # exactly is the fallback signal, and it is right far more often than
        # a model that happens to stop on the last allowed token.
        "truncated": finish == "length" or (
            finish is None and out_tokens and out_tokens >= max_tokens),
    }


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

# Models write French with typographic punctuation; checks are typed with the
# ASCII forms. Without this, "n'est pas" never matches "n’est pas" and the case
# fails for a reason that has nothing to do with the model's answer.
PUNCT = {
    "’": "'", "‘": "'", "‛": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"',
    # Dashes and hyphens. U+2011 is the nasty one: a NON-BREAKING HYPHEN is
    # pixel-identical to the ASCII "-", and a model writing "lui‑même" with it
    # silently fails every check typed with a plain hyphen. Same family as the
    # typographic apostrophe, found the same way -- by a check that refused a
    # correct answer for no visible reason.
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "‒": "-", "⁃": "-",
    " ": " ", " ": " ", " ": " ",
    "…": "...",
}


def normalize(text):
    for src, dst in PUNCT.items():
        text = text.replace(src, dst)
    # Models wrap technical terms in markdown: "la lettre `r`" must match a
    # check written as "lettre r". Emphasis markers go the same way.
    return text.replace("`", "").replace("**", "").replace("*", "")


def strip_fence(text):
    match = FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def first_json(text):
    """Parse the first JSON value in the text.

    A model that emits the right object and then adds a second one, or a word
    of prose, has still produced the right object. Strictness about *only*
    emitting JSON belongs in a length check, not in the parser.
    """
    body = strip_fence(text)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(body):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(body[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in the answer")


def dig(obj, path):
    """Walk a dotted path; list indices allowed (a.0.b). Returns (found, value)."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            if not part.isdigit() or int(part) >= len(cur):
                return False, None
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        else:
            return False, None
    return True, cur


# Markers that turn a mention into a refusal. Deliberately strong and
# unambiguous: a weak marker ("no", "not") would excuse a real prescription
# that merely happens to sit near an unrelated negation.
NEGATIONS = (
    "sans", "jamais", "ne pas", "pas de", "pas d'", "aucun", "aucune",
    "evite", "eviter", "évite", "éviter", "proscri", "interdit", "exclu",
    "surtout pas", "never", "without", "avoid", "forbidden",
)

# A sentence ends at .!? followed by whitespace or the end of the text. The
# trailing context matters: it keeps "ex. `nginx.service`" and "3.7 To" from
# counting as sentence ends, which a bare split on [.!?] would do.
SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")

# A negation only excuses a term inside its own clause, so the look-back stops
# at the nearest boundary. Without this, "Jamais de fuite d'identite. Ensuite
# git push --force" reads as a refusal because "jamais" happens to sit nearby
# -- it governs a different sentence entirely. Found by the two-way test.
CLAUSE_BREAK = re.compile(r"[.;:!?\n]")

# How far back to look, before the clause cut is applied. Wide enough for
# "sans jamais avoir a utiliser --force".
DENY_WINDOW = 60


def bare_uses(text, term, window=DENY_WINDOW):
    """Return the mentions of `term` that are NOT inside a negation.

    A correct answer often has to *name* what it rules out ("rebaser, sans
    utiliser --force"). Forbidding the string outright punishes it for that,
    and a fixed-width lookbehind only sees the characters glued to the term:
    `(?<!sans )` matches "sans --force" but not "sans utiliser --force", which
    is how this check failed a right answer. Look back over the current clause
    instead -- the property is "does not prescribe it", not "does not contain
    one of these exact phrasings". Same lesson as the rest of this file:
    describe the property, never enumerate the wordings.
    """
    bare = []
    for match in re.finditer(re.escape(normalize(str(term))), text, re.IGNORECASE):
        before = text[max(0, match.start() - window):match.start()]
        cuts = list(CLAUSE_BREAK.finditer(before))
        if cuts:
            before = before[cuts[-1].end():]
        if not any(neg in before.lower() for neg in NEGATIONS):
            bare.append(text[max(0, match.start() - 40):match.end()])
    return bare


def run_check(text, check):
    """Evaluate one check. Returns (passed: bool, explanation: str)."""
    kind = check.get("type")
    text = normalize(text)
    haystack = text if check.get("case_sensitive") else text.lower()

    def needle(value):
        value = normalize(str(value))
        return value if check.get("case_sensitive") else value.lower()

    if kind == "contains":
        want = check["value"]
        ok = needle(want) in haystack
        return ok, "contains %r" % want

    if kind == "not_contains":
        want = check["value"]
        ok = needle(want) not in haystack
        return ok, "must not contain %r" % want

    if kind == "any_of":
        wants = check["values"]
        hit = [w for w in wants if needle(w) in haystack]
        return bool(hit), "any of %r (matched %r)" % (wants, hit)

    if kind == "all_of":
        wants = check["values"]
        missing = [w for w in wants if needle(w) not in haystack]
        return not missing, "all of %r (missing %r)" % (wants, missing)

    if kind == "regex":
        flags = 0 if check.get("case_sensitive") else re.IGNORECASE
        ok = re.search(check["value"], text, flags) is not None
        return ok, "regex %r" % check["value"]

    if kind == "not_regex":
        # For forbidding a *prescription* rather than a mention: a correct
        # answer often has to name the thing it rules out ("push, sans
        # --force"), and a plain not_contains would punish it for that.
        flags = 0 if check.get("case_sensitive") else re.IGNORECASE
        hit = re.search(check["value"], text, flags)
        return hit is None, "must not match %r%s" % (
            check["value"], "" if hit is None else " (matched %r)" % hit.group(0))

    if kind == "max_sentences":
        # "Explique en trois phrases" is a checkable instruction, and leaving
        # it unchecked lets a ten-sentence answer score full marks. Found by
        # the symmetric audit, not by any of the frontier passes.
        count = len(SENTENCE_END.findall(text.strip()))
        want = check["value"]
        return count <= want, "at most %d sentence(s) (got %d)" % (want, count)

    if kind == "not_prescribed":
        bare = bare_uses(text, check["value"], check.get("window", DENY_WINDOW))
        return not bare, "%r is never prescribed%s" % (
            check["value"], "" if not bare else " (bare use: %r)" % bare[0])

    if kind == "valid_json":
        try:
            first_json(text)
            return True, "contains a parsable JSON value"
        except (json.JSONDecodeError, ValueError) as exc:
            return False, "JSON parse failed: %s" % exc

    if kind == "json_field":
        try:
            obj = first_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return False, "JSON parse failed: %s" % exc
        found, value = dig(obj, check["path"])
        if not found:
            return False, "missing field %s" % check["path"]
        if "value" in check:
            ok = str(value).strip().lower() == str(check["value"]).strip().lower()
            return ok, "%s == %r (got %r)" % (check["path"], check["value"], value)
        return True, "%s present" % check["path"]

    if kind == "max_words":
        n = len(text.split())
        return n <= check["value"], "at most %d words (got %d)" % (check["value"], n)

    if kind == "max_chars":
        n = len(text.strip())
        return n <= check["value"], "at most %d chars (got %d)" % (check["value"], n)

    if kind == "starts_with":
        ok = text.strip().lower().startswith(str(check["value"]).lower())
        return ok, "starts with %r" % check["value"]

    return False, "unknown check type %r" % kind


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def diagnose(error, args):
    """Turn a transport error into the one sentence that actually helps."""
    hints = []
    if "HTTP 401" in error or "HTTP 403" in error:
        var = ("$env:%s" % args.api_key_env) if args.api_key_env else "(no --api-key-env given)"
        hints.append("authentication refused -- %s does not hold a valid key for %s"
                     % (var, args.url))
    if "HTTP 402" in error:
        hints.append("the account is out of credit")
    if "HTTP 404" in error or "model" in error.lower():
        hints.append("model id %r may not exist on this provider" % args.model)
    if "transport:" in error:
        hints.append("no answer from %s -- endpoint down, or wrong host" % args.url)
    if not hints:
        hints.append("check the url, the model id, and the key")
    return "hint: " + "\n      ".join(hints)


def load_cases(path, category=None, only_id=None):
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    if category:
        cases = [c for c in cases if c.get("category") == category]
    if only_id:
        cases = [c for c in cases if c["id"] == only_id]
    return cases


def build_messages(case):
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


def cmd_run(args):
    cases = load_cases(args.cases, args.category, args.id)
    if not cases:
        print("no case selected", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)

    # The key is read from the environment, never passed on the command line:
    # a shell history is not a place for a token.
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        print("env var %s is empty -- no Authorization header will be sent"
              % args.api_key_env, file=sys.stderr)

    if api_key and not args.url.startswith("https://"):
        print("refusing to send a bearer token over a non-HTTPS URL: %s" % args.url,
              file=sys.stderr)
        return 1

    # A pasted command whose placeholders were never replaced is a real failure
    # mode, not a hypothetical one. Catch it before burning 32 requests.
    for label, value in (("--model", args.model), ("--tag", args.tag),
                         ("the API key", api_key or "")):
        if "<" in value or ">" in value:
            shown = value if label != "the API key" else "<redacted>"
            print("%s still holds a placeholder: %s\n"
                  "replace it with a real value before running."
                  % (label, shown), file=sys.stderr)
            return 1

    if api_key:
        print("NOTE: prompts are leaving this machine towards %s" % args.url)

    # One cheap probe before the run: credentials and model id are wrong far
    # more often than the cases are, and finding out on request 32 is useless.
    print("preflight ... ", end="", flush=True)
    probe = call_model(args.url, args.model,
                       [{"role": "user", "content": "Reply with the single word: ready"}],
                       16, 0.0, None, min(args.timeout, 60.0), api_key)
    if "error" in probe:
        print("FAILED", flush=True)
        print("\n%s\n%s" % (probe["error"], diagnose(probe["error"], args)),
              file=sys.stderr)
        return 2
    print("ok")

    if args.warmup:
        print("warmup ...", flush=True)
        call_model(args.url, args.model,
                   [{"role": "user", "content": "Reply with the single word: ready"}],
                   16, 0.0, args.seed, args.timeout, api_key)

    records = []
    transport_failures = 0
    consecutive = 0
    empty_answers = 0

    for index, case in enumerate(cases, start=1):
        label = "%-22s %-14s" % (case["id"], case.get("category", "-"))
        print("[%2d/%2d] %s " % (index, len(cases), label), end="", flush=True)

        result = call_model(
            args.url, args.model, build_messages(case),
            case.get("max_tokens", args.max_tokens),
            case.get("temperature", args.temperature),
            args.seed, args.timeout, api_key,
        )

        if "error" in result:
            transport_failures += 1
            consecutive += 1
            print("ERROR  %s" % result["error"][:70])
            records.append({
                "id": case["id"], "category": case.get("category"),
                "error": result["error"], "passed": False, "score": 0.0,
            })
            if consecutive >= args.max_consecutive_errors:
                print("\naborting after %d consecutive transport errors\n%s"
                      % (consecutive, diagnose(result["error"], args)),
                      file=sys.stderr)
                break
            continue
        consecutive = 0

        # A reasoning model can spend the entire budget thinking and emit
        # nothing. That is a budget problem, not an answer, so give it one
        # bigger envelope before recording a miss.
        retried = False
        wasted = 0
        # Two ways the budget can masquerade as a quality signal: nothing came
        # back at all, or the answer stops mid-sentence because it ran out of
        # room. Both are budget problems and both used to be scored as wrong
        # answers -- the second one silently, which is worse.
        if not result["text"].strip() or result.get("truncated"):
            why = "empty" if not result["text"].strip() else "truncated"
            budget = case.get("max_tokens", args.max_tokens) * 3
            print("%s, retrying at %d tokens ... " % (why, budget),
                  end="", flush=True)
            # Whatever happens next, the first attempt is spent. Count it here:
            # if the retry also comes back empty, `result` is still the first
            # attempt and its token count alone would hide the retry entirely.
            wasted = result["output_tokens"] or 0
            bigger = call_model(
                args.url, args.model, build_messages(case), budget,
                case.get("temperature", args.temperature),
                args.seed, args.timeout, api_key,
            )
            if "error" not in bigger:
                wasted += bigger["output_tokens"] or 0
            if "error" not in bigger and bigger["text"].strip():
                result = bigger
                retried = True
                # The retry produced the answer, so its tokens are not waste --
                # only the first attempt was burned for nothing.
                wasted -= bigger["output_tokens"] or 0

        # An empty completion is not a wrong answer, it is a missing one -- and
        # it must never score. `max_words: 1` is satisfied by zero words, so
        # without this an empty answer would pass a brevity case outright.
        if not result["text"].strip():
            empty_answers += 1
            print("EMPTY  %s tokens spent, no visible content" % result["output_tokens"])
            records.append({
                "id": case["id"], "category": case.get("category"),
                "passed": False, "score": 0.0, "empty": True,
                "output_tokens": result["output_tokens"],
                "tokens_burned": wasted or result["output_tokens"] or 0,
                "retry_attempted": wasted > (result["output_tokens"] or 0),
                "ttft_s": result["ttft_s"], "total_s": result["total_s"],
                "answer": "",
                "note": "empty completion -- a reasoning model probably spent the "
                        "whole max_tokens budget on hidden reasoning",
            })
            continue

        checks = []
        for check in case.get("checks", []):
            ok, why = run_check(result["text"], check)
            checks.append({"type": check.get("type"), "ok": ok, "detail": why})

        n_ok = sum(1 for c in checks if c["ok"])
        score = n_ok / len(checks) if checks else None
        passed = bool(checks) and n_ok == len(checks)

        records.append({
            "id": case["id"],
            "category": case.get("category"),
            "passed": passed,
            "score": score,
            "checks": checks,
            "ttft_s": result["ttft_s"],
            "total_s": result["total_s"],
            "output_tokens": result["output_tokens"],
            "tokens_per_s": result["tokens_per_s"],
            "retried_with_bigger_budget": retried,
            "tokens_wasted_before_retry": wasted if retried else 0,
            # Still truncated after the bigger budget: the score below counts
            # against an answer the model was not allowed to finish. Recorded
            # so that never has to be rediscovered by reading the raw text.
            "truncated": bool(result.get("truncated")),
            "answer": result["text"],
        })

        mark = "PASS" if passed else ("part" if n_ok else "FAIL")
        print("%-4s %s/%s  %5.1f tok/s  ttft %s" % (
            mark, n_ok, len(checks),
            result["tokens_per_s"] or 0.0,
            ("%.2fs" % result["ttft_s"]) if result["ttft_s"] else "n/a"))

    # The report is written even when everything failed: a run with no artifact
    # leaves nothing to examine afterwards, which is exactly when you need it.
    report = {
        "meta": {
            "tag": args.tag,
            "model": args.model,
            "url": args.url,
            "cases_file": os.path.basename(args.cases),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "temperature": args.temperature,
            "seed": args.seed,
            "n_cases": len(cases),
            "remote": bool(api_key),
        },
        "cases": records,
        "summary": summarize(records),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, "%s-%s.json" % (args.tag, stamp))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print_summary(report["summary"], args.tag)
    print("\nwritten: %s" % path)

    if empty_answers:
        print("\n%d case(s) returned an empty completion -- raise --max-tokens. "
              "A reasoning model can spend the whole budget before writing a "
              "single visible character." % empty_answers, file=sys.stderr)

    if transport_failures:
        print("\n%d of %d case(s) never reached the model -- the scores above "
              "count them as failures and are NOT a quality verdict."
              % (transport_failures, len(records)), file=sys.stderr)
        return 2
    return 0


def summarize(records):
    by_category = {}
    for rec in records:
        cat = rec.get("category") or "-"
        slot = by_category.setdefault(cat, {"n": 0, "passed": 0, "scores": [],
                                            "tps": [], "ttft": [],
                                            "silent": 0, "burned": 0,
                                            "burned_partial": False,
                                            "rescued": 0})
        slot["n"] += 1
        slot["passed"] += 1 if rec.get("passed") else 0
        if rec.get("score") is not None:
            slot["scores"].append(rec["score"])
        if rec.get("tokens_per_s"):
            slot["tps"].append(rec["tokens_per_s"])
        if rec.get("ttft_s"):
            slot["ttft"].append(rec["ttft_s"])
        # Silence accounting. Across three models this turned out to be the
        # only thing separating them -- not one wrong answer in 96 pairs, but
        # four cases where nothing came back. A property that decides the
        # verdict has no business being read off the console by hand.
        if rec.get("empty"):
            slot["silent"] += 1
            # `tokens_burned` covers both attempts when a retry also came back
            # empty. Result files written before that field existed only have
            # the first attempt, so the total is a LOWER BOUND for them -- and
            # says so, rather than passing an undercount off as a measurement.
            if rec.get("tokens_burned") is None:
                slot["burned_partial"] = True
            slot["burned"] += (rec.get("tokens_burned")
                               or rec.get("output_tokens") or 0)
        if rec.get("retried_with_bigger_budget"):
            slot["rescued"] += 1
            slot["burned"] += rec.get("tokens_wasted_before_retry") or 0

    out = {}
    for cat, slot in sorted(by_category.items()):
        out[cat] = {
            "n": slot["n"],
            "passed": slot["passed"],
            "pass_rate": round(slot["passed"] / slot["n"], 3),
            "mean_score": round(statistics.fmean(slot["scores"]), 3) if slot["scores"] else None,
            "median_tokens_per_s": round(statistics.median(slot["tps"]), 1) if slot["tps"] else None,
            "median_ttft_s": round(statistics.median(slot["ttft"]), 2) if slot["ttft"] else None,
            "silent": slot["silent"],
            "silence_rate": round(slot["silent"] / slot["n"], 3),
            "tokens_burned_silent": slot["burned"],
            "tokens_burned_is_lower_bound": slot["burned_partial"],
            "rescued": slot["rescued"],
        }

    total_n = sum(s["n"] for s in out.values())
    total_pass = sum(s["passed"] for s in out.values())
    all_tps = [r["tokens_per_s"] for r in records if r.get("tokens_per_s")]
    all_scores = [r["score"] for r in records if r.get("score") is not None]
    total_silent = sum(s["silent"] for s in out.values())
    total_burned = sum(s["tokens_burned_silent"] for s in out.values())
    total_rescued = sum(s["rescued"] for s in out.values())
    out["ALL"] = {
        "n": total_n,
        "passed": total_pass,
        "pass_rate": round(total_pass / total_n, 3) if total_n else None,
        "mean_score": round(statistics.fmean(all_scores), 3) if all_scores else None,
        "median_tokens_per_s": round(statistics.median(all_tps), 1) if all_tps else None,
        "median_ttft_s": None,
        "silent": total_silent,
        "silence_rate": round(total_silent / total_n, 3) if total_n else None,
        "tokens_burned_silent": total_burned,
        # Aggregate the caveat too. Computing it per category and forgetting it
        # here meant the warning existed in the data and never reached a human.
        "tokens_burned_is_lower_bound": any(
            s.get("tokens_burned_is_lower_bound") for s in out.values()),
        "rescued": total_rescued,
    }
    return out


def print_summary(summary, tag):
    print("\n%s" % ("=" * 74))
    print("%s" % tag)
    print("%-18s %4s %7s %9s %11s %9s" %
          ("category", "n", "passed", "pass rate", "mean score", "tok/s"))
    print("-" * 74)
    for cat, slot in summary.items():
        if cat == "ALL":
            continue
        print("%-18s %4d %7d %9.0f%% %11s %9s" % (
            cat, slot["n"], slot["passed"], slot["pass_rate"] * 100,
            slot["mean_score"] if slot["mean_score"] is not None else "-",
            slot["median_tokens_per_s"] if slot["median_tokens_per_s"] else "-"))
    print("-" * 74)
    all_slot = summary["ALL"]
    print("%-18s %4d %7d %9.0f%% %11s %9s" % (
        "ALL", all_slot["n"], all_slot["passed"], all_slot["pass_rate"] * 100,
        all_slot["mean_score"], all_slot["median_tokens_per_s"]))

    # Silence is reported separately because it is not a wrong answer and must
    # not be read as one. On this bench it was the only thing that separated
    # three models: zero wrong answers in 96 pairs, four silences.
    silent = all_slot.get("silent") or 0
    rescued = all_slot.get("rescued") or 0
    burned = all_slot.get("tokens_burned_silent") or 0
    if silent or rescued:
        print("\n%-18s %s" % ("silence", "(a spent budget with nothing to show)"))
        print("-" * 74)
        print("%-18s %4d case(s), %.0f%% of the bench"
              % ("never answered", silent, (all_slot.get("silence_rate") or 0) * 100))
        print("%-18s %4d case(s) answered only after a 3x budget"
              % ("rescued", rescued))
        # A result file written before `tokens_burned` existed only recorded
        # the first attempt, so the retry that also came back empty is missing
        # from the total. Say "at least" rather than pass an undercount off as
        # a measurement -- the whole point of this bench is not doing that.
        partial = all_slot.get("tokens_burned_is_lower_bound")
        print("%-18s %s%d token(s) produced, none of it visible%s"
              % ("burned", "at least " if partial else "", burned,
                 "" if not partial else
                 "\n%-18s (lower bound: this file predates per-retry accounting)"
                 % ""))
        for cat, slot in summary.items():
            if cat != "ALL" and (slot.get("silent") or slot.get("rescued")):
                print("%-18s   %s: %d silent, %d rescued"
                      % ("", cat, slot.get("silent") or 0, slot.get("rescued") or 0))


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------

def cmd_rescore(args):
    """Re-apply the current checks to answers already on disk.

    Checks get fixed; the model's answers do not change. Re-running a paid
    endpoint to measure the same text again would be waste.
    """
    path = resolve(args.file)
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    cases = {c["id"]: c for c in load_cases(args.cases)}
    changed, skipped = [], []

    for rec in report["cases"]:
        case = cases.get(rec["id"])
        answer = (rec.get("answer") or "").strip()
        if case is None or not answer:
            skipped.append(rec["id"])
            continue
        checks = []
        for chk in case["checks"]:
            ok, why = run_check(rec["answer"], chk)
            checks.append({"type": chk.get("type"), "ok": ok, "detail": why})
        n_ok = sum(1 for c in checks if c["ok"])
        before = rec.get("passed")
        rec["checks"] = checks
        rec["score"] = n_ok / len(checks) if checks else None
        rec["passed"] = bool(checks) and n_ok == len(checks)
        if rec["passed"] != before:
            changed.append((rec["id"], before, rec["passed"]))

    report["summary"] = summarize(report["cases"])
    report["meta"]["rescored_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for case_id, before, after in changed:
        print("%-28s %s -> %s" % (case_id, before, after))
    if skipped:
        print("skipped (no stored answer): %s" % ", ".join(skipped))
    print_summary(report["summary"], report["meta"]["tag"] + " (rescored)")
    print("\nrewritten: %s" % path)
    return 0


def cmd_compare(args):
    """Put N result files side by side, category by category, then case by case.

    `diff` answers "what does B lose against A". With a third model in the
    picture -- a frontier ceiling, the big local model, the small challenger --
    the useful view is the matrix, and above all the short list of cases where
    the models actually disagree. Cases they all pass carry no information.
    """
    reports = []
    for pattern in args.files:
        path = resolve(pattern)
        with open(path, encoding="utf-8") as handle:
            reports.append((json.load(handle), os.path.basename(path)))

    tags = [rep["meta"]["tag"] for rep, _ in reports]
    width = max(14, max(len(t) for t in tags) + 2)
    for index, (rep, name) in enumerate(reports):
        print("%s = %-22s (%s)" % ("ABCDEFGH"[index], rep["meta"]["tag"], name))

    categories = sorted({c for rep, _ in reports for c in rep["summary"]} - {"ALL"})
    categories.append("ALL")

    print("\n%-18s%s" % ("category", "".join(t.rjust(width) for t in tags)))
    print("-" * (18 + width * len(tags)))
    for cat in categories:
        cells = []
        for rep, _ in reports:
            slot = rep["summary"].get(cat)
            cells.append(("%.0f%%" % (slot["pass_rate"] * 100)) if slot else "-")
        print("%-18s%s" % (cat, "".join(c.rjust(width) for c in cells)))

    print("\n%-18s%s" % ("median tok/s", "".join(
        str(rep["summary"]["ALL"]["median_tokens_per_s"]).rjust(width)
        for rep, _ in reports)))

    indexes = [{c["id"]: c for c in rep["cases"]} for rep, _ in reports]
    all_ids = [c["id"] for c in reports[0][0]["cases"]]
    disagreed = []
    for case_id in all_ids:
        verdicts = [idx.get(case_id, {}).get("passed") for idx in indexes]
        if len(set(verdicts)) > 1:
            disagreed.append((case_id, verdicts))

    # Case ids run past 18 characters ("lh-07-checklist-publication"), and a
    # truncated id is useless: it is the handle you feed back to `diff --show`.
    id_width = max(18, max(len(c) for c in all_ids) + 2)
    print("\ncases where the models disagree (%d of %d)"
          % (len(disagreed), len(all_ids)))
    print("-" * (id_width + width * len(tags)))
    if not disagreed:
        print("none -- every case is passed, or failed, by all of them alike")
    # Show WHY a case failed, not just that it did. A bare FAIL invites the
    # reader to supply the reason, and the plausible reason is usually "it got
    # it wrong" -- which on this bench is almost never what happened. Reported
    # as a wrong answer once here, from this very table, before anyone checked
    # the record that said `empty: true` all along.
    def verdict_of(index, case_id):
        record = indexes[index].get(case_id)
        if record is None:
            return "-"
        if record.get("passed"):
            return "PASS"
        if record.get("empty"):
            return "SILENT"
        if record.get("truncated"):
            return "CUT"
        return "WRONG"

    for case_id, _ in disagreed:
        cells = [verdict_of(i, case_id) for i in range(len(indexes))]
        print("%-*s%s" % (id_width, case_id, "".join(c.rjust(width) for c in cells)))
    if disagreed:
        print("\nPASS / WRONG (answered, incorrect) / SILENT (no visible content) "
              "/ CUT (hit the budget)")
    return 0


def resolve(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit("no result file matches %r" % pattern)
    return matches[-1]


def cmd_diff(args):
    path_a, path_b = resolve(args.a), resolve(args.b)
    with open(path_a, encoding="utf-8") as handle:
        report_a = json.load(handle)
    with open(path_b, encoding="utf-8") as handle:
        report_b = json.load(handle)

    tag_a = report_a["meta"]["tag"]
    tag_b = report_b["meta"]["tag"]
    print("A = %s   (%s)" % (tag_a, os.path.basename(path_a)))
    print("B = %s   (%s)" % (tag_b, os.path.basename(path_b)))

    shared = (set(report_a["summary"]) | set(report_b["summary"])) - {"ALL"}
    categories = sorted(shared) + ["ALL"]

    print("\n%-18s %14s %14s %10s" % ("category", "A pass", "B pass", "delta"))
    print("-" * 60)
    for cat in categories:
        sa = report_a["summary"].get(cat)
        sb = report_b["summary"].get(cat)
        if not sa or not sb:
            continue
        delta = (sb["pass_rate"] - sa["pass_rate"]) * 100
        print("%-18s %13.0f%% %13.0f%% %+9.0f pt" % (
            cat, sa["pass_rate"] * 100, sb["pass_rate"] * 100, delta))

    print("\n%-18s %14s %14s" % ("speed", "A", "B"))
    print("-" * 60)
    print("%-18s %13s  %13s" % (
        "median tok/s",
        report_a["summary"]["ALL"]["median_tokens_per_s"],
        report_b["summary"]["ALL"]["median_tokens_per_s"]))

    index_a = {c["id"]: c for c in report_a["cases"]}
    index_b = {c["id"]: c for c in report_b["cases"]}

    regressions, gains = [], []
    for case_id, rec_a in index_a.items():
        rec_b = index_b.get(case_id)
        if not rec_b:
            continue
        if rec_a.get("passed") and not rec_b.get("passed"):
            regressions.append((case_id, rec_a.get("category")))
        elif not rec_a.get("passed") and rec_b.get("passed"):
            gains.append((case_id, rec_a.get("category")))

    print("\nB loses %d case(s) A had, B gains %d." % (len(regressions), len(gains)))
    for case_id, cat in regressions:
        print("  LOST  %-22s %s" % (case_id, cat))
    for case_id, cat in gains:
        print("  WON   %-22s %s" % (case_id, cat))

    if args.show:
        print("\n--- answers for %s ---" % args.show)
        for tag, index in ((tag_a, index_a), (tag_b, index_b)):
            rec = index.get(args.show)
            print("\n[%s] passed=%s" % (tag, rec and rec.get("passed")))
            print((rec or {}).get("answer", "<absent>"))
    return 0


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the bench against one endpoint")
    run.add_argument("--url", default="http://100.x.y.z:8000/v1")
    run.add_argument("--model", default="<local-model-alias>")
    run.add_argument("--tag", required=True,
                     help="short name for this configuration, used in the filename")
    run.add_argument("--cases", default=DEFAULT_CASES)
    run.add_argument("--out", default=DEFAULT_OUT)
    run.add_argument("--category", help="run only this category")
    run.add_argument("--id", help="run only this case id")
    run.add_argument("--max-tokens", type=int, default=2500,
                     help="deliberately generous: a reasoning model spends part "
                          "of this budget before emitting anything visible. "
                          "Length limits belong in the checks, not here.")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--max-consecutive-errors", type=int, default=3,
                     help="give up after this many transport errors in a row "
                          "(default 3) -- 32 identical 401s help nobody")
    run.add_argument("--api-key-env", metavar="VAR",
                     help="name of the environment variable holding the bearer "
                          "token, for a remote frontier endpoint. Requires https. "
                          "The token is never accepted on the command line.")
    run.add_argument("--warmup", action="store_true",
                     help="send one throwaway request first (model just loaded)")
    run.set_defaults(func=cmd_run)

    rescore = sub.add_parser(
        "rescore", help="re-apply current checks to a stored result file")
    rescore.add_argument("file", help="path or glob of the result file to rescore")
    rescore.add_argument("--cases", default=DEFAULT_CASES)
    rescore.set_defaults(func=cmd_rescore)

    diff = sub.add_parser("diff", help="compare two result files")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.add_argument("--show", help="print both answers for this case id")
    diff.set_defaults(func=cmd_diff)

    compare = sub.add_parser(
        "compare", help="put two or more result files side by side")
    compare.add_argument("files", nargs="+",
                         help="paths or globs, in the order to display them")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
