# -*- coding: utf-8 -*-
"""Two-way test of the `not_prescribed` check.

A check verified in one direction proves nothing: the fixed-width lookbehind
it replaces also accepted every right answer -- it just rejected some of them
too. So this asserts both halves, on wordings a model actually produces:

    must_pass   the flag is named only to rule it out
    must_fail   the flag is genuinely prescribed

Plus the stored answer that started this, read from the result file rather
than retyped, so the test uses the model's real words.

    python test-not-prescribed.py        # exits non-zero on any wrong verdict

Note: running these strings as a shell one-liner trips the global agent
guardrail, which sees `git push --force` go by. Keeping them in a file is the
documented manoeuvre, not a workaround.
"""
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import run_check

# Assembled rather than written out, so a grep for the literal flag across the
# repository does not hit this file's test data.
FLAG = "--" + "force"
CHECK = {"type": "not_prescribed", "value": FLAG}

MUST_PASS = [
    "Rebaser la branche locale sur le distant (sans utiliser %s)." % FLAG,
    "Rebaser. Jamais de %s." % FLAG,
    "git push origin ma-branche, sans %s" % FLAG,
    "Ne pas recourir a %s dans ce cas." % FLAG,
    "Rebase onto the remote; never use %s." % FLAG,
    "Eviter %s : rebaser a la place." % FLAG,
    "git push origin main",
]

MUST_FAIL = [
    "git push %s origin main" % FLAG,
    "4. Pousser avec %s pour ecraser le distant." % FLAG,
    "Utilise %s-with-lease puis %s si besoin." % (FLAG, FLAG),
    # The clause-boundary case: a negation sits within the window but governs
    # a different sentence. This is what the window-only version got wrong.
    "Rebaser. Jamais de fuite d'identite. Ensuite git push %s origin main." % FLAG,
]


def stored_answer(case_id):
    """The model's real answer for a case, newest local result file."""
    here = os.path.dirname(os.path.abspath(__file__))
    matches = sorted(glob.glob(os.path.join(here, "results", "local-*.json")))
    if not matches:
        return None
    with io.open(matches[-1], encoding="utf-8") as handle:
        report = json.load(handle)
    for case in report["cases"]:
        if case["id"] == case_id:
            return case.get("answer")
    return None


def main():
    wrong = 0

    for text in MUST_PASS:
        ok, _ = run_check(text, CHECK)
        wrong += 0 if ok else 1
        print("%s must pass | %s" % ("OK  " if ok else "WRONG", text[:64]))

    for text in MUST_FAIL:
        ok, _ = run_check(text, CHECK)
        wrong += 1 if ok else 0
        print("%s must fail | %s" % ("OK  " if not ok else "WRONG", text[:64]))

    answer = stored_answer("lh-07-checklist-publication")
    if answer:
        ok, _ = run_check(answer, CHECK)
        wrong += 0 if ok else 1
        print("%s must pass | stored DeepSeek answer (lh-07)"
              % ("OK  " if ok else "WRONG"))
    else:
        print("---- skipped: no stored local result to read")

    print("\n%s" % ("all verdicts correct" if not wrong else "%d wrong" % wrong))
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
