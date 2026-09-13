# -*- coding: utf-8 -*-
"""Symmetric audit: look for checks that are too LENIENT.

`selftest.py` asks whether a case can be passed at all, and whether junk or an
empty answer slips through. It never asks the opposite question -- whether a
case's checks would accept an answer that is simply *wrong*. That blind spot
matters: checks only ever get inspected where the measured model failed, so
every correction runs in the score-raising direction. A check that is wrongly
generous stays invisible and inflates the result.

Cross-validation is the objective half of the audit. Every stored answer is
submitted to every *other* case's checks. A case whose checks accept an
unrelated answer is not measuring what it claims to.

    python audit-leniency.py results/local-a-*.json

Reports each foreign answer that passes, so the case can be judged by hand.
Exits non-zero if any case is accepted by a foreign answer.
"""
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import run_check, load_cases, DEFAULT_CASES


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "results/local-*.json"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit("no result file matches %r" % pattern)
    path = matches[-1]
    with io.open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    print("auditing %s\n" % os.path.basename(path))

    cases = {c["id"]: c for c in load_cases(DEFAULT_CASES)}
    answers = {c["id"]: (c.get("answer") or "")
               for c in report["cases"] if (c.get("answer") or "").strip()}

    porous = {}
    for case_id, case in cases.items():
        checks = case.get("checks", [])
        if not checks:
            continue
        for other_id, answer in answers.items():
            if other_id == case_id:
                continue
            if all(run_check(answer, chk)[0] for chk in checks):
                porous.setdefault(case_id, []).append(other_id)

    for case_id in sorted(porous):
        others = porous[case_id]
        print("POREUX  %-28s accepte %d reponse(s) etrangere(s)"
              % (case_id, len(others)))
        for other_id in others[:6]:
            print("            <- %s" % other_id)
        if len(others) > 6:
            print("            <- ... et %d autres" % (len(others) - 6))

    tested = len([c for c in cases.values() if c.get("checks")])
    print("\n%d cas testes, %d poreux, %d etanches"
          % (tested, len(porous), tested - len(porous)))
    return 1 if porous else 0


if __name__ == "__main__":
    sys.exit(main())
