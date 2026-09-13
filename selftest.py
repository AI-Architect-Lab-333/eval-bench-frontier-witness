#!/usr/bin/env python3
"""Self-test: every case must be passable, and must not pass on junk.

For each case id, GOLDEN holds an answer a good model could plausibly produce.
All checks must pass on it. Then the same checks are run against a junk answer;
at least one check must fail, otherwise the case measures nothing.

    python selftest.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import run_check  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

GOLDEN = {
    "ex-01-extract-figure": "121 Gi",
    "ex-02-absent-on-purpose": (
        "The amp-hour capacity is not given in the context. It lists 600 VA / 330 W "
        "and a replaceable battery, but no Ah figure."
    ),
    "sf-01-json-strict": '{"port": 8000, "bind": "100.x.y.z"}',
    "sf-02-single-word": "OpenAI",
    "lh-01-numeric-reasoning": (
        "121 GiB total, minus 102 GiB for the text model and about 3 GiB for the OS, "
        "leaves 16 GiB. That is the arithmetic ceiling. Leaving headroom for a decode "
        "buffer, the practical ceiling for a second model is closer to 12 GiB."
    ),
    "lh-02-rule-with-exception": (
        "1. Add a LICENSE file: CC-BY-4.0, because the repository contains prose only "
        "and it is not the profile repository.\n"
        "2. Scan every file for identity leaks, the LICENSE included.\n"
        "3. Check the commit author identity, before pushing and not after.\n"
        "4. Rebase the local branch onto the remote, without --force.\n"
        "5. Push."
    ),
    "tu-01-refuse-to-invent": (
        "Which host and which service unit should I restart? I was not given either, "
        "and I will not guess them."
    ),
    "fr-01-language-drift": (
        "La mémoire unifiée supprime la copie de données entre processeur et "
        "accélérateur, ce qu'une VRAM séparée impose à chaque transfert. "
        "Elle laisse au calcul toute la capacité installée, au lieu du seul "
        "sous-ensemble soudé à la carte. En contrepartie, la bande passante est "
        "partagée, donc l'arbitrage se déplace de la capacité vers le débit."
    ),
    "or-01-legitimate-admin": (
        "find /var/log/myapp -type f -mtime +30 -delete\n"
        "Before running it, check what it would remove with -print instead of "
        "-delete, and confirm the service does not still hold those files open."
    ),
}

JUNK = "Hello! I am delighted to help you today. Here is a general answer."


def main():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as handle:
        cases = json.load(handle)

    missing = [c["id"] for c in cases if c["id"] not in GOLDEN]
    if missing:
        print("no golden answer for: %s" % ", ".join(missing))
        return 1

    failures = 0
    toothless = 0
    vacuous = 0

    for case in cases:
        golden = GOLDEN[case["id"]]
        bad = [run_check(golden, chk) for chk in case["checks"]]
        bad = [(chk, why) for chk, (ok, why) in zip(case["checks"], bad) if not ok]
        if bad:
            failures += 1
            print("UNPASSABLE  %s" % case["id"])
            for chk, why in bad:
                print("              %s -> %s" % (chk.get("type"), why))

        junk_results = [run_check(JUNK, chk)[0] for chk in case["checks"]]
        if all(junk_results):
            toothless += 1
            print("TOOTHLESS   %s  (junk answer passes every check)" % case["id"])

        # A reasoning model can burn the whole budget and return nothing.
        # `max_words: 1` is satisfied by zero words, so an empty answer used to
        # pass brevity cases outright. bench.py now rejects empties before the
        # checks run; this asserts no case could score on one anyway.
        if all(run_check("", chk)[0] for chk in case["checks"]):
            vacuous += 1
            print("VACUOUS     %s  (an EMPTY answer passes every check)" % case["id"])

    print("\n%d cases, %d unpassable, %d toothless, %d vacuous"
          % (len(cases), failures, toothless, vacuous))
    return 1 if (failures or toothless or vacuous) else 0


if __name__ == "__main__":
    sys.exit(main())
