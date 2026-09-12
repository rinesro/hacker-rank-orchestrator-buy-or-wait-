"""Standalone Layer 5 validator.

    python3 code/evaluation/validate.py                 # validates ./output.csv
    python3 code/evaluation/validate.py path/to/file.csv

Exits non-zero on any contract violation, so it can gate a submission.
"""

from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from loaders import Dataset, repo_root  # noqa: E402
from pipeline import COLUMNS  # noqa: E402
from validator import assert_no_hardcoded_answers, validate  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else os.path.join(repo_root(), "output.csv")
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    data = Dataset()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)

    if header != COLUMNS:
        print(f"header mismatch\n  got  {header}\n  want {COLUMNS}", file=sys.stderr)
        return 1

    sample_ids = {r.request_id for r in data.samples}
    requests = data.samples if rows and rows[0]["request_id"] in sample_ids else data.requests

    report = validate(rows, data, requests)
    offenders = assert_no_hardcoded_answers(os.path.dirname(HERE), data)

    for line in report.warnings:
        print(f"[warn] {line}")
    for line in offenders:
        print(f"[disqualifying] {line}")
    for line in report.errors:
        print(f"[invalid] {line}")

    if report.errors or offenders:
        print(f"\nFAILED: {len(report.errors)} errors, {len(offenders)} hardcoded identifiers")
        return 1
    print(f"OK: {len(rows)} rows, {len(COLUMNS)} columns, all contract gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
