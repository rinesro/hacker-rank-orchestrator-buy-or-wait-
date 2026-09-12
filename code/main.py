"""Buy or Wait? - entry point.

    python3 code/main.py            # writes ./output.csv for dataset/requests.csv
    python3 code/main.py --samples  # writes ./predictions_samples.csv instead

Deterministic by construction: no wall-clock input, no randomness, requests
processed in file order, and AI evidence read from an on-disk cache.  Two runs
produce byte-identical output.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT, ForecastConfig  # noqa: E402
from evidence import stub_evidence  # noqa: E402
from loaders import Dataset, repo_root  # noqa: E402
from pipeline import COLUMNS, run  # noqa: E402
from validator import assert_no_hardcoded_answers, validate  # noqa: E402


def load_evidence(use_ai: bool):
    """Layer 1 evidence, or the zero-cost stub.

    The real extractors land behind this call; until then the deterministic
    pipeline runs on stubbed evidence and the flag is a no-op with a notice.
    """
    if not use_ai:
        return stub_evidence()
    try:
        from evidence_ai import load_cached_evidence  # type: ignore
    except ImportError:
        print("[evidence] Layer 1 not wired yet; running on stubbed evidence", file=sys.stderr)
        return stub_evidence()
    return load_cached_evidence()


def write_csv(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? decision agent")
    parser.add_argument("--samples", action="store_true", help="run the 25 solved samples")
    parser.add_argument("--out", default=None, help="output path (default ./output.csv)")
    parser.add_argument("--no-ai", action="store_true", help="force stubbed evidence")
    args = parser.parse_args(argv)

    root = repo_root()
    here = os.path.dirname(os.path.abspath(__file__))
    data = Dataset()
    cfg: ForecastConfig = DEFAULT

    offenders = assert_no_hardcoded_answers(here, data)
    if offenders:
        for line in offenders:
            print(f"[disqualifying] {line}", file=sys.stderr)
        return 2

    requests = data.samples if args.samples else data.requests
    evidence = load_evidence(use_ai=not args.no_ai)
    rows = run(data, requests, evidence, cfg)

    report = validate(rows, data, requests)
    for line in report.warnings:
        print(f"[warn] {line}", file=sys.stderr)
    if not report.ok:
        for line in report.errors[:40]:
            print(f"[invalid] {line}", file=sys.stderr)
        print(f"[invalid] {len(report.errors)} validation errors; refusing to write", file=sys.stderr)
        return 1

    default_name = "predictions_samples.csv" if args.samples else "output.csv"
    out_path = args.out or os.path.join(root, default_name)
    write_csv(out_path, rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
