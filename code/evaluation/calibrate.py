"""Select the forecasting model against the 25 solved samples.

    python3 code/evaluation/calibrate.py            # full sweep
    python3 code/evaluation/calibrate.py --top 8    # how many finalists to score

The problem statement says "forecast essential variable spending
conservatively" without defining the estimator, the window, or how spend is
distributed across the horizon.  Rather than guess once, every open choice is
a switch in ``config.py`` and this script picks the combination that scores
best across all 25 samples at once.

That is model selection over general rules, not fitting to rows: a config is
judged only by its aggregate score, no branch is ever keyed on a request_id or
a user_id, and a combination that wins one row while losing two is rejected.

Two stages, because the full scorer is dominated by the day-by-day search for
``earliest_date_for_full_payment``:

1. score every combination on ``amount_safe_to_pay`` alone, which needs one
   forecast per sample and is the quantity every other field depends on;
2. run the complete scorer on the finalists and rank by overall accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import replace
from decimal import Decimal
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import DEFAULT, ForecastConfig  # noqa: E402
from evidence import Evidence  # noqa: E402
from loaders import Dataset  # noqa: E402
from money import D  # noqa: E402
from pipeline import run  # noqa: E402
from simulator import max_safe_payment_on  # noqa: E402
from state import StateBuilder  # noqa: E402

from score import sample_truth, score  # noqa: E402

GRID: Dict[str, tuple] = {
    "amount_estimator": ("median", "mean", "last", "max"),
    "estimation_occurrences": (0, 4, 6, 8),
    "income_estimator": ("last", "median", "min"),
    "variable_mode": ("discrete", "daily_burn"),
    "gap_tolerance": (0.35, 0.45, 0.6),
}


def combinations() -> List[ForecastConfig]:
    keys = sorted(GRID)
    out = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        out.append(replace(DEFAULT, **dict(zip(keys, values))))
    return out


def amount_error(data: Dataset, evidence: Evidence, cfg: ForecastConfig) -> Tuple[float, int]:
    """Mean absolute percentage error on amount_safe_to_pay, and exact hits."""
    builder = StateBuilder(data, evidence, cfg)
    errors: List[float] = []
    exact = 0
    for request in data.samples:
        state = builder.build(request.user_id, request.request_date)
        ours = max_safe_payment_on(state, request.request_date, cap=request.requested_amount)
        truth = D(request.truth["amount_safe_to_pay"])
        delta = abs(ours - truth)
        if delta <= Decimal("0.01"):
            exact += 1
        denominator = abs(truth) if truth != 0 else request.requested_amount
        errors.append(float(delta / denominator) * 100 if denominator else 0.0)
    return sum(errors) / len(errors), exact


def describe(cfg: ForecastConfig) -> str:
    return (
        f"est={cfg.amount_estimator:<6} n={cfg.estimation_occurrences} "
        f"inc={cfg.income_estimator:<6} mode={cfg.variable_mode:<10} "
        f"tol={cfg.gap_tolerance}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="calibrate the forecasting model")
    parser.add_argument("--top", type=int, default=8, help="finalists to score fully")
    parser.add_argument("--evidence", choices=("stub", "ai"), default="ai")
    args = parser.parse_args(argv)

    data = Dataset()
    if args.evidence == "stub":
        from evidence import stub_evidence

        evidence = stub_evidence()
    else:
        from evidence_ai import load_cached_evidence

        evidence = load_cached_evidence(data)

    grid = combinations()
    print(f"stage 1: {len(grid)} combinations scored on amount_safe_to_pay")
    started = time.monotonic()
    stage1: List[Tuple[float, int, ForecastConfig]] = []
    for cfg in grid:
        mape, exact = amount_error(data, evidence, cfg)
        stage1.append((mape, exact, cfg))
    stage1.sort(key=lambda row: (row[0], -row[1]))
    print(f"  done in {time.monotonic() - started:.1f}s")
    print(f"  best MAPE {stage1[0][0]:.1f}%  worst {stage1[-1][0]:.1f}%")
    print()
    for mape, exact, cfg in stage1[: args.top]:
        print(f"  MAPE {mape:7.2f}%  exact {exact:2d}/25  {describe(cfg)}")

    print()
    print(f"stage 2: full scorer on the top {args.top}")
    truth = sample_truth(data)
    finalists = []
    for mape, exact, cfg in stage1[: args.top]:
        result = score(run(data, data.samples, evidence, cfg), truth)
        finalists.append((result["field_points"], result, cfg))
        print(
            f"  overall {result['field_points']:3d}/{result['field_max']} "
            f"({result['field_points'] / result['field_max'] * 100:5.1f}%)  "
            f"MAPE {result['mape']:7.2f}%  F1 {result['explanation_f1']:.3f}  {describe(cfg)}"
        )

    finalists.sort(key=lambda row: (-row[0], row[1]["mape"]))
    best_points, best_result, best_cfg = finalists[0]
    print()
    print("winner:")
    print(f"  {describe(best_cfg)}")
    print(
        f"  overall {best_points}/{best_result['field_max']} "
        f"({best_points / best_result['field_max'] * 100:.1f}%)  "
        f"MAPE {best_result['mape']:.2f}%"
    )
    print()
    print("apply by editing the defaults in code/config.py:")
    for key in sorted(GRID):
        print(f"  {key} = {getattr(best_cfg, key)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
