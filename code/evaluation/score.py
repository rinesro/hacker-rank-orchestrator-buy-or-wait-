"""Score a predicted CSV against the 25 solved samples.

    python3 code/evaluation/score.py                 # run the pipeline and score it
    python3 code/evaluation/score.py --pred FILE     # score an existing CSV
    python3 code/evaluation/score.py --diff          # per-row diff table

Reports per-field accuracy, a relaxed payment_plan match, relative-error
buckets and MAPE for amount_safe_to_pay, and token-level similarity for
decision_explanation (the ground truth uses more than one paraphrase per
case, so exact match there is not attainable and not the target).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import DEFAULT, ForecastConfig  # noqa: E402
from evidence import Evidence, stub_evidence  # noqa: E402
from loaders import Dataset  # noqa: E402
from pipeline import COLUMNS, run  # noqa: E402

EXACT_FIELDS = (
    "affordability_status",
    "recommended_payment_method",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)
TOLERANCE = Decimal("0.01")


def _dec(text: str) -> Optional[Decimal]:
    try:
        return Decimal((text or "").strip())
    except (InvalidOperation, ValueError):
        return None


def _plan(text: str):
    text = (text or "").strip()
    if not text or text == "none":
        return []
    out = []
    for chunk in text.split("|"):
        when, _, amount = chunk.partition(":")
        value = _dec(amount)
        out.append((when.strip(), value))
    return out


def _plan_relaxed(a: str, b: str) -> bool:
    left, right = _plan(a), _plan(b)
    if len(left) != len(right):
        return False
    for (d1, a1), (d2, a2) in zip(left, right):
        if d1 != d2:
            return False
        if a1 is None or a2 is None or abs(a1 - a2) > TOLERANCE:
            return False
    return True


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9.,]*", (text or "").lower())


def _token_f1(predicted: str, truth: str) -> float:
    p, t = _tokens(predicted), _tokens(truth)
    if not p or not t:
        return 1.0 if p == t else 0.0
    overlap = 0
    pool = list(t)
    for token in p:
        if token in pool:
            pool.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(t)
    return 2 * precision * recall / (precision + recall)


def score(predicted: Sequence[Dict[str, str]], truth: Sequence[Dict[str, str]]) -> Dict:
    by_id = {row["request_id"]: row for row in predicted}
    counts = {field: 0 for field in EXACT_FIELDS}
    counts["payment_plan_exact"] = 0
    counts["payment_plan_relaxed"] = 0
    counts["amount_exact"] = 0
    explanation_scores: List[float] = []
    abs_pct: List[float] = []
    buckets = {"=0.01": 0, "<=1%": 0, "<=5%": 0, "<=25%": 0, ">25%": 0, "n/a": 0}
    rows: List[Dict] = []
    total = len(truth)

    for want in truth:
        rid = want["request_id"]
        got = by_id.get(rid, {})
        row = {"request_id": rid, "diffs": []}
        for field in EXACT_FIELDS:
            if (got.get(field, "") or "").strip() == (want.get(field, "") or "").strip():
                counts[field] += 1
            else:
                row["diffs"].append(
                    f"{field}: got {got.get(field, '')!r} want {want.get(field, '')!r}"
                )
        if (got.get("payment_plan", "") or "").strip() == (want.get("payment_plan", "") or "").strip():
            counts["payment_plan_exact"] += 1
            counts["payment_plan_relaxed"] += 1
        elif _plan_relaxed(got.get("payment_plan", ""), want.get("payment_plan", "")):
            counts["payment_plan_relaxed"] += 1
            row["diffs"].append("payment_plan: relaxed match only")
        else:
            row["diffs"].append(
                f"payment_plan: got {got.get('payment_plan', '')!r} want {want.get('payment_plan', '')!r}"
            )

        pred_amount = _dec(got.get("amount_safe_to_pay", ""))
        true_amount = _dec(want.get("amount_safe_to_pay", ""))
        if pred_amount is None or true_amount is None:
            buckets["n/a"] += 1
            row["diffs"].append("amount_safe_to_pay: unparseable")
        else:
            delta = abs(pred_amount - true_amount)
            exact = delta <= TOLERANCE
            if exact:
                counts["amount_exact"] += 1
            if true_amount != 0:
                pct = float(delta / abs(true_amount)) * 100
                abs_pct.append(pct)
            else:
                pct = 0.0 if exact else float("inf")
            if exact:
                buckets["=0.01"] += 1
            elif pct <= 1:
                buckets["<=1%"] += 1
            elif pct <= 5:
                buckets["<=5%"] += 1
            elif pct <= 25:
                buckets["<=25%"] += 1
            else:
                buckets[">25%"] += 1
            if not exact:
                row["diffs"].append(
                    f"amount_safe_to_pay: got {pred_amount} want {true_amount}"
                )

        explanation_scores.append(
            _token_f1(got.get("decision_explanation", ""), want.get("decision_explanation", ""))
        )
        row["ok"] = not row["diffs"]
        rows.append(row)

    mape = sum(abs_pct) / len(abs_pct) if abs_pct else 0.0
    headline = sum(
        counts[field] for field in EXACT_FIELDS
    ) + counts["payment_plan_exact"] + counts["amount_exact"]
    return {
        "total": total,
        "counts": counts,
        "buckets": buckets,
        "mape": mape,
        "explanation_f1": sum(explanation_scores) / len(explanation_scores)
        if explanation_scores
        else 0.0,
        "rows": rows,
        "field_points": headline,
        "field_max": total * (len(EXACT_FIELDS) + 2),
    }


def format_report(result: Dict, show_diff: bool = False) -> str:
    total = result["total"]
    counts = result["counts"]
    lines = [f"samples scored: {total}", ""]
    lines.append(f"{'field':<34}{'exact':>8}{'pct':>9}")
    lines.append("-" * 51)

    def line(label: str, value: int) -> str:
        return f"{label:<34}{value:>8}{value / total * 100:>8.1f}%"

    for field in EXACT_FIELDS:
        lines.append(line(field, counts[field]))
    lines.append(line("payment_plan (exact)", counts["payment_plan_exact"]))
    lines.append(line("payment_plan (relaxed)", counts["payment_plan_relaxed"]))
    lines.append(line("amount_safe_to_pay (+/-0.01)", counts["amount_exact"]))
    lines.append("-" * 51)
    lines.append(
        f"{'overall field accuracy':<34}"
        f"{result['field_points']:>8}"
        f"{result['field_points'] / result['field_max'] * 100:>8.1f}%"
    )
    lines.append("")
    lines.append(f"amount_safe_to_pay MAPE: {result['mape']:.2f}%")
    lines.append(f"amount error buckets:    {result['buckets']}")
    lines.append(f"decision_explanation F1: {result['explanation_f1']:.3f}")

    if show_diff:
        lines.append("")
        lines.append("row diffs")
        lines.append("-" * 51)
        for row in result["rows"]:
            mark = "ok " if row["ok"] else "FAIL"
            lines.append(f"{mark} {row['request_id']}")
            for diff in row["diffs"]:
                lines.append(f"       {diff}")
    return "\n".join(lines)


def sample_truth(data: Dataset) -> List[Dict[str, str]]:
    return [
        {"request_id": request.request_id, **request.truth} for request in data.samples
    ]


def predict_samples(
    data: Dataset, evidence: Optional[Evidence] = None, cfg: ForecastConfig = DEFAULT
) -> List[Dict[str, str]]:
    return run(data, data.samples, evidence or stub_evidence(), cfg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="score predictions against the solved samples")
    parser.add_argument("--pred", default=None, help="CSV of predictions to score")
    parser.add_argument("--diff", action="store_true", help="print the per-row diff table")
    args = parser.parse_args(argv)

    data = Dataset()
    if args.pred:
        with open(args.pred, newline="", encoding="utf-8") as handle:
            predicted = list(csv.DictReader(handle))
    else:
        predicted = predict_samples(data)

    result = score(predicted, sample_truth(data))
    print(format_report(result, show_diff=args.diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
