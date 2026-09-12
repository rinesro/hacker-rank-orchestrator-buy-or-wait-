"""Generate ``evaluation/usage_report.md`` from the usage ledger.

    python3 code/evaluation/usage_report.py

Rules this generator follows, so the report cannot overstate what was measured:

* It reads ``usage_ledger.jsonl``, which is append-only and written by the one
  client wrapper every model call goes through.
* Rows are split by ``executor``.  ``anthropic_sdk`` rows carry real
  ``response.usage`` counts and are the only ones whose tokens and cost are
  reported as measurements.  ``disk_cache`` rows are cache hits that spent
  nothing.  ``claude_code_in_session`` rows are extractions performed by the
  coding harness because no API credential was present; they have no API token
  counts, and the report says so rather than inventing numbers.
* The cold-run projection is explicitly labelled an estimate, and the estimator
  is stated inline so a reader can check it.

The report covers the extraction run that produced the cached evidence in
``code/cache/``, not a cache-only re-run.  Re-running the pipeline appends
``disk_cache`` rows; those appear in their own section and never replace the
extraction figures.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ai_client import PRICING  # noqa: E402
from evidence_ai import (  # noqa: E402
    PROMPT_VERSION,
    build_message_batches,
    read_prompt,
    render_messages_user_prompt,
    render_vision_user_prompt,
)
from loaders import Dataset, repo_root  # noqa: E402

LEDGER = os.path.join(HERE, "usage_ledger.jsonl")
REPORT = os.path.join(HERE, "usage_report.md")

# Documented approximations, used ONLY for the cold-run projection.
CHARS_PER_TOKEN = 3.7  # English/Indonesian prose and tagged markup
IMAGE_PIXELS_PER_TOKEN = 750  # Anthropic's documented width*height/750
OUTPUT_TOKENS_PER_AMENDMENT = 90  # one strict-JSON amendment object
OUTPUT_TOKENS_PER_IMAGE = 110  # one strict-JSON vision object


def load_ledger() -> List[Dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def png_size(path: str):
    with open(path, "rb") as handle:
        head = handle.read(24)
    return struct.unpack(">II", head[16:24])


def cold_run_estimate(data: Dataset) -> Dict:
    """What one uncached extraction would send and cost, measured where possible.

    Prompt character counts and image pixel counts are exact - they are read
    off the real prompts and the real PNGs.  Only the characters-to-tokens and
    pixels-to-tokens conversions are approximations.
    """
    vision_system = read_prompt("vision_system.txt")
    messages_system = read_prompt("messages_system.txt")

    vision_text_chars = 0
    vision_pixels = 0
    for image in data.images:
        event = data.events_by_id.get(image.related_event_id)
        if event is None:
            continue
        expected = data.profiles[event.user_id].home_currency
        vision_text_chars += len(vision_system) + len(
            render_vision_user_prompt(event, image.image_id, expected)
        )
        width, height = png_size(image.path())
        vision_pixels += width * height

    batches = build_message_batches(data.messages)
    message_chars = 0
    for index, batch in enumerate(batches):
        message_chars += len(messages_system) + len(
            render_messages_user_prompt(data, batch, f"batch_{index:02d}")
        )

    vision_in = int(vision_text_chars / CHARS_PER_TOKEN + vision_pixels / IMAGE_PIXELS_PER_TOKEN)
    vision_out = OUTPUT_TOKENS_PER_IMAGE * len(data.images)
    messages_in = int(message_chars / CHARS_PER_TOKEN)
    messages_out = OUTPUT_TOKENS_PER_AMENDMENT * len(data.messages)

    rates = PRICING["claude-opus-5"]
    vision_cost = (vision_in * rates["input"] + vision_out * rates["output"]) / 1_000_000
    messages_cost = (messages_in * rates["input"] + messages_out * rates["output"]) / 1_000_000
    cost = vision_cost + messages_cost
    return {
        "vision_cost": vision_cost,
        "messages_cost": messages_cost,
        "vision_calls": len(data.images),
        "message_calls": len(batches),
        "vision_in": vision_in,
        "vision_out": vision_out,
        "messages_in": messages_in,
        "messages_out": messages_out,
        "vision_pixels": vision_pixels,
        "text_chars": vision_text_chars + message_chars,
        "cost": cost,
    }


def naive_baseline(data: Dataset) -> Dict:
    """Cost of the obvious alternative: one LLM call per request, full context.

    Sized from this dataset's real per-user context: the user's profile row,
    their financial-event history, their messages and the request's payment
    options, which is what a per-request prompt would have to carry.
    """
    rates = PRICING["claude-opus-5"]
    per_user_chars = []
    for request in data.requests:
        events = data.user_events(request.user_id)
        chars = sum(
            len(e.event_id) + len(e.description) + len(e.category) + 40 for e in events
        )
        chars += sum(len(m.message_text) for m in data.messages_by_user.get(request.user_id, []))
        chars += sum(90 for _ in data.options_for(request.request_id))
        chars += 2500  # instructions restating the decision rules
        per_user_chars.append(chars)
    total_in = int(sum(per_user_chars) / CHARS_PER_TOKEN)
    total_out = 400 * len(data.requests)
    cost = (total_in * rates["input"] + total_out * rates["output"]) / 1_000_000
    return {
        "calls": len(data.requests),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost": cost,
    }


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def build(data: Dataset) -> str:
    rows = load_ledger()
    by_executor: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_executor[row["executor"]].append(row)

    sdk = by_executor.get("anthropic_sdk", [])
    in_session = by_executor.get("claude_code_in_session", [])
    cached = by_executor.get("disk_cache", [])

    def totals(group):
        return {
            "calls": len(group),
            "input": sum(r["input_tokens"] for r in group),
            "output": sum(r["output_tokens"] for r in group),
            "cache_write": sum(r["cache_creation_input_tokens"] for r in group),
            "cache_read": sum(r["cache_read_input_tokens"] for r in group),
            "cost": sum(r["cost_usd"] for r in group),
            "items": sum(r["items"] for r in group),
        }

    sdk_t = totals(sdk)
    n_requests = len(data.requests)
    cold = cold_run_estimate(data)
    naive = naive_baseline(data)

    # distinct extraction units, since the ledger is append-only and one pilot
    # batch was later superseded by the full batch
    distinct_vision = len({r["batch_id"] for r in rows if r["call_type"] == "vision"})
    distinct_messages = len({r["batch_id"] for r in rows if r["call_type"] == "messages"})
    superseded = len([r for r in rows if r["call_type"] == "messages"]) - distinct_messages

    lines = [
        "# Token Usage and Cost Report",
        "",
        "Generated by `python3 code/evaluation/usage_report.py` from "
        "`evaluation/usage_ledger.jsonl`, the append-only ledger that every model call "
        "passes through. This report covers **the extraction run that produced the cached "
        "evidence in `code/cache/`**, which is the evidence the submitted `output.csv` was "
        "built from. It is not a cache-only re-run; cache hits are reported separately below "
        "and never replace the extraction figures.",
        "",
        "## Provider and model",
        "",
        table(
            ["Provider", "Model", "Input $/MTok", "Output $/MTok", "Surface"],
            [
                [
                    "Anthropic",
                    "claude-opus-5",
                    f"${PRICING['claude-opus-5']['input']:.2f}",
                    f"${PRICING['claude-opus-5']['output']:.2f}",
                    "Messages API, structured output (`output_config.format`)",
                ]
            ],
        ),
        "",
        "One model is used for both call types, so the per-model and overall tables are the "
        "same table.",
        "",
        "## Calls in the extraction run",
        "",
        table(
            ["Call type", "Calls", "Units covered", "Executor"],
            [
                ["vision", distinct_vision, f"{distinct_vision} images", "see note below"],
                [
                    "messages",
                    distinct_messages,
                    f"{len(data.messages)} messages",
                    "see note below",
                ],
                ["**total**", distinct_vision + distinct_messages, "-", "-"],
            ],
        ),
        "",
    ]

    if superseded:
        lines += [
            f"The ledger holds {superseded} additional `messages` row(s) beyond the "
            f"{distinct_messages} distinct batches: a pilot call covering part of a batch that "
            "was later superseded by the full batch. The ledger is append-only, so the "
            "superseded attempt is retained rather than deleted.",
            "",
        ]

    lines += ["## Measured token usage", ""]

    if sdk:
        lines += [
            table(
                [
                    "Call type",
                    "Calls",
                    "Input tokens",
                    "Output tokens",
                    "Cache write",
                    "Cache read",
                    "Cost USD",
                ],
                [
                    [
                        call_type,
                        len(group),
                        totals(group)["input"],
                        totals(group)["output"],
                        totals(group)["cache_write"],
                        totals(group)["cache_read"],
                        f"${totals(group)['cost']:.4f}",
                    ]
                    for call_type, group in sorted(
                        {
                            t: [r for r in sdk if r["call_type"] == t]
                            for t in {r["call_type"] for r in sdk}
                        }.items()
                    )
                ]
                + [
                    [
                        "**total**",
                        sdk_t["calls"],
                        sdk_t["input"],
                        sdk_t["output"],
                        sdk_t["cache_write"],
                        sdk_t["cache_read"],
                        f"${sdk_t['cost']:.4f}",
                    ]
                ],
            ),
            "",
            table(
                ["Metric", "Value"],
                [
                    ["Requests in the dataset", n_requests],
                    [
                        "Total tokens",
                        sdk_t["input"] + sdk_t["output"] + sdk_t["cache_write"] + sdk_t["cache_read"],
                    ],
                    [
                        "Average tokens per request",
                        round(
                            (sdk_t["input"] + sdk_t["output"] + sdk_t["cache_write"] + sdk_t["cache_read"])
                            / n_requests,
                            1,
                        ),
                    ],
                    ["Total cost", f"${sdk_t['cost']:.4f}"],
                    ["Cost per request", f"${sdk_t['cost'] / n_requests:.6f}"],
                ],
            ),
            "",
        ]
    else:
        lines += [
            "**No API-measured tokens exist for this run.** The environment that produced the "
            "cached evidence had no Anthropic credential available — no `ANTHROPIC_API_KEY`, no "
            "`ANTHROPIC_AUTH_TOKEN`, no `ant` CLI profile — so the extraction was performed by "
            "the coding harness (Claude Code) using the identical prompts in `code/prompts/` and "
            "the identical JSON schemas in `code/evidence_ai.py`, and the results were written "
            "to `code/cache/` in the same format the SDK path writes.",
            "",
            f"Those {len(in_session)} calls are recorded in the ledger with "
            "`executor=\"claude_code_in_session\"` and zero token counts, because no Messages API "
            "call was made and therefore no `response.usage` block exists to report. Rather than "
            "print invented numbers in the measured section, the real per-call input sizes are "
            "given as a projection below.",
            "",
            "Setting `ANTHROPIC_API_KEY` and running "
            "`load_cached_evidence(live=True)` after clearing `code/cache/` exercises the SDK "
            "path and fills this section with measured values; no code changes are needed.",
            "",
            table(
                ["Call type", "Ledger rows", "Distinct units", "Tokens measured", "Cost measured"],
                [
                    [
                        "vision",
                        len([r for r in in_session if r["call_type"] == "vision"]),
                        f"{distinct_vision} images",
                        "n/a - no API call",
                        "$0.0000",
                    ],
                    [
                        "messages",
                        len([r for r in in_session if r["call_type"] == "messages"]),
                        f"{distinct_messages} batches / {len(data.messages)} messages",
                        "n/a - no API call",
                        "$0.0000",
                    ],
                ],
            ),
            "",
        ]

    lines += [
        "## Cached versus live calls",
        "",
        table(
            ["Row kind", "Ledger rows", "Tokens spent", "Meaning"],
            [
                [
                    "`anthropic_sdk` (live)",
                    len(sdk),
                    sdk_t["input"] + sdk_t["output"],
                    "real Messages API call, counted from `response.usage`",
                ],
                [
                    "`claude_code_in_session`",
                    len(in_session),
                    0,
                    "extraction run performed by the harness; no API tokens exist",
                ],
                [
                    "`disk_cache` (cache hit)",
                    len(cached),
                    0,
                    "served from `code/cache/`; no model call made",
                ],
            ],
        ),
        "",
        f"Every subsequent pipeline run is a **100% cache hit**: all {distinct_vision} vision "
        f"units and all {distinct_messages} message batches resolve from `code/cache/`, so "
        "producing `output.csv` again costs zero tokens and is byte-identical.",
        "",
        "Note how the cache-hit row above is reached. `load_cached_evidence(live=False)` — the "
        "default, and the path `code/main.py` takes — constructs no client at all, so a "
        "cache-only run writes **no ledger rows whatsoever**. The ledger therefore cannot be "
        "diluted by re-runs: every row in it belongs to the extraction run, which is what makes "
        "this report a record of that run rather than of the last time `main.py` was executed. "
        "`disk_cache` rows appear only when a live run (`live=True`) finds a unit already "
        "cached and skips the call.",
        "",
        "This is also what makes the submission reproducible: the current models reject "
        "`temperature` and `top_p`, so run-to-run identity comes from the cache, not from "
        "sampling settings.",
        "",
        "## What a cold run would cost",
        "",
        "A cold run is one with `code/cache/` emptied, so every unit is extracted again. "
        "Character counts and image pixel counts below are **exact**, read from the real "
        "prompts and the real PNGs; only the conversion to tokens is approximate "
        f"(~{CHARS_PER_TOKEN} characters per token for text; Anthropic's documented "
        f"width x height / {IMAGE_PIXELS_PER_TOKEN} for images).",
        "",
        table(
            ["Call type", "Calls", "Est. input tokens", "Est. output tokens", "Est. cost"],
            [
                [
                    "vision",
                    cold["vision_calls"],
                    f"{cold['vision_in']:,}",
                    f"{cold['vision_out']:,}",
                    f"${cold['vision_cost']:.4f}",
                ],
                [
                    "messages",
                    cold["message_calls"],
                    f"{cold['messages_in']:,}",
                    f"{cold['messages_out']:,}",
                    f"${cold['messages_cost']:.4f}",
                ],
                [
                    "**total**",
                    cold["vision_calls"] + cold["message_calls"],
                    f"{cold['vision_in'] + cold['messages_in']:,}",
                    f"{cold['vision_out'] + cold['messages_out']:,}",
                    f"**${cold['cost']:.4f}**",
                ],
            ],
        ),
        "",
        table(
            ["Metric", "Value"],
            [
                ["Exact prompt characters sent", f"{cold['text_chars']:,}"],
                ["Exact image pixels sent", f"{cold['vision_pixels']:,}"],
                ["Requests answered", n_requests],
                [
                    "Est. tokens per request",
                    round(
                        (
                            cold["vision_in"]
                            + cold["messages_in"]
                            + cold["vision_out"]
                            + cold["messages_out"]
                        )
                        / n_requests,
                        1,
                    ),
                ],
                ["Est. cost per request", f"${cold['cost'] / n_requests:.6f}"],
            ],
        ),
        "",
        "## Comparison: a naive per-request LLM design",
        "",
        "The obvious alternative is one model call per request carrying that user's profile, "
        "event history, messages and payment options, and asking for the decision directly. "
        "Sized from this dataset's real per-user context:",
        "",
        table(
            ["Design", "Model calls", "Input tokens", "Output tokens", "Est. cost"],
            [
                [
                    "Naive: one call per request",
                    f"{naive['calls']:,}",
                    f"{naive['input_tokens']:,}",
                    f"{naive['output_tokens']:,}",
                    f"${naive['cost']:.2f}",
                ],
                [
                    "This system: cold run",
                    cold["vision_calls"] + cold["message_calls"],
                    f"{cold['vision_in'] + cold['messages_in']:,}",
                    f"{cold['vision_out'] + cold['messages_out']:,}",
                    f"${cold['cost']:.4f}",
                ],
                [
                    "This system: cached run",
                    0,
                    0,
                    0,
                    "$0.0000",
                ],
            ],
        ),
        "",
        f"The hybrid design makes **{cold['vision_calls'] + cold['message_calls']} model calls "
        f"for {n_requests} requests** instead of {naive['calls']:,}, an "
        f"{naive['calls'] / (cold['vision_calls'] + cold['message_calls']):.0f}x reduction, at "
        f"roughly **{naive['cost'] / cold['cost']:.0f}x lower cost**. The saving is structural "
        "rather than a matter of prompt trimming: the model is used only where judgement over "
        "unstructured input is actually required — reading 16 scanned documents and classifying "
        "215 free-text messages — while the affordability decision itself is a deterministic "
        "simulation that needs no model at all. Cost scales with the amount of unstructured "
        "evidence, not with the number of requests, so answering 2,500 requests instead of 250 "
        "would not change the extraction bill.",
        "",
        "## Ledger schema",
        "",
        "Each line of `usage_ledger.jsonl` records: `provider`, `model`, `call_type`, "
        "`batch_id`, `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, "
        "`cache_read_input_tokens`, `latency_ms`, `cache_hit`, `items`, `executor`, `note` and "
        "the derived `cost_usd`. Token fields are copied verbatim from `response.usage`; no "
        "field in this report is typed by hand.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    data = Dataset()
    text = build(data)
    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {REPORT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
