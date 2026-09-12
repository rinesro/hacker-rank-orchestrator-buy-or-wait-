"""The single model-call wrapper.  Every token the system spends passes through here.

Two responsibilities:

1. Make the call (Anthropic Messages API, structured output, no sampling knobs).
2. Append one JSONL row per call to ``evaluation/usage_ledger.jsonl`` with the
   provider, model, call type, real token counts from ``response.usage``,
   latency and whether the on-disk cache served it.

``usage_report.md`` is generated from that ledger, so every number in the report
is a measurement rather than an estimate.

A note on determinism: ``temperature`` and ``top_p`` are rejected by the current
models, so run-to-run identity cannot come from sampling settings.  It comes
from the evidence cache instead - a call is made once, its validated result is
written to ``code/cache/``, and every later run reads the file.  That is why the
cache is committed with the submission.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PROVIDER = "anthropic"
MODEL = "claude-opus-5"

# USD per million tokens, Anthropic first-party API rates for claude-opus-5.
PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,  # 1.25x input
        "cache_read": 0.50,  # 0.10x input
    }
}


@dataclass
class LedgerEntry:
    call_type: str  # "vision" | "messages"
    provider: str
    model: str
    batch_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0
    cache_hit: bool = False
    items: int = 0
    executor: str = "anthropic_sdk"
    note: str = ""

    def cost_usd(self) -> float:
        rates = PRICING.get(self.model)
        if rates is None:
            return 0.0
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_creation_input_tokens * rates["cache_write"]
            + self.cache_read_input_tokens * rates["cache_read"]
        ) / 1_000_000


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, entry: LedgerEntry) -> None:
        row = asdict(entry)
        row["cost_usd"] = round(entry.cost_usd(), 6)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def entries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class ModelClient:
    """Thin wrapper over ``client.messages.create`` with ledger instrumentation.

    Credentials are read from the environment by the SDK itself; no key is ever
    passed in, printed, or logged.
    """

    def __init__(self, ledger: Ledger, model: str = MODEL):
        self.ledger = ledger
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so the stub path needs no SDK

            self._client = anthropic.Anthropic()
        return self._client

    def complete_json(
        self,
        *,
        call_type: str,
        batch_id: str,
        system: str,
        content: List[Dict[str, Any]],
        schema: Dict[str, Any],
        items: int,
        max_tokens: int = 8000,
        effort: str = "low",
    ) -> Dict[str, Any]:
        """One structured-output call.  Returns the parsed JSON object."""
        started = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        usage = response.usage
        self.ledger.record(
            LedgerEntry(
                call_type=call_type,
                provider=PROVIDER,
                model=self.model,
                batch_id=batch_id,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                latency_ms=latency_ms,
                cache_hit=False,
                items=items,
            )
        )

        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(f"model declined batch {batch_id}")
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)

    def record_cache_hit(self, call_type: str, batch_id: str, items: int) -> None:
        self.ledger.record(
            LedgerEntry(
                call_type=call_type,
                provider=PROVIDER,
                model=self.model,
                batch_id=batch_id,
                cache_hit=True,
                items=items,
                executor="disk_cache",
                note="served from code/cache, no tokens spent",
            )
        )
