"""Layer 1 - narrow AI evidence extraction, cached to disk.

Two extractors, both producing the typed records in ``evidence.py``:

* ``extract_images``   16 vision calls, one per blank-amount event.
* ``extract_messages`` the 215 messages, batched by user, ~15 per call.

Everything is validated after the model returns.  The model's output is treated
as a proposal, never as truth:

* ``amendment_type`` must be a member of the closed enum, else NO_FINANCIAL_IMPACT.
* ``applies_to_event_id`` must exist in financial_events.csv AND belong to the
  message's own user, else it is dropped to null.  A message can never conjure
  an event.
* amounts, percentages and dates must parse, else the field is dropped.
* a message flagged ``contains_instructions`` is forced to NO_FINANCIAL_IMPACT
  regardless of what the model classified it as.

Results are cached under ``code/cache/`` keyed by a content hash of the prompt
text, model id and payload, so a re-run costs zero tokens and produces
byte-identical output.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ai_client import Ledger, ModelClient
from evidence import Amendment, AmendmentType, Evidence, ImageAmount
from loaders import Dataset, Event, Message, repo_root

PROMPT_VERSION = "v1"
MESSAGES_PER_BATCH = 15

VISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "extracted_amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "document_date": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "evidence_snippet": {"type": "string"},
        "contains_instructions": {"type": "boolean"},
    },
    "required": [
        "event_id",
        "extracted_amount",
        "currency",
        "document_date",
        "confidence",
        "evidence_snippet",
        "contains_instructions",
    ],
    "additionalProperties": False,
}

AMENDMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "message_id": {"type": "string"},
        "user_id": {"type": "string"},
        "amendment_type": {"type": "string", "enum": [t.value for t in AmendmentType]},
        "applies_to_event_id": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"]},
        "amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "percent": {"type": ["number", "null"]},
        "category": {"type": ["string", "null"]},
        "recurrence": {"type": ["string", "null"], "enum": ["one_off", "recurring", None]},
        "confidence": {"type": "number"},
        "quote": {"type": "string"},
        "contains_instructions": {"type": "boolean"},
    },
    "required": [
        "message_id",
        "user_id",
        "amendment_type",
        "applies_to_event_id",
        "effective_date",
        "amount",
        "currency",
        "percent",
        "category",
        "recurrence",
        "confidence",
        "quote",
        "contains_instructions",
    ],
    "additionalProperties": False,
}

MESSAGES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"amendments": {"type": "array", "items": AMENDMENT_SCHEMA}},
    "required": ["amendments"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# paths and caching
# --------------------------------------------------------------------------


def cache_dir() -> str:
    return os.path.join(repo_root(), "code", "cache")


def prompts_dir() -> str:
    return os.path.join(repo_root(), "code", "prompts")


def read_prompt(name: str) -> str:
    with open(os.path.join(prompts_dir(), name), encoding="utf-8") as handle:
        return handle.read()


def content_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def cache_read(name: str) -> Optional[Any]:
    path = os.path.join(cache_dir(), name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def cache_write(name: str, payload: Any) -> None:
    os.makedirs(cache_dir(), exist_ok=True)
    path = os.path.join(cache_dir(), name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------
# untrusted-data wrapping
# --------------------------------------------------------------------------


def sanitise(text: str) -> str:
    """Neutralise any attempt to close the untrusted-data wrapper from inside."""
    return (
        text.replace("</untrusted_data>", "<!-- /untrusted_data -->")
        .replace("<untrusted_data", "<!-- untrusted_data")
        .replace("</message>", "<!-- /message -->")
    )


def message_block(message: Message) -> str:
    return (
        f'<message id="{message.message_id}" user="{message.user_id}" '
        f'sent_at="{message.sent_at}" source_type="{message.source_type}">\n'
        f"{sanitise(message.message_text)}\n"
        f"</message>"
    )


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def build_message_batches(
    messages: Sequence[Message], per_batch: int = MESSAGES_PER_BATCH
) -> List[List[Message]]:
    """Group by user, keep a user's messages together, cap the batch size.

    Grouping by user matters: several archetypes reference an earlier notice
    from the same employer ("this replaces the payroll date shown in the earlier
    update"), which can only be resolved with the sibling messages in view.
    Ordering is fully determined by (user_id, message_id), so batch boundaries
    are stable across runs and the cache keys are reproducible.
    """
    by_user: Dict[str, List[Message]] = {}
    for message in sorted(messages, key=lambda m: (m.user_id, m.message_id)):
        by_user.setdefault(message.user_id, []).append(message)

    batches: List[List[Message]] = []
    current: List[Message] = []
    for user_id in sorted(by_user):
        group = by_user[user_id]
        if len(group) >= per_batch:
            if current:
                batches.append(current)
                current = []
            for start in range(0, len(group), per_batch):
                batches.append(group[start : start + per_batch])
            continue
        if len(current) + len(group) > per_batch:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def allowed_event_ids(data: Dataset, messages: Sequence[Message]) -> List[str]:
    """Only the ids a message in this batch is permitted to reference."""
    ids: List[str] = []
    for message in messages:
        if message.related_event_id:
            ids.append(message.related_event_id)
    return sorted(set(ids))


def render_messages_user_prompt(data: Dataset, batch: Sequence[Message], batch_id: str) -> str:
    template = read_prompt("messages_user.txt")
    ids = allowed_event_ids(data, batch)
    return template.format(
        batch_id=batch_id,
        message_count=len(batch),
        allowed_event_ids="\n".join(ids) if ids else "(none - every applies_to_event_id must be null)",
        message_blocks="\n".join(message_block(m) for m in batch),
    )


def render_vision_user_prompt(event: Event, image_id: str, expected_currency: str) -> str:
    template = read_prompt("vision_user.txt")
    return template.format(
        event_id=event.event_id,
        description=event.description,
        category=event.category,
        direction=event.direction,
        expected_currency=expected_currency,
        event_date=event.event_date.isoformat(),
        settlement_date=event.settlement_date.isoformat(),
        status=event.status,
        image_id=image_id,
    )


def image_block(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        encoded = base64.standard_b64encode(handle.read()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def validate_image_record(record: Dict[str, Any], event: Event, image_id: str,
                          expected_currency: str) -> Optional[ImageAmount]:
    if record.get("event_id") != event.event_id:
        return None
    amount = _decimal(record.get("extracted_amount"))
    if amount is None or amount < 0:
        return None
    currency = (record.get("currency") or "").strip().upper() or event.currency
    return ImageAmount(
        image_id=image_id,
        event_id=event.event_id,
        amount=amount,
        currency=currency,
        document_date=_date(record.get("document_date")),
        confidence=_decimal(record.get("confidence")) or Decimal("0"),
        evidence_snippet=str(record.get("evidence_snippet", ""))[:120],
        currency_matches_event=(currency == event.currency),
    )


def validate_amendment(record: Dict[str, Any], message: Message, data: Dataset) -> Amendment:
    """Every field is checked against reality before it can reach Layer 2."""
    raw_type = str(record.get("amendment_type", "")).strip()
    try:
        kind = AmendmentType(raw_type)
    except ValueError:
        kind = AmendmentType.NO_FINANCIAL_IMPACT

    if record.get("contains_instructions"):
        # An injection attempt is never a financial fact, whatever it claimed.
        kind = AmendmentType.NO_FINANCIAL_IMPACT

    event_id = (record.get("applies_to_event_id") or "").strip() or None
    if event_id is not None:
        event = data.events_by_id.get(event_id)
        if event is None or event.user_id != message.user_id:
            event_id = None  # a message may never invent an event

    percent = _decimal(record.get("percent"))
    amount = _decimal(record.get("amount"))
    if percent is not None and amount is not None:
        amount = None  # the schema allows one or the other, not both

    recurrence = record.get("recurrence")
    if recurrence not in ("one_off", "recurring", None):
        recurrence = None

    return Amendment(
        message_id=message.message_id,
        user_id=message.user_id,
        amendment_type=kind,
        applies_to_event_id=event_id,
        effective_date=_date(record.get("effective_date")),
        amount=amount,
        currency=(record.get("currency") or None),
        percent=percent,
        category=(record.get("category") or None),
        recurrence=recurrence,
        confidence=_decimal(record.get("confidence")) or Decimal("0"),
        quote=str(record.get("quote", ""))[:120],
    )


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------


def extract_images(data: Dataset, client: Optional[ModelClient]) -> Dict[str, ImageAmount]:
    system = read_prompt("vision_system.txt")
    out: Dict[str, ImageAmount] = {}
    for image in data.images:
        event = data.events_by_id.get(image.related_event_id)
        if event is None:
            continue
        expected = data.profiles[event.user_id].home_currency
        user_prompt = render_vision_user_prompt(event, image.image_id, expected)
        with open(image.path(), "rb") as handle:
            image_hash = hashlib.sha256(handle.read()).hexdigest()[:16]
        key = content_key(PROMPT_VERSION, system, user_prompt, image_hash)
        name = f"vision_{image.image_id}_{key}.json"

        record = cache_read(name)
        if record is None:
            if client is None:
                continue
            record = client.complete_json(
                call_type="vision",
                batch_id=image.image_id,
                system=system,
                content=[image_block(image.path()), {"type": "text", "text": user_prompt}],
                schema=VISION_SCHEMA,
                items=1,
                max_tokens=1000,
            )
            cache_write(name, record)
        elif client is not None:
            client.record_cache_hit("vision", image.image_id, 1)

        parsed = validate_image_record(record, event, image.image_id, expected)
        if parsed is not None:
            out[event.event_id] = parsed
    return out


def extract_messages(
    data: Dataset, client: Optional[ModelClient]
) -> Dict[str, List[Amendment]]:
    system = read_prompt("messages_system.txt")
    batches = build_message_batches(data.messages)
    out: Dict[str, List[Amendment]] = {}

    for index, batch in enumerate(batches):
        batch_id = f"batch_{index:02d}"
        user_prompt = render_messages_user_prompt(data, batch, batch_id)
        key = content_key(PROMPT_VERSION, system, user_prompt)
        name = f"messages_{batch_id}_{key}.json"

        payload = cache_read(name)
        if payload is None:
            if client is None:
                continue
            payload = client.complete_json(
                call_type="messages",
                batch_id=batch_id,
                system=system,
                content=[{"type": "text", "text": user_prompt}],
                schema=MESSAGES_SCHEMA,
                items=len(batch),
                max_tokens=6000,
            )
            cache_write(name, payload)
        elif client is not None:
            client.record_cache_hit("messages", batch_id, len(batch))

        by_id = {m.message_id: m for m in batch}
        for record in payload.get("amendments", []):
            message = by_id.get(str(record.get("message_id", "")))
            if message is None:
                continue  # the model may not answer about a message it was not given
            amendment = validate_amendment(record, message, data)
            out.setdefault(amendment.user_id, []).append(amendment)

    for amendments in out.values():
        amendments.sort(key=lambda a: a.message_id)
    return out


def load_cached_evidence(data: Optional[Dataset] = None, live: bool = False) -> Evidence:
    """Entry point used by ``main.py``.

    ``live=False`` (the default) reads only what is already cached, so the full
    pipeline runs offline and for free once the evidence has been extracted.
    """
    data = data or Dataset()
    client = None
    if live:
        ledger = Ledger(os.path.join(repo_root(), "code", "evaluation", "usage_ledger.jsonl"))
        client = ModelClient(ledger)
    return Evidence(
        image_amounts=extract_images(data, client),
        amendments_by_user=extract_messages(data, client),
    )
