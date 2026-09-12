# Buy or Wait? — solution

A hybrid system: a deterministic financial simulation that makes every
decision, with a model used only where judgement over unstructured input is
genuinely required — reading 16 scanned documents and classifying 215 free-text
messages.

## Run it

```bash
cd <repo root>
python3 code/main.py            # reads dataset/, writes ./output.csv
```

Python 3.9+ standard library only. No third-party packages are needed for the
default run: the AI evidence is already extracted and cached in `code/cache/`,
so the pipeline is offline, costs nothing, and takes about 1.3 seconds for all
250 requests.

```bash
python3 code/main.py --samples              # score-able run over the 25 solved samples
python3 code/main.py --no-ai                # force the zero-evidence baseline
python3 code/tests/test_simulator.py -v     # 23 simulator tests
python3 code/evaluation/score.py --compare  # sample score, with and without evidence
python3 code/evaluation/validate.py         # re-verify a finished output.csv
python3 code/evaluation/calibrate.py        # re-run the model-selection sweep
python3 code/evaluation/usage_report.py     # regenerate usage_report.md from the ledger
```

### Re-extracting the evidence (optional, costs money)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # read from the environment only, never stored
rm -rf code/cache/*                 # force a cold run
python3 -c "import sys; sys.path.insert(0,'code'); \
            from evidence_ai import load_cached_evidence; load_cached_evidence(live=True)"
```

Estimated cold-run cost: **31 calls, ~$0.94 total, ~$0.0038 per request**. See
`evaluation/usage_report.md`.

## Architecture

```
Layer 0  loaders.py      typed records, Decimal money, dated FX conversion
Layer 1  evidence_ai.py  16 vision calls + 215 messages in 15 batched calls,
         ai_client.py    cached to code/cache/, every call logged to the ledger
         prompts/        the exact system and user prompts used
Layer 2  state.py        cash-state rules, recurrence detection, amendments
Layer 3  simulator.py    the 90-day safety check — everything is built on this
Layer 4  planner.py      candidate generation and the spec's six-level ranking
Layer 5  validator.py    hard contract gates, run before output.csv is written
Layer 6  explain.py      deterministic explanation templates
         config.py       every open modelling choice, as a named switch
```

**Layer 3 is the core.** A payment on day *d* lowers every balance from *d*
onwards by exactly its amount and leaves earlier days untouched, so the minimum
projected balance is piecewise linear in the payment amount. That gives
`amount_safe_to_pay` a closed form; `max_safe_payment_by_search` re-derives it
by bisection and a property test asserts the two always agree.

## Determinism

Two consecutive runs produce byte-identical `output.csv`. The current models
reject `temperature` and `top_p`, so run-to-run identity cannot come from
sampling settings — it comes from the evidence cache. Extract once, commit
`code/cache/`, and every later run reads files. Iteration order is sorted
throughout and no wall-clock value enters a decision.

## Untrusted evidence

Messages and images are wrapped in tagged data blocks, the system prompts state
that their content is data and never instruction, and any attempt to close the
wrapper from inside is neutralised. Model output is treated as a proposal, not
truth: the amendment enum is closed, a message may only cite an `event_id` that
exists *and* belongs to that user, amounts and dates must parse, and anything
flagged as carrying instructions is forced to `no_financial_impact` whatever it
claimed to be. Both advance-fee scam messages in the dataset are caught this
way.

## No fitting to the solved samples

`validator.assert_no_hardcoded_answers` walks the source tree and fails the run
if any dataset identifier — request, user, event, image, message or payment
option — appears in it. The open forecasting choices are selected by
`evaluation/calibrate.py` on the aggregate score across all 25 samples at once,
never per row. `evaluation/semantics.md` records every rule with the sample
rows that prove it, and the questions the samples cannot settle are listed as
open rather than guessed at.

## Where to look

| File | What it holds |
|---|---|
| `evaluation/semantics.md` | 10 output-contract rules, 6 forecast rules, 4 open questions, each with proving rows |
| `evaluation/usage_report.md` | token and cost accounting, generated from the ledger |
| `evaluation/usage_ledger.jsonl` | append-only record of every model call |
| `evaluation/score.py` | per-field accuracy, row diffs, stub-vs-evidence delta |
| `evaluation/validate.py` | standalone contract check on a finished `output.csv` |
| `prompts/` | the exact prompts, versioned |
| `cache/` | extracted evidence, so a re-run costs nothing |
