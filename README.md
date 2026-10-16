# FinePrint

A fine-tuned small model turns advisor and client instructions (CRM notes, emails, IPS text)
into typed trade constraints. A deterministic pre-trade checker enforces them, and a human
approves each constraint set. The model never decides allow/block; the checker does.

Phase 1 (this state) is foundations only: schema, checker, eval matcher, securities master.
No training code yet. See `.planning/specs/phase1.md`.

## Layout

| Path | Purpose |
|---|---|
| `src/fineprint/schema.py` | Constraint types, `ConstraintSet` (human approval), `validate_evidence` |
| `src/fineprint/checker.py` | `check()` allow/block per trade with reasons; `find_conflicts()` |
| `src/fineprint/eval/match.py` | Strict and key-level P/R/F1, value errors, bootstrap 95% CIs |
| `configs/theme_sic.json` | Theme to SIC table (approximation, source URL inside) |
| `scripts/securities_master.py` | SEC ticker + SIC sample cached under `data/raw/sec/` |

## Usage

```
uv sync
uv run pytest
uv run python scripts/securities_master.py --contact-email you@example.com
```

SEC requires a contact email in the User-Agent, so the script refuses to run without one.

## CI

Every PR into `main` must pass `ci-success`, which aggregates:

| Job | What it enforces |
|---|---|
| `lint` | `uv lock --check` (lockfile in sync) and `ruff check` |
| `test` | `pytest` with a 90% coverage floor |
| `build` | `uv build` produces an sdist and wheel |
| `data-guard` | existing files in `data/gold` and `data/test` are never modified, renamed or deleted (new files are fine) |

Run the same checks locally:

```
uv lock --check && uv run ruff check . && uv run pytest --cov && uv build
```

Training and serving extras are not installed in CI; they need a GPU machine.

## Known limitations

- **Fund and ETF look-through is deferred.** Exclusions and sector caps only see operating
  companies with a SIC code; a fund holding an excluded name is not detected.
- **Theme to SIC is an approximation.** SIC is coarse: casino operators sit under codes that
  also contain non-gambling firms, SIC 2080 covers all beverages, and weapons are approximated
  by 3480/3760.
- Each document is scored on its final net constraints; there is no cross-document state or
  "lift" constraint.
- The checker judges each trade independently against the pre-trade account. Wash sales are
  ignored, and `realized_gains_ytd` is not split by short/long term.
- The securities master is a ~1,500-ticker sample, not the whole market (bulk archive is a TODO).
