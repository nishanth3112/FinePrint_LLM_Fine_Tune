"""Deterministic scoring of predicted vs gold constraints. No LLM judge anywhere.

Callers must handle unparseable model output before scoring (e.g. count it as an example
with no predicted constraints); dicts passed here are validated against the schema.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel

from fineprint.schema import Constraint, SecurityRef, parse_constraint

# type -> (identity fields, value fields). Everything else (id, evidence, label, reason,
# question, candidates) is deliberately never scored.
_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "security_restriction": (("security",), ("action", "until")),
    "sector_esg_exclusion": (("basis", "theme"), ("sic_codes", "applies_to")),
    "concentration_cap": (("scope", "target"), ("max_weight",)),
    "cash_floor": ((), ("min_amount", "min_weight")),
    "capital_gains_budget": ((), ("max_net_realized_gain", "tax_year", "term")),
    "lot_hold": (("security",), ("lot_ids", "hold_until")),
    "recurring_withdrawal": ((), ("amount", "frequency", "start_date", "end_date")),
    "clarification_needed": (("related_security",), ()),
}

DEFAULT_N_BOOT = 10_000
CI_PERCENTILES = (2.5, 97.5)


def _canon(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, SecurityRef):
        return value.ticker
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return ",".join(_canon(v) for v in value)
    return str(value)


@dataclass(frozen=True)
class Normalized:
    type: str
    key: str
    values: tuple[tuple[str, str], ...]


def _coerce(item: Any):
    return item if isinstance(item, BaseModel) else parse_constraint(item)


def normalize(item: Constraint | dict) -> Normalized:
    c = _coerce(item)
    key_fields, value_fields = _SPEC[c.type]
    key = "|".join([c.type, *(_canon(getattr(c, f)) for f in key_fields)])
    values = tuple((f, _canon(getattr(c, f))) for f in value_fields)
    return Normalized(type=c.type, key=key, values=values)


@dataclass
class ValueMismatch:
    type: str
    key: str
    diffs: dict[str, tuple[str, str]]
    gold: Any
    pred: Any


@dataclass
class ExampleMatch:
    exact: list[tuple[Any, Any]] = field(default_factory=list)
    value_errors: list[ValueMismatch] = field(default_factory=list)
    hallucinated: list[Any] = field(default_factory=list)
    missed: list[Any] = field(default_factory=list)


def match_constraints(gold: Sequence, pred: Sequence) -> ExampleMatch:
    g = [(normalize(c), _coerce(c)) for c in gold]
    p = [(normalize(c), _coerce(c)) for c in pred]
    used: set[int] = set()
    result = ExampleMatch()

    def first_unused(key: str, values: tuple | None = None) -> int | None:
        return next(
            (
                j
                for j, (n, _) in enumerate(p)
                if j not in used and n.key == key and (values is None or n.values == values)
            ),
            None,
        )

    # Exact pairs first so a value-error pairing can never steal a perfect match.
    leftover = []
    for ng, cg in g:
        j = first_unused(ng.key, ng.values)
        if j is None:
            leftover.append((ng, cg))
        else:
            used.add(j)
            result.exact.append((cg, p[j][1]))

    for ng, cg in leftover:
        j = first_unused(ng.key)
        if j is None:
            result.missed.append(cg)
            continue
        used.add(j)
        np_, cp = p[j]
        diffs = {
            f: (gv, pv) for (f, gv), (_, pv) in zip(ng.values, np_.values) if gv != pv
        }
        result.value_errors.append(ValueMismatch(ng.type, ng.key, diffs, cg, cp))

    result.hallucinated = [c for j, (_, c) in enumerate(p) if j not in used]
    return result


class Counts(BaseModel):
    exact: int = 0
    value_error: int = 0
    hallucinated: int = 0
    missed: int = 0


class PRF(BaseModel):
    precision: float
    recall: float
    f1: float


class TypeScores(BaseModel):
    counts: Counts
    strict: PRF
    key_level: PRF


class Item(BaseModel):
    type: str
    key: str


class FieldError(BaseModel):
    type: str
    key: str
    field: str
    gold: str
    pred: str


class Report(BaseModel):
    n_examples: int
    n_boot: int
    seed: int
    counts: Counts
    strict: PRF
    key_level: PRF
    by_type: dict[str, TypeScores]
    value_errors: list[FieldError]
    hallucinated: list[Item]
    missed: list[Item]
    ci: dict[str, tuple[float, float]] | None


def _prf_np(tp, fp, fn):
    """Vectorised P/R/F1. Nothing to find and nothing predicted scores 1.0; otherwise an
    undefined ratio is 0.0."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    p = np.divide(tp, tp + fp, out=np.where(fn == 0, 1.0, 0.0), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.where(fp == 0, 1.0, 0.0), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return p, r, f1


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p, r, f1 = _prf_np([tp], [fp], [fn])
    return PRF(precision=float(p[0]), recall=float(r[0]), f1=float(f1[0]))


def _scores(c: Counts) -> tuple[PRF, PRF]:
    strict = _prf(c.exact, c.value_error + c.hallucinated, c.value_error + c.missed)
    key_level = _prf(c.exact + c.value_error, c.hallucinated, c.missed)
    return strict, key_level


def _type_scores(c: Counts) -> TypeScores:
    strict, key_level = _scores(c)
    return TypeScores(counts=c, strict=strict, key_level=key_level)


def score(
    examples: Sequence[tuple[Sequence, Sequence]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> Report:
    """Micro-averaged strict and key-level P/R/F1 with example-level bootstrap 95% CIs."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    total = Counts()
    by_type: dict[str, Counts] = defaultdict(Counts)
    value_errors: list[FieldError] = []
    hallucinated: list[Item] = []
    missed: list[Item] = []
    per_example = np.zeros((len(examples), 6), dtype=np.int64)

    for i, (gold, pred) in enumerate(examples):
        m = match_constraints(gold, pred)
        local = Counts(
            exact=len(m.exact),
            value_error=len(m.value_errors),
            hallucinated=len(m.hallucinated),
            missed=len(m.missed),
        )
        for g, _ in m.exact:
            by_type[g.type].exact += 1
        for ve in m.value_errors:
            by_type[ve.type].value_error += 1
            value_errors += [
                FieldError(type=ve.type, key=ve.key, field=f, gold=g, pred=p_)
                for f, (g, p_) in ve.diffs.items()
            ]
        for c in m.hallucinated:
            by_type[c.type].hallucinated += 1
            hallucinated.append(Item(type=c.type, key=normalize(c).key))
        for c in m.missed:
            by_type[c.type].missed += 1
            missed.append(Item(type=c.type, key=normalize(c).key))

        for name in Counts.model_fields:
            setattr(total, name, getattr(total, name) + getattr(local, name))
        s_tp, s_fp, s_fn = local.exact, local.value_error + local.hallucinated, local.value_error + local.missed
        per_example[i] = [s_tp, s_fp, s_fn, local.exact + local.value_error, local.hallucinated, local.missed]

    strict, key_level = _scores(total)
    return Report(
        n_examples=len(examples),
        n_boot=n_boot,
        seed=seed,
        counts=total,
        strict=strict,
        key_level=key_level,
        by_type={t: _type_scores(c) for t, c in sorted(by_type.items())},
        value_errors=value_errors,
        hallucinated=hallucinated,
        missed=missed,
        ci=_bootstrap(per_example, n_boot, seed) if len(examples) else None,
    )


def _bootstrap(per_example: np.ndarray, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    n = len(per_example)
    rng = np.random.default_rng(seed)
    # Resample whole documents: constraints within a document are not independent.
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    sums = weights @ per_example
    out: dict[str, tuple[float, float]] = {}
    for prefix, cols in (("strict", (0, 1, 2)), ("key", (3, 4, 5))):
        tp, fp, fn = (sums[:, c] for c in cols)
        for name, values in zip(("precision", "recall", "f1"), _prf_np(tp, fp, fn)):
            lo, hi = np.percentile(values, CI_PERCENTILES)
            out[f"{prefix}_{name}"] = (float(lo), float(hi))
    return out
