import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, ClassVar, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

CFACategory = Literal[
    "risk_objectives",
    "return_objectives",
    "liquidity",
    "time_horizon",
    "tax_concerns",
    "legal_regulatory",
    "unique_circumstances",
]
Theme = Literal["tobacco", "weapons", "fossil_fuels", "alcohol", "gambling"]

_TICKER_RE = re.compile(r"[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,2})?")
_SIC_RE = re.compile(r"\d{4}")
_SIC_MAJOR_RE = re.compile(r"\d{2}")


def normalize_ticker(raw: str) -> str:
    """Canonical class-share form: BRK-B, BRK/B and brk.b all become BRK.B."""
    ticker = raw.strip().upper().replace("-", ".").replace("/", ".")
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"invalid ticker: {raw!r}")
    return ticker


def _to_decimal(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            raise ValueError(f"not a number: {value!r}") from None
    return value


def _to_weight(value: Any) -> Any:
    # Only an explicit "%" is rescaled; a bare 5 must stay invalid rather than guess 5%.
    if isinstance(value, str) and value.strip().endswith("%"):
        return _to_decimal(value.strip()[:-1]) / 100
    return _to_decimal(value)


Ticker = Annotated[str, AfterValidator(normalize_ticker)]
Weight = Annotated[Decimal, BeforeValidator(_to_weight), Field(gt=0, le=1)]
Amount = Annotated[Decimal, BeforeValidator(_to_decimal)]


class SecurityRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: Ticker
    name: str | None = None


class _Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cfa_category: ClassVar[CFACategory | None] = None

    # Model-emitted ids are never scored; they only let reasons point at a constraint.
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    evidence: str = Field(min_length=1)

    @field_validator("evidence")
    @classmethod
    def _evidence_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence must not be blank")
        return value


class SecurityRestriction(_Constraint):
    # Becomes legal_regulatory when the restriction is compliance-driven.
    cfa_category = "unique_circumstances"
    type: Literal["security_restriction"] = "security_restriction"
    action: Literal["no_buy", "no_sell", "no_trade"]
    security: SecurityRef
    until: date | None = None


class SectorESGExclusion(_Constraint):
    cfa_category = "unique_circumstances"
    type: Literal["sector_esg_exclusion"] = "sector_esg_exclusion"
    basis: Literal["sic", "theme"]
    sic_codes: list[str] = Field(default_factory=list)
    theme: Theme | None = None
    label: str = ""
    applies_to: Literal["no_buy", "no_buy_and_divest"] = "no_buy"

    @field_validator("sic_codes")
    @classmethod
    def _sic_codes(cls, codes: list[str]) -> list[str]:
        for code in codes:
            if not _SIC_RE.fullmatch(code):
                raise ValueError(f"SIC codes are 4 digits: {code!r}")
        return sorted(set(codes))

    @model_validator(mode="after")
    def _basis_consistency(self):
        if self.basis == "sic" and (not self.sic_codes or self.theme is not None):
            raise ValueError("basis=sic needs sic_codes and no theme")
        if self.basis == "theme" and (self.theme is None or self.sic_codes):
            raise ValueError("basis=theme needs a theme and no sic_codes")
        return self


class ConcentrationCap(_Constraint):
    cfa_category = "risk_objectives"
    type: Literal["concentration_cap"] = "concentration_cap"
    scope: Literal["single_security", "sector", "asset_class"]
    target: str | None = None
    max_weight: Weight

    @model_validator(mode="after")
    def _normalize_target(self):
        if self.target is None:
            raise ValueError("target is required")
        if self.scope == "single_security":
            self.target = normalize_ticker(self.target)
        elif self.scope == "sector":
            if not _SIC_MAJOR_RE.fullmatch(self.target):
                raise ValueError("sector target is a 2-digit SIC major group")
        else:
            self.target = self.target.strip().lower()
        return self


class CashFloor(_Constraint):
    cfa_category = "liquidity"
    type: Literal["cash_floor"] = "cash_floor"
    min_amount: Annotated[Amount, Field(ge=0)] | None = None
    min_weight: Weight | None = None

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.min_amount is None) == (self.min_weight is None):
            raise ValueError("set exactly one of min_amount, min_weight")
        return self


class CapitalGainsBudget(_Constraint):
    cfa_category = "tax_concerns"
    type: Literal["capital_gains_budget"] = "capital_gains_budget"
    max_net_realized_gain: Annotated[Amount, Field(ge=0)]
    tax_year: int = Field(ge=1990, le=2100)
    term: Literal["short", "long", "all"] = "all"


class LotHold(_Constraint):
    cfa_category = "tax_concerns"
    type: Literal["lot_hold"] = "lot_hold"
    security: SecurityRef
    lot_ids: list[str] | None = None
    hold_until: date | None = None
    reason: str = ""

    @field_validator("lot_ids")
    @classmethod
    def _lot_ids(cls, ids: list[str] | None) -> list[str] | None:
        if ids is None:
            return None
        if not ids or any(not i.strip() for i in ids):
            raise ValueError("lot_ids must be None or a non-empty list of ids")
        return sorted(set(ids))


class RecurringWithdrawal(_Constraint):
    cfa_category = "liquidity"
    type: Literal["recurring_withdrawal"] = "recurring_withdrawal"
    amount: Annotated[Amount, Field(gt=0)]
    frequency: Literal["weekly", "monthly", "quarterly", "annually"]
    start_date: date
    end_date: date | None = None

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date precedes start_date")
        return self


class ClarificationNeeded(_Constraint):
    type: Literal["clarification_needed"] = "clarification_needed"
    question: str = Field(min_length=1)
    ambiguity_kind: Literal[
        "ambiguous_security",
        "ambiguous_amount",
        "ambiguous_date",
        "ambiguous_scope",
        "conflicting_instruction",
        "other",
    ]
    related_security: SecurityRef | None = None
    candidates: list[str] = Field(default_factory=list)


Constraint = Annotated[
    SecurityRestriction | SectorESGExclusion | ConcentrationCap | CashFloor | CapitalGainsBudget | LotHold | RecurringWithdrawal | ClarificationNeeded,
    Field(discriminator="type"),
]

_constraint_adapter: TypeAdapter = TypeAdapter(Constraint)


def parse_constraint(obj: Any):
    return _constraint_adapter.validate_python(obj)


class ConstraintSet(BaseModel):
    """What a human approves: a document's final net constraints."""

    model_config = ConfigDict(extra="forbid")
    constraints: list[Constraint]
    source_hash: str | None = None
    status: Literal["draft", "approved"] = "draft"
    approved_by: str | None = None
    approved_at: datetime | None = None

    @model_validator(mode="after")
    def _approval_needs_approver(self):
        if self.status == "approved" and not (self.approved_by and self.approved_at):
            raise ValueError("approved sets need approved_by and approved_at")
        return self


class EvidenceFlag(BaseModel):
    index: int
    constraint_id: str
    evidence: str


def validate_evidence(text: str, cset: ConstraintSet) -> list[EvidenceFlag]:
    """Flag constraints whose evidence is not a verbatim substring of the source text."""
    return [
        EvidenceFlag(index=i, constraint_id=c.id, evidence=c.evidence)
        for i, c in enumerate(cset.constraints)
        if c.evidence not in text
    ]
