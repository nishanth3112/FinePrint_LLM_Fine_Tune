import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from fineprint.schema import Amount, Ticker, normalize_ticker

Quantity = Annotated[Amount, Field(gt=0)]


class TaxLot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lot_id: str
    quantity: Quantity
    cost_basis_per_share: Annotated[Amount, Field(ge=0)]
    acquired: date


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: Ticker
    price: Annotated[Amount, Field(gt=0)]
    lots: list[TaxLot]
    asset_class: str | None = None

    @property
    def quantity(self) -> Decimal:
        return sum((lot.quantity for lot in self.lots), Decimal(0))

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.price


class Account(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cash: Amount
    positions: list[Position]
    realized_gains_ytd: Amount = Decimal(0)

    @property
    def total_value(self) -> Decimal:
        return self.cash + sum((p.market_value for p in self.positions), Decimal(0))

    def position(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker == ticker), None)


class ProposedTrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: Ticker
    side: Literal["buy", "sell"]
    quantity: Quantity
    price: Annotated[Amount, Field(gt=0)]
    lot_ids: list[str] | None = None

    @property
    def value(self) -> Decimal:
        return self.quantity * self.price


class SecurityMaster(Protocol):
    def sic_for(self, ticker: str) -> str | None: ...


class DictSecurityMaster:
    def __init__(self, sic_by_ticker: Mapping[str, str]):
        self._sic = {normalize_ticker(t): sic for t, sic in sic_by_ticker.items()}

    def sic_for(self, ticker: str) -> str | None:
        return self._sic.get(normalize_ticker(ticker))

    @classmethod
    def from_json(cls, path: str | Path) -> "DictSecurityMaster":
        """Load the file written by scripts/securities_master.py."""
        securities = json.loads(Path(path).read_text())["securities"]
        return cls({ticker: rec["sic"] for ticker, rec in securities.items()})
