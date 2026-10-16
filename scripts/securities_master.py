"""Build a small securities master (ticker -> SIC) from SEC EDGAR, cached under data/raw/sec/.

    uv run python scripts/securities_master.py --contact-email you@example.com

SEC requires a User-Agent that includes a contact email, so the script refuses to run
without one (--contact-email, or a full SEC_USER_AGENT string in the environment).

Scope: a seeded random sample of ~1,500 NYSE + Nasdaq tickers whose shape looks like common
stock; SIC comes from each company's submissions JSON at <= 5 req/s. Funds, trusts and other
non-operating entities, and companies without a SIC, are dropped and counted.

TODO: switch to SEC's bulk submissions archive to cover the whole market in one download.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from fineprint.schema import normalize_ticker

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EXCHANGES = {"NYSE", "Nasdaq"}
# Plain tickers plus class shares (BRK-B); excludes preferreds like BAC-PB.
TICKER_RE = re.compile(r"[A-Z]{1,5}(-[A-B])?")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
DEFAULT_SAMPLE_SIZE = 1500
DEFAULT_RPS = 5.0
MAX_ATTEMPTS = 3
RETRYABLE = {429, 500, 502, 503, 504}
CACHE_FLUSH_EVERY = 100
TICKERS_FILE = "company_tickers_exchange.json"
SIC_CACHE_FILE = "sic_by_cik.json"
MASTER_FILE = "securities_master.json"
MANIFEST_FILE = "manifest.json"


class MissingContactError(ValueError):
    pass


class FetchError(RuntimeError):
    pass


def build_user_agent(contact_email: str | None, env: Mapping[str, str]) -> str:
    if contact_email:
        if not EMAIL_RE.fullmatch(contact_email):
            raise MissingContactError(f"not a valid email: {contact_email!r}")
        return f"FinePrint research {contact_email}"
    ua = env.get("SEC_USER_AGENT", "")
    if "@" in ua:
        return ua
    raise MissingContactError(
        "SEC requires a contact email: pass --contact-email or set SEC_USER_AGENT"
    )


def sample_tickers(rows: Sequence[Mapping], sample_size: int, seed: int) -> list[dict]:
    eligible = [
        {**r, "ticker": normalize_ticker(r["ticker"])}
        for r in rows
        if r["exchange"] in EXCHANGES and TICKER_RE.fullmatch(r["ticker"])
    ]
    # Sort before sampling so the result does not depend on the file's row order.
    eligible.sort(key=lambda r: r["ticker"])
    picked = random.Random(seed).sample(eligible, min(sample_size, len(eligible)))
    return sorted(picked, key=lambda r: r["ticker"])


class _Fetcher:
    def __init__(self, client: httpx.Client, user_agent: str, rps: float,
                 sleep: Callable[[float], None], clock: Callable[[], float]):
        self._client, self._ua = client, user_agent
        self._interval = 1.0 / rps
        self._sleep, self._clock = sleep, clock
        self._last = float("-inf")

    def _throttle(self) -> None:
        now = self._clock()
        wait = max(0.0, self._last + self._interval - now)
        if wait:
            self._sleep(wait)
        self._last = now + wait

    def get(self, url: str) -> httpx.Response:
        """200 or 404 responses are returned; everything else retries or raises FetchError."""
        last_error = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                resp = self._client.get(url, headers={"User-Agent": self._ua})
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code in (200, 404):
                    return resp
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE:
                    break
            if attempt < MAX_ATTEMPTS:
                self._sleep(2.0**attempt)
        raise FetchError(f"{url}: {last_error}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def run(
    *,
    out_dir: Path,
    user_agent: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
    rps: float = DEFAULT_RPS,
    refresh: bool = False,
    client: httpx.Client,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetcher = _Fetcher(client, user_agent, rps, sleep, clock)
    prior = _read_json(out_dir / MANIFEST_FILE, {})

    tickers_path = out_dir / TICKERS_FILE
    if refresh or not tickers_path.exists():
        resp = fetcher.get(TICKERS_URL)
        if resp.status_code != 200:
            raise FetchError(f"{TICKERS_URL}: HTTP {resp.status_code}")
        tickers_path.write_text(resp.text)
        tickers_fetched_at = _now()
    else:
        tickers_fetched_at = prior.get("tickers_fetched_at") or _now()
    raw = json.loads(tickers_path.read_text())
    rows = [dict(zip(raw["fields"], row)) for row in raw["data"]]
    sample = sample_tickers(rows, sample_size, seed)

    cache_path = out_dir / SIC_CACHE_FILE
    cache: dict[str, dict] = _read_json(cache_path, {})
    failed: list[dict] = []
    fetched = 0
    try:
        for cik in sorted({r["cik"] for r in sample}):
            if str(cik) in cache:
                continue
            try:
                resp = fetcher.get(SUBMISSIONS_URL.format(cik=cik))
            except FetchError as exc:
                failed.append({"cik": cik, "error": str(exc)})
                continue
            if resp.status_code == 404:
                cache[str(cik)] = {"status": "not_found"}
            else:
                body = resp.json()
                cache[str(cik)] = {
                    "status": "ok",
                    "sic": body.get("sic") or "",
                    "sic_description": body.get("sicDescription") or "",
                    "entity_type": body.get("entityType") or "",
                }
            fetched += 1
            if fetched % CACHE_FLUSH_EVERY == 0:
                cache_path.write_text(json.dumps(cache, sort_keys=True))
    finally:
        cache_path.write_text(json.dumps(cache, sort_keys=True))

    securities: dict[str, dict] = {}
    dropped = {"no_sic": 0, "not_operating": 0, "not_found": 0}
    for r in sample:
        rec = cache.get(str(r["cik"]))
        if rec is None:  # fetch failed this run; it is reported under `failed`
            continue
        if rec["status"] == "not_found":
            dropped["not_found"] += 1
        elif rec["entity_type"] != "operating":
            dropped["not_operating"] += 1
        elif not rec["sic"]:
            dropped["no_sic"] += 1
        else:
            securities[r["ticker"]] = {
                "cik": r["cik"],
                "name": r["name"],
                "exchange": r["exchange"],
                "sic": rec["sic"],
                "sic_description": rec["sic_description"],
            }

    summary = {
        "requested": len(sample),
        "kept": len(securities),
        "dropped": dropped,
        "failed": failed,
    }
    master_path = out_dir / MASTER_FILE
    master_path.write_text(
        json.dumps(
            {
                "meta": {
                    "fetched_at": tickers_fetched_at,
                    "built_at": _now(),
                    "seed": seed,
                    "sample_size": sample_size,
                    "sources": [TICKERS_URL, SUBMISSIONS_URL],
                    **summary,
                },
                "securities": dict(sorted(securities.items())),
            },
            indent=1,
        )
    )
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(
            {
                "tickers_fetched_at": tickers_fetched_at,
                "built_at": _now(),
                "seed": seed,
                "sample_size": sample_size,
                "files": {
                    name: _sha256(out_dir / name)
                    for name in (TICKERS_FILE, SIC_CACHE_FILE, MASTER_FILE)
                },
            },
            indent=1,
        )
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--contact-email", help="required by SEC for the User-Agent header")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/sec"))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rps", type=float, default=DEFAULT_RPS)
    parser.add_argument("--refresh", action="store_true", help="re-download the ticker list")
    args = parser.parse_args(argv)
    try:
        user_agent = build_user_agent(args.contact_email, os.environ)
    except MissingContactError as exc:
        parser.error(str(exc))
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        summary = run(
            out_dir=args.out_dir, user_agent=user_agent, sample_size=args.sample_size,
            seed=args.seed, rps=args.rps, refresh=args.refresh, client=client,
        )
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
