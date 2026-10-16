import hashlib
import json

import httpx
import pytest
import securities_master as sm

TICKER_ROWS = [
    [320193, "Apple Inc.", "AAPL", "Nasdaq"],
    [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
    [19617, "JPMorgan Chase", "JPM", "NYSE"],
    [34088, "Exxon Mobil", "XOM", "NYSE"],
    [1067983, "Berkshire Hathaway", "BRK-B", "NYSE"],
    [1, "No Sic Co", "NOSIC", "NYSE"],
    [2, "Some Trust", "TRUST", "Nasdaq"],
    [3, "Gone Co", "GONE", "NYSE"],
    [4, "Preferred", "BAC-PB", "NYSE"],
    [5, "Arca Fund", "SPY", "NYSE Arca"],
    [6, "Pink Sheet", "OTCX", "OTC"],
]
SUBMISSIONS = {
    320193: {"sic": "3571", "sicDescription": "Electronic Computers", "entityType": "operating"},
    789019: {"sic": "7372", "sicDescription": "Services-Prepackaged Software", "entityType": "operating"},
    19617: {"sic": "6022", "sicDescription": "State Commercial Banks", "entityType": "operating"},
    34088: {"sic": "2911", "sicDescription": "Petroleum Refining", "entityType": "operating"},
    1067983: {"sic": "6331", "sicDescription": "Fire, Marine & Casualty Insurance", "entityType": "operating"},
    1: {"sic": "", "sicDescription": "", "entityType": "operating"},
    2: {"sic": "6798", "sicDescription": "REITs", "entityType": "other"},
}
OK_TICKERS = {"AAPL", "MSFT", "JPM", "XOM", "BRK.B"}
UA = "FinePrint research tester@example.com"


class FakeSec:
    def __init__(self, fail_ciks=(), missing_ciks=(3,)):
        self.requests = []
        self.fail_ciks = set(fail_ciks)
        self.missing_ciks = set(missing_ciks)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("company_tickers_exchange.json"):
            return httpx.Response(
                200, json={"fields": ["cik", "name", "ticker", "exchange"], "data": TICKER_ROWS}
            )
        cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
        if cik in self.fail_ciks:
            return httpx.Response(500)
        if cik in self.missing_ciks:
            return httpx.Response(404)
        return httpx.Response(200, json={"name": "x", **SUBMISSIONS[cik]})

    @property
    def submission_calls(self):
        return [r for r in self.requests if "submissions" in str(r.url)]


def run(tmp_path, sec, **kw):
    sleeps = []
    now = [0.0]

    def sleep(seconds):  # a sleeping process lets the clock advance
        sleeps.append(seconds)
        now[0] += seconds

    client = httpx.Client(transport=httpx.MockTransport(sec))
    kw.setdefault("sample_size", 1500)
    summary = sm.run(
        out_dir=tmp_path, user_agent=UA, seed=0, rps=5, client=client,
        sleep=sleep, clock=lambda: now[0], **kw,
    )
    return summary, sleeps


def load(tmp_path):
    return json.loads((tmp_path / "securities_master.json").read_text())


def test_contact_email_required():
    with pytest.raises(sm.MissingContactError):
        sm.build_user_agent(None, {})
    with pytest.raises(sm.MissingContactError):
        sm.build_user_agent("not-an-email", {})
    assert "me@example.com" in sm.build_user_agent("me@example.com", {})
    assert sm.build_user_agent(None, {"SEC_USER_AGENT": "Lab x@y.org"}) == "Lab x@y.org"


def test_sample_filters_exchange_and_ticker_shape():
    rows = [{"cik": r[0], "name": r[1], "ticker": r[2], "exchange": r[3]} for r in TICKER_ROWS]
    got = {r["ticker"] for r in sm.sample_tickers(rows, sample_size=100, seed=0)}
    assert got == OK_TICKERS | {"NOSIC", "TRUST", "GONE"}  # BAC-PB, SPY, OTCX excluded


def test_sample_is_seeded_and_capped():
    def ticker(i):
        return "Q" + chr(65 + i // 26) + chr(65 + i % 26)

    rows = [{"cik": i, "name": f"c{i}", "ticker": ticker(i), "exchange": "NYSE"} for i in range(1, 400)]
    a = sm.sample_tickers(rows, 50, seed=1)
    assert a == sm.sample_tickers(list(reversed(rows)), 50, seed=1)
    assert a != sm.sample_tickers(rows, 50, seed=2)
    assert len(a) <= 50


def test_run_writes_master_and_manifest(tmp_path):
    sec = FakeSec()
    summary, _ = run(tmp_path, sec)
    master = load(tmp_path)
    assert set(master["securities"]) == OK_TICKERS
    assert master["securities"]["AAPL"]["sic"] == "3571"
    assert master["securities"]["BRK.B"]["cik"] == 1067983
    assert master["meta"]["seed"] == 0 and master["meta"]["fetched_at"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    digest = hashlib.sha256((tmp_path / "securities_master.json").read_bytes()).hexdigest()
    assert manifest["files"]["securities_master.json"] == digest
    assert summary["kept"] == len(OK_TICKERS)


def test_every_request_sends_user_agent(tmp_path):
    sec = FakeSec()
    run(tmp_path, sec)
    assert sec.requests and all(r.headers["user-agent"] == UA for r in sec.requests)


def test_drops_missing_sic_non_operating_and_404_and_counts_them(tmp_path):
    summary, _ = run(tmp_path, FakeSec())
    assert summary["dropped"] == {"no_sic": 1, "not_operating": 1, "not_found": 1}
    assert summary["failed"] == []


def test_rate_limited_between_requests(tmp_path):
    sec = FakeSec()
    _, sleeps = run(tmp_path, sec)
    assert len(sleeps) >= len(sec.requests) - 1
    assert all(s == pytest.approx(0.2) for s in sleeps)


def test_second_run_is_fully_cached(tmp_path):
    run(tmp_path, FakeSec())
    sec = FakeSec()
    run(tmp_path, sec)
    assert sec.requests == []


def test_refresh_refetches_ticker_list(tmp_path):
    run(tmp_path, FakeSec())
    sec = FakeSec()
    run(tmp_path, sec, refresh=True)
    assert [str(r.url).endswith("company_tickers_exchange.json") for r in sec.requests] == [True]


def test_resume_after_partial_failure(tmp_path):
    first = FakeSec(fail_ciks={34088})
    summary, sleeps = run(tmp_path, first)
    assert [f["cik"] for f in summary["failed"]] == [34088]
    assert "XOM" not in load(tmp_path)["securities"]
    assert len([r for r in first.submission_calls if "CIK0000034088" in str(r.url)]) == sm.MAX_ATTEMPTS
    assert any(s >= 1 for s in sleeps)  # backoff between retries

    second = FakeSec()
    summary, _ = run(tmp_path, second)
    assert [str(r.url).rsplit("/", 1)[1] for r in second.submission_calls] == ["CIK0000034088.json"]
    assert load(tmp_path)["securities"]["XOM"]["sic"] == "2911"
    assert summary["failed"] == []


def test_not_found_is_not_retried(tmp_path):
    sec = FakeSec()
    run(tmp_path, sec)
    assert len([r for r in sec.submission_calls if "CIK0000000003" in str(r.url)]) == 1


def test_cli_refuses_without_contact(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit) as exc:
        sm.main(["--out-dir", str(tmp_path)])
    assert exc.value.code != 0
    assert not list(tmp_path.iterdir())
