"""Massive market data adapter — equities, futures and indices.

Massive is Polygon.io, rebranded in October 2025. The company, the API and the
endpoints are the same; the host moved to ``api.massive.com`` and
``api.polygon.io`` runs in parallel. Anyone who has integrated Polygon before
will recognise everything here, which is why this adapter accepts either host.

Unlike the Databento adapter, this one speaks REST directly rather than through
a vendor SDK. The reasoning that forced an SDK there does not apply: Databento's
wire format is DBN, a binary encoding whose fixed-point prices are easy to
mis-scale, while Massive returns plain JSON with prices already in dollars.
There is nothing to decode wrongly.

What there *is* to get wrong is that **Massive is two different APIs wearing one
name.**

Equities and futures do not share a path, a response schema, a resolution
vocabulary, or a time zone:

===============  ==========================================  ==================
                 equities                                    futures
===============  ==========================================  ==================
path             ``/v2/aggs/ticker/{t}/range/{m}/{span}/…``  ``/futures/v1/aggs/{t}``
timestamp key    ``t``                                       ``window_start``
price keys       ``o h l c v``                               ``open high low close volume``
minute bars      ``timespan=minute``                         ``resolution=1min``
daily bars       ``timespan=day``                            ``resolution=1session``
session times    Eastern                                     Central
===============  ==========================================  ==================

Two of those differences are the dangerous kind. A futures response parsed with
the equities key names yields nothing rather than an error, and ``day`` sent
where ``session`` is expected is rejected rather than silently wrong — so the
first fails quietly and the second loudly. Both are handled by keeping the two
paths genuinely separate below instead of parameterising one.

**Continuous contracts do not exist here.** Massive addresses individual
contracts by their exchange ticker — ``ESH6`` is the March 2026 E-mini. A bare
``ES`` resolves to nothing. This adapter looks the root up in the contracts
endpoint and fetches the **front month**, then says which contract it used. It
deliberately does *not* stitch several contracts into one long series: the price
gap at a roll is not a return, and a provider that silently concatenated
contracts would manufacture exactly that error. For a genuine multi-year
continuous series use :mod:`axiom.data.databento`, which has roll rules and
reports every stitch.

Credentials
-----------
::

    MASSIVE_API_KEY=...

(``POLYGON_API_KEY`` is also accepted — same key, older name.) Get one at
massive.com → Dashboard → API Keys.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.provenance import Provenance
from axiom.core.types import AssetClass
from axiom.data.base import (
    BarRequest,
    BaseProvider,
    ProviderError,
    ProviderUnavailableError,
)

__all__ = ["DEFAULT_HOST", "MassiveProvider", "MassiveSeriesInfo"]

DEFAULT_HOST = "https://api.massive.com"

#: The pre-rebrand host. Still live, still serving the same data with the same
#: keys — accepted so an existing Polygon integration needs no edits.
POLYGON_HOST = "https://api.polygon.io"

#: Equity/index timeframes → (multiplier, timespan) for the v2 aggregates path.
_EQUITY_SPANS: dict[str, tuple[int, str]] = {
    "1s": (1, "second"), "1m": (1, "minute"), "5m": (5, "minute"),
    "15m": (15, "minute"), "30m": (30, "minute"), "1h": (1, "hour"),
    "4h": (4, "hour"), "1d": (1, "day"), "1w": (1, "week"),
}

#: Futures timeframes → the `resolution` string. A separate vocabulary, not an
#: abbreviation of the one above: `min` not `minute`, and a trading `session`
#: rather than a calendar `day` — futures sessions span midnight, so the two are
#: genuinely different bars and not two spellings of one.
_FUTURES_RESOLUTIONS: dict[str, str] = {
    "1s": "1sec", "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "4h": "4hour", "1d": "1session",
    "1w": "1week",
}

#: Massive caps a response at 50,000 rows and paginates with `next_url`.
_LIMIT = 50_000

#: Stop following `next_url` after this many pages. A pathological request — a
#: decade of one-second bars — would otherwise page for hours while looking
#: like a hang.
_MAX_PAGES = 40


@dataclass(slots=True)
class MassiveSeriesInfo:
    """What the last fetch actually did, beyond the bars themselves."""

    host: str = ""
    endpoint: str = ""
    ticker_requested: str = ""
    ticker_resolved: str = ""
    contract: dict[str, Any] = field(default_factory=dict)
    pages: int = 0
    truncated: bool = False

    def render(self) -> str:
        lines = [
            f"host      : {self.host}",
            f"endpoint  : {self.endpoint}",
            f"ticker    : {self.ticker_requested}"
            + (
                f" → {self.ticker_resolved}"
                if self.ticker_resolved != self.ticker_requested
                else ""
            ),
            f"pages     : {self.pages}",
        ]
        if self.contract:
            lines.append(
                f"contract  : {self.contract.get('name') or self.ticker_resolved}, "
                f"last trade {self.contract.get('last_trade_date', '?')}"
            )
            lines.append(
                "            One contract, not a stitched series. The gap at a "
                "roll is not a return, so this adapter will not manufacture one."
            )
        if self.truncated:
            lines.append(
                f"⚠ truncated after {_MAX_PAGES} pages — the window is larger "
                f"than one request can return. Narrow it or use a coarser "
                f"timeframe; the bars you have are real but incomplete."
            )
        return "\n".join(lines)


class MassiveProvider(BaseProvider):
    """Equity, index and futures bars from Massive (formerly Polygon.io).

    Args:
        api_key: falls back to ``MASSIVE_API_KEY`` then ``POLYGON_API_KEY``.
        host: defaults to ``api.massive.com``; pass :data:`POLYGON_HOST` to use
            the pre-rebrand endpoint.
        adjusted: split-adjust equity bars. On by default because any
            multi-year price comparison is meaningless without it.
    """

    name = "massive"
    #: Options and forex are also sold, but this adapter does not translate
    #: their symbologies (`O:SPY251219C00650000`, `C:EURUSD`), so claiming them
    #: would route requests here that cannot be served.
    asset_classes = frozenset(
        {AssetClass.EQUITY, AssetClass.FUTURES, AssetClass.INDEX}
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        host: str = DEFAULT_HOST,
        adjusted: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or get_credential("MASSIVE_API_KEY", "POLYGON_API_KEY")
        self.host = host.rstrip("/")
        self.adjusted = adjusted
        self.timeout = timeout
        self.last = MassiveSeriesInfo()

    def is_available(self) -> bool:
        return bool(self.api_key)

    # --- plumbing -----------------------------------------------------------

    def _get(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise ProviderUnavailableError(
                "no Massive API key. Put it in a .env file at the project root:\n"
                "    MASSIVE_API_KEY=...\n"
                "Get one at massive.com → Dashboard → API Keys. A Polygon.io key "
                "works — POLYGON_API_KEY is read too."
            )

        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self.host}{path_or_url}"
        )
        if params:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urllib.parse.urlencode(params)}"

        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ProviderError(f"{self.name}: {_explain(exc.code, detail)}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"{self.name}: could not reach {self.host}: {exc.reason}. A 403 "
                f"tunnel failure means the host is blocked by a network policy, "
                f"not by your key — allow api.massive.com."
            ) from exc

    def _paginate(self, path: str, params: dict[str, Any]) -> tuple[list[Any], bool]:
        """Follow ``next_url`` until the window is exhausted or the cap is hit.

        Returns ``(rows, truncated)``. Truncation is returned rather than
        raised: partial real bars are still useful, but a caller who does not
        know the window was cut short will read a shorter dataset as a quieter
        market.
        """
        rows: list[Any] = []
        url: str | None = path
        query: dict[str, Any] | None = params
        pages = 0

        while url and pages < _MAX_PAGES:
            payload = self._get(url, query)
            rows.extend(payload.get("results") or [])
            pages += 1
            # next_url already carries the query, and re-appending the original
            # params would override its cursor and loop on page one forever.
            url, query = payload.get("next_url"), None

        self.last.pages = pages
        return rows, bool(url)

    # --- futures symbology --------------------------------------------------

    def front_contract(self, product_code: str) -> dict[str, Any]:
        """The nearest active contract for a futures root, e.g. ``ES`` → ``ESH6``.

        Chosen by nearest ``last_trade_date`` among active contracts — a
        **calendar** roll, which is the only rule this endpoint supports without
        a second round of volume queries. It is the crudest of the standard
        rules: it can hold a contract that liquidity has already left, days
        before expiry. Named here rather than buried so that a thin-looking
        front month is recognisable as the rule working, not the market.
        """
        payload = self._get(
            "/futures/v1/contracts",
            {"product_code": product_code.upper(), "active": "true",
             "limit": 100, "sort": "last_trade_date"},
        )
        contracts = [c for c in (payload.get("results") or []) if c.get("ticker")]
        if not contracts:
            raise ProviderError(
                f"{self.name}: no active contracts for product code "
                f"{product_code.upper()!r}. Massive addresses futures by "
                f"contract (ESH6), not by root — check the root is right, or "
                f"pass the contract ticker directly."
            )

        today = datetime.now(UTC).date().isoformat()
        live = [c for c in contracts if str(c.get("last_trade_date", "")) >= today]
        return min(
            live or contracts,
            key=lambda c: str(c.get("last_trade_date") or "9999-12-31"),
        )

    def resolve(self, request: BarRequest) -> tuple[str, dict[str, Any]]:
        """Return ``(ticker, contract)`` for this instrument."""
        symbol = request.instrument.symbol.upper()
        if request.instrument.asset_class is not AssetClass.FUTURES:
            if request.instrument.asset_class is AssetClass.INDEX:
                return (symbol if symbol.startswith("I:") else f"I:{symbol}"), {}
            return symbol, {}

        # A root is 1-3 letters with no month code; anything longer already
        # names a contract and is passed through.
        if len(symbol) <= 3 and symbol.isalpha():
            contract = self.front_contract(symbol)
            return str(contract["ticker"]), contract
        return symbol, {}

    # --- fetch --------------------------------------------------------------

    def _provenance(self, request: BarRequest) -> Provenance:
        detail = f"{self.last.ticker_resolved}@{request.timeframe} {self.last.endpoint}"
        if self.last.truncated:
            detail += " TRUNCATED"
        return Provenance.real(self.name, detail=detail)

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        ticker, contract = self.resolve(request)
        is_futures = request.instrument.asset_class is AssetClass.FUTURES

        self.last = MassiveSeriesInfo(
            host=self.host,
            endpoint="futures/v1/aggs" if is_futures else "v2/aggs",
            ticker_requested=request.instrument.symbol.upper(),
            ticker_resolved=ticker,
            contract=contract,
        )

        rows, truncated = (
            self._fetch_futures(request, ticker)
            if is_futures
            else self._fetch_equity(request, ticker)
        )
        self.last.truncated = truncated

        if not rows:
            raise ProviderError(
                f"{self.name}: no bars for {ticker} {request.timeframe} between "
                f"{request.start.date()} and {request.end.date()}. "
                + (
                    "A futures contract returns nothing outside its own trading "
                    "window — check the contract was listed over this range."
                    if is_futures
                    else "Thinly-traded symbols and very recent windows can "
                    "legitimately be empty."
                )
            )

        frame = (
            _futures_frame(rows) if is_futures else _equity_frame(rows)
        )
        return frame.loc[
            (frame.index >= request.start) & (frame.index <= request.end)
        ]

    def _fetch_equity(
        self, request: BarRequest, ticker: str
    ) -> tuple[list[Any], bool]:
        span = _EQUITY_SPANS.get(request.timeframe.label)
        if span is None:
            raise ProviderError(
                f"{self.name}: unsupported timeframe {request.timeframe} for "
                f"equities; supported: {sorted(_EQUITY_SPANS)}"
            )
        multiplier, timespan = span
        path = (
            f"/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/"
            f"{multiplier}/{timespan}/"
            f"{request.start.astimezone(UTC):%Y-%m-%d}/"
            f"{request.end.astimezone(UTC):%Y-%m-%d}"
        )
        return self._paginate(
            path,
            {
                "adjusted": "true" if self.adjusted else "false",
                "sort": "asc",
                "limit": _LIMIT,
            },
        )

    def _fetch_futures(
        self, request: BarRequest, ticker: str
    ) -> tuple[list[Any], bool]:
        resolution = _FUTURES_RESOLUTIONS.get(request.timeframe.label)
        if resolution is None:
            raise ProviderError(
                f"{self.name}: unsupported timeframe {request.timeframe} for "
                f"futures; supported: {sorted(_FUTURES_RESOLUTIONS)}"
            )
        return self._paginate(
            f"/futures/v1/aggs/{urllib.parse.quote(ticker)}",
            {
                "resolution": resolution,
                "window_start.gte": f"{request.start.astimezone(UTC):%Y-%m-%d}",
                "window_start.lte": f"{request.end.astimezone(UTC):%Y-%m-%d}",
                "limit": _LIMIT,
                "sort": "window_start.asc",
            },
        )

    # --- diagnostics --------------------------------------------------------

    def check_credentials(self) -> str:
        """Verify the key and report what it reaches."""
        if not self.api_key:
            return (
                "✗ No Massive API key.\n"
                "  Add it to the .env file at the project root:\n"
                "    MASSIVE_API_KEY=...\n"
                "  Get one at massive.com → Dashboard → API Keys.\n"
                "  A Polygon.io key works — POLYGON_API_KEY is read too."
            )
        try:
            payload = self._get(
                "/v2/aggs/ticker/AAPL/prev", {"adjusted": "true"}
            )
        except ProviderError as exc:
            return f"✗ {exc}"

        equities = bool(payload.get("results"))
        try:
            self.front_contract("ES")
            futures = "yes"
        except ProviderError as exc:
            # Futures is a separate entitlement, so its absence is a plan
            # question and not a broken key — the distinction decides whether
            # someone goes looking at their credentials or their subscription.
            futures = f"no — {str(exc)[:120]}"

        return (
            f"✓ Massive key valid ({self.host})\n"
            f"  equities : {'yes' if equities else 'reachable, no data returned'}\n"
            f"  futures  : {futures}"
        )


def _equity_frame(rows: list[Any]) -> pd.DataFrame:
    """Build a frame from the v2 aggregates schema (single-letter keys)."""
    frame = pd.DataFrame(rows)
    missing = {"t", "o", "h", "l", "c", "v"} - set(frame.columns)
    if missing:
        raise ProviderError(
            f"massive: equity response is missing {sorted(missing)} — got "
            f"{sorted(frame.columns)[:12]}"
        )
    frame = frame.rename(
        columns={"t": "timestamp", "o": "open", "h": "high",
                 "l": "low", "c": "close", "v": "volume"}
    )
    frame = frame.set_index(_to_utc(frame["timestamp"]))
    return frame[["open", "high", "low", "close", "volume"]].sort_index()


def _futures_frame(rows: list[Any]) -> pd.DataFrame:
    """Build a frame from the futures schema (spelled-out keys).

    A separate function rather than a column-map parameter, because the two
    schemas differ in the timestamp's *units* as well as its name, and folding
    that into a rename table is how the units get forgotten.
    """
    frame = pd.DataFrame(rows)
    missing = {"window_start", "open", "high", "low", "close", "volume"} - set(
        frame.columns
    )
    if missing:
        raise ProviderError(
            f"massive: futures response is missing {sorted(missing)} — got "
            f"{sorted(frame.columns)[:12]}. Note the futures schema spells its "
            f"fields out; the single-letter equity keys do not appear here."
        )
    frame = frame.set_index(_to_utc(frame["window_start"]))
    return frame[["open", "high", "low", "close", "volume"]].sort_index()


def _to_utc(values: pd.Series) -> pd.DatetimeIndex:
    """Convert epoch integers to UTC, detecting the unit rather than assuming it.

    Massive's equity aggregates timestamp in milliseconds and its futures REST
    parameters are documented in nanoseconds, with the response unit not stated
    either way. Guessing has a specific cost: reading nanoseconds as
    milliseconds puts every bar around the year 57,000, and reading
    milliseconds as nanoseconds puts them all in January 1970.

    Detection is safe because the two are six orders of magnitude apart. Any
    timestamp this century is ~1.7e12 ms, ~1.7e15 µs or ~1.7e18 ns — there is
    no value a sane date could take that is ambiguous between them.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().all():
        raise ProviderError("massive: no parseable timestamps in the response")

    magnitude = float(numeric.dropna().abs().median())
    if magnitude < 1e11:
        unit = "s"
    elif magnitude < 1e14:
        unit = "ms"
    elif magnitude < 1e17:
        unit = "us"
    else:
        unit = "ns"
    return pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit, utc=True))


def _explain(code: int, detail: str) -> str:
    """Name the likely cause of an HTTP failure rather than restating the code."""
    if code == 401:
        return f"HTTP 401 — check MASSIVE_API_KEY. {detail}"
    if code == 403:
        return (
            f"HTTP 403 — this key is valid but not entitled to that data. "
            f"Futures, options and second-resolution bars are separate plans "
            f"from equities. {detail}"
        )
    if code == 429:
        return (
            f"HTTP 429 — rate limited. The free tier allows 5 requests per "
            f"minute, and one backtest window can be several pages. Narrow the "
            f"window, use a coarser timeframe, or upgrade the plan. {detail}"
        )
    if code == 404:
        return (
            f"HTTP 404 — the ticker did not resolve. Futures need a contract "
            f"ticker (ESH6), indices need an 'I:' prefix. {detail}"
        )
    return f"HTTP {code}: {detail}"
