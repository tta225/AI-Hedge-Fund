"""Alpaca market data adapter.

Alpaca gives free, keyed access to US equity and crypto bars, which makes it the
shortest path from this platform to **real market data** — the one thing the
whole engine is still missing.

Credentials
-----------
Set both of::

    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

(``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` are also accepted.) Generate them at
https://app.alpaca.markets — the **paper** dashboard is enough for market data.

Feeds
-----
``iex`` is the free feed: a single exchange, so volume is a fraction of the
consolidated tape and the bars are thinner than what actually traded. ``sip`` is
the full consolidated feed and requires a paid subscription.

This matters more than it sounds for ICT work. Liquidity pools, equal highs, and
sweeps are all statements about *where price traded*, and IEX-only bars can miss
prints that set the extreme. Research on the free feed is fine; treat any level
derived from it as approximate.

This adapter deliberately implements **market data only**. Order routing to
Alpaca would be a live :class:`~axiom.execution.base.ExecutionVenue`, and the
router refuses to reach one outside live mode — that boundary is the point, so
it is left for a deliberate, separate step.
"""

from __future__ import annotations

import os
from datetime import UTC

import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.types import AssetClass
from axiom.data.base import BarRequest, BaseProvider, ProviderError, ProviderUnavailableError

#: Our timeframe labels mapped onto Alpaca's TimeFrame strings.
_ALPACA_TIMEFRAMES: dict[str, str] = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week",
}

_DATA_URL = "https://data.alpaca.markets"


class AlpacaProvider(BaseProvider):
    """US equity and crypto bars from Alpaca.

    Uses the REST endpoint directly rather than ``alpaca-py`` so the core
    platform needs no extra dependency. Install the extra only if you want the
    SDK for other purposes.
    """

    name = "alpaca"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: str = "iex",
        adjustment: str = "split",
    ) -> None:
        """
        Args:
            feed: ``iex`` (free, single-exchange) or ``sip`` (paid, consolidated).
            adjustment: corporate-action handling — ``raw``, ``split``,
                ``dividend``, or ``all``. ``split`` is the sane default: split
                adjustment is required for any multi-year price comparison to
                mean anything, while dividend adjustment distorts the actual
                traded levels that ICT analysis depends on.
        """
        self.api_key = (
            api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        )
        self.secret_key = (
            secret_key
            or os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        self.feed = feed
        self.adjustment = adjustment

    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "accept": "application/json",
        }

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(
            self.name,
            detail=f"{request.instrument.symbol}@{request.timeframe} feed={self.feed}",
        )

    def _endpoint(self, request: BarRequest) -> str:
        if request.instrument.asset_class is AssetClass.CRYPTO:
            return f"{_DATA_URL}/v1beta3/crypto/us/bars"
        return f"{_DATA_URL}/v2/stocks/bars"

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        if not self.is_available():
            raise ProviderUnavailableError(
                "Alpaca credentials not found. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY (get them at app.alpaca.markets)."
            )

        timeframe = _ALPACA_TIMEFRAMES.get(request.timeframe.label)
        if timeframe is None:
            raise ProviderError(
                f"{self.name}: unsupported timeframe {request.timeframe}; "
                f"supported: {sorted(_ALPACA_TIMEFRAMES)}"
            )

        symbol = request.instrument.symbol
        is_crypto = request.instrument.asset_class is AssetClass.CRYPTO
        params: dict[str, str] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": request.start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": request.end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "limit": "10000",
        }
        if not is_crypto:
            params["feed"] = self.feed
            params["adjustment"] = self.adjustment

        rows = self._paginate(self._endpoint(request), params, symbol)
        if not rows:
            raise ProviderError(
                f"{self.name}: no bars for {symbol} {request.timeframe} between "
                f"{request.start.date()} and {request.end.date()}. On the free "
                f"'iex' feed, thinly-traded symbols and very recent windows can "
                f"legitimately be empty."
            )

        frame = pd.DataFrame(rows)
        frame = frame.rename(
            columns={"t": "timestamp", "o": "open", "h": "high",
                     "l": "low", "c": "close", "v": "volume"}
        )
        frame = frame.set_index(pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601"))
        return frame[["open", "high", "low", "close", "volume"]].sort_index()

    def _paginate(
        self, url: str, params: dict[str, str], symbol: str
    ) -> list[dict[str, object]]:
        """Follow Alpaca's ``next_page_token`` until the window is exhausted.

        Alpaca caps a response at 10,000 bars. A year of 1-minute data is far
        more than that, so a single request silently returns a truncated window
        — which would look like a shorter dataset rather than an error.
        """
        import json
        import urllib.error
        import urllib.parse
        import urllib.request

        collected: list[dict[str, object]] = []
        token: str | None = None

        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            request = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(query)}", headers=self._headers
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                hint = ""
                if exc.code == 401:
                    hint = " — check APCA_API_KEY_ID / APCA_API_SECRET_KEY"
                elif exc.code == 403:
                    hint = (
                        f" — the '{self.feed}' feed may not be included in your "
                        "subscription; 'iex' is the free one"
                    )
                elif exc.code == 429:
                    hint = " — rate limited; retry with a narrower window"
                raise ProviderError(
                    f"{self.name}: HTTP {exc.code}{hint}. {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                raise ProviderError(f"{self.name}: network error: {exc.reason}") from exc

            bars = payload.get("bars") or {}
            batch = bars.get(symbol, []) if isinstance(bars, dict) else bars
            collected.extend(batch)

            token = payload.get("next_page_token")
            if not token:
                return collected

    def check_credentials(self) -> str:
        """Verify the keys work and report what they unlock.

        Run this first — it turns an opaque 401 deep inside a backtest into a
        clear message before any work is wasted.
        """
        if not self.is_available():
            return (
                "✗ No Alpaca credentials found.\n"
                "  Set APCA_API_KEY_ID and APCA_API_SECRET_KEY, then start a new\n"
                "  session — environment variables are injected at container start."
            )
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/account", headers=self._headers
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                account = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return f"✗ Alpaca rejected the credentials: HTTP {exc.code}"
        except urllib.error.URLError as exc:
            return f"✗ Could not reach Alpaca: {exc.reason}"

        return (
            f"✓ Alpaca credentials valid\n"
            f"  account status : {account.get('status')}\n"
            f"  buying power   : ${float(account.get('buying_power', 0)):,.2f}\n"
            f"  data feed      : {self.feed}"
            + ("  (free, single-exchange)" if self.feed == "iex" else "")
        )
