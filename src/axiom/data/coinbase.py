"""Coinbase Exchange market data adapter — real bars, no credentials.

Why this exists
---------------
Every other free source this platform can reach is gated. Alpaca needs a key,
Yahoo rate-limits datacenter addresses, and Stooq serves a bot challenge. That
left the engine with **no real market data at all**, and a backtest on generated
bars is a correctness check, never evidence.

Coinbase Exchange publishes historical candles on an open, documented, keyless
endpoint. It is the shortest honest path from this repository to bars that
actually traded.

What this is and is not
-----------------------
This is **crypto**, not the futures/FX/equities this platform targets. It
validates the *engine* — that swing detection, fair value gaps, order blocks,
liquidity sweeps and structure shifts behave sanely against real microstructure
rather than a generator that was written to contain them.

It does **not** transfer as evidence about ES or SPY:

* BTC trades 24/7, so killzones, the daily open, and session-relative concepts
  (judas swings, Power of Three) mean much less than they do on an instrument
  with a real cash open and close.
* Crypto volatility, tick behaviour and liquidity structure are not equity
  index behaviour. A base rate measured here is a base rate *for BTC*.

Say that plainly wherever numbers from this source are reported. The provenance
stamp records the venue, so a report can always name what it was measured on.

API shape
---------
``GET /products/{id}/candles?granularity=&start=&end=`` returns at most **300**
candles, newest first, as ``[time, low, high, open, close, volume]``. That
column order is unusual and getting it wrong silently produces bars whose highs
and lows are swapped — which would corrupt every ICT level downstream — so it is
mapped explicitly and validated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, timedelta

import pandas as pd

from axiom.core.provenance import Provenance
from axiom.core.types import AssetClass
from axiom.data.base import BarRequest, BaseProvider, ProviderError

_API_URL = "https://api.exchange.coinbase.com"

#: Our timeframe labels mapped onto Coinbase's granularity, in seconds. These
#: are the only values the endpoint accepts; anything else returns HTTP 400.
_GRANULARITIES: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}

#: Hard cap on candles per response, set by Coinbase.
_MAX_CANDLES = 300

#: Public endpoints are rate limited (~10 requests/second sustained). Deep
#: history needs many requests, so pace them rather than absorbing 429s.
_REQUEST_SPACING_S = 0.15


class CoinbaseProvider(BaseProvider):
    """Historical crypto candles from Coinbase Exchange. No credentials.

    Args:
        product_map: optional symbol → Coinbase product id overrides. Symbols
            are otherwise resolved by the rules in :meth:`product_id`.
        max_requests: safety cap on pagination. A year of 1-minute candles is
            ~1,750 requests; the cap turns that into a clear error rather than
            a silent twenty-minute stall.
    """

    name = "coinbase"
    asset_classes = frozenset({AssetClass.CRYPTO})

    def __init__(
        self,
        *,
        product_map: dict[str, str] | None = None,
        max_requests: int = 400,
    ) -> None:
        self.product_map = dict(product_map or {})
        self.max_requests = max_requests

    def is_available(self) -> bool:
        # No credentials and no optional dependency — always usable. Network
        # failure surfaces as a ProviderError at fetch time, which is the
        # honest place for it: "reachable" is not knowable without a request.
        return True

    def check_reachable(self) -> str:
        """Probe the public endpoint and report whether bars can be fetched.

        ``is_available`` is true unconditionally because this provider needs no
        credentials, so it says nothing about whether the endpoint can actually
        be reached. On a network with egress restrictions those are opposite
        answers, and the difference is invisible until a fetch fails. One cheap
        request settles it.
        """
        request = urllib.request.Request(
            f"{_API_URL}/products/BTC-USD", headers={"User-Agent": "axiom"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return f"✗ Coinbase returned HTTP {exc.code}"
        except urllib.error.URLError as exc:
            return f"✗ Could not reach Coinbase: {exc.reason}"
        except (TimeoutError, OSError) as exc:  # pragma: no cover - network shape
            return f"✗ Could not reach Coinbase: {exc}"

        return "✓ Coinbase reachable (public candles, no credentials)"

    def product_id(self, symbol: str) -> str:
        """Resolve a symbol to a Coinbase product id.

        ``BTC-USD`` passes through, ``BTC`` and ``BTCUSD`` become ``BTC-USD``.
        """
        upper = symbol.upper()
        if upper in self.product_map:
            return self.product_map[upper]
        if "-" in upper:
            return upper
        for quote in ("USD", "USDT", "USDC", "EUR", "GBP"):
            if upper.endswith(quote) and len(upper) > len(quote):
                return f"{upper[: -len(quote)]}-{quote}"
        return f"{upper}-USD"

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance.real(
            self.name,
            detail=(
                f"{self.product_id(request.instrument.symbol)}"
                f"@{request.timeframe} venue=coinbase-exchange"
            ),
        )

    def _get(self, url: str) -> list[list[float]]:
        req = urllib.request.Request(
            url, headers={"accept": "application/json", "User-Agent": "axiom/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            hint = ""
            if exc.code == 404:
                hint = " — unknown product id; try e.g. BTC-USD"
            elif exc.code == 429:
                hint = " — rate limited; narrow the window or retry"
            raise ProviderError(f"{self.name}: HTTP {exc.code}{hint}. {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"{self.name}: network error — {exc}") from exc
        if not isinstance(payload, list):
            raise ProviderError(f"{self.name}: unexpected response shape {type(payload)}")
        return payload

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        # Declared up front so a registry lookup for SPY falls through to the
        # next provider immediately, instead of spending a round trip to
        # discover that "SPY-USD" is not a product.
        if request.instrument.asset_class is not AssetClass.CRYPTO:
            raise ProviderError(
                f"{self.name}: serves crypto only; "
                f"{request.instrument.symbol} is {request.instrument.asset_class.value}"
            )

        label = request.timeframe.label
        granularity = _GRANULARITIES.get(label)
        if granularity is None:
            raise ProviderError(
                f"{self.name}: unsupported timeframe {label}; Coinbase serves "
                f"only {sorted(_GRANULARITIES)}"
            )

        product = self.product_id(request.instrument.symbol)
        step = timedelta(seconds=granularity * _MAX_CANDLES)

        # Walk backwards from the end: the endpoint returns newest-first and
        # caps each page, so paging forward from `start` would need the same
        # number of calls with more edge cases at the boundary.
        rows: list[list[float]] = []
        window_end = request.end.astimezone(UTC)
        start = request.start.astimezone(UTC)
        requests_made = 0

        while window_end > start and requests_made < self.max_requests:
            window_start = max(start, window_end - step)
            query = urllib.parse.urlencode(
                {
                    "granularity": granularity,
                    "start": window_start.isoformat().replace("+00:00", "Z"),
                    "end": window_end.isoformat().replace("+00:00", "Z"),
                }
            )
            page = self._get(f"{_API_URL}/products/{product}/candles?{query}")
            requests_made += 1
            if not page:
                # A genuinely empty stretch (listing gap, venue outage) is not
                # an error; keep walking back rather than stopping short.
                window_end = window_start
                continue
            rows.extend(page)
            window_end = window_start
            if window_end > start:
                time.sleep(_REQUEST_SPACING_S)

        if requests_made >= self.max_requests and window_end > start:
            raise ProviderError(
                f"{self.name}: window needs more than {self.max_requests} requests "
                f"at {label}. Request a shorter span or a coarser timeframe — a "
                f"year of 1m candles is ~1,750 pages of 300."
            )
        if not rows:
            raise ProviderError(
                f"{self.name}: no candles for {product} {label} between "
                f"{request.start.date()} and {request.end.date()}"
            )

        # [time, low, high, open, close, volume] — not the conventional order.
        frame = pd.DataFrame(
            rows, columns=["time", "low", "high", "open", "close", "volume"]
        )
        frame = frame.set_index(pd.to_datetime(frame["time"], unit="s", utc=True))
        frame = frame[["open", "high", "low", "close", "volume"]]
        # Pages overlap at their boundaries; duplicates would double-count
        # volume and can create phantom equal highs, which ICT reads as a
        # liquidity pool that was never there.
        frame = frame[~frame.index.duplicated(keep="first")].sort_index()
        return frame
