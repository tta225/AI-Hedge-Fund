"""Databento market data adapter — futures, and everything else CME carries.

Databento is the first provider here that can supply **futures**. The Hugging
Face bucket holds equities and ETFs only; Alpaca holds equities and crypto. So
until this adapter exists, asking this platform for ES gets you Eversource
Energy, a utility trading around $71, rather than the E-mini S&P around 5,000 —
and nothing errors, because both are real prices for a real ticker.

Three decisions in here are worth reading before using it.

**It uses the official SDK rather than raw HTTP.** Every other adapter in this
package calls the vendor's REST API directly, to keep the core dependency-free.
Not this one. Databento's wire format is DBN, a binary encoding whose prices are
fixed-point int64 scaled by 1e-9 — an unscaled ES print reads as
5,000,000,000,000 rather than 5,000. That particular error is loud. The one that
would not be is a partial: a schema change that scales some fields and not
others, silently producing a price series that is wrong by a factor of a billion
in one column. ``DBNStore.to_df(price_type="float")`` is the vendor's own answer
to that, and reimplementing it against documentation this environment cannot
reach is how the Hugging Face bucket got misdiagnosed twice.

**Continuous contracts are unadjusted, and this adapter says so.** A futures
contract expires, so any multi-year series is several contracts stitched
together. At each stitch the price jumps — the December contract does not trade
where the September one did — and that jump is **not a return**. A backtest that
treats it as one is measuring the roll. Databento does not offer back-adjusted
continuous contracts, so the jumps are present in what comes back, and this
adapter detects each roll and reports it rather than smoothing it away. See
:attr:`DatabentoSeriesInfo.rolls`.

**It costs money per request.** Databento bills by bytes delivered, with no
monthly minimum, so a careless request is a bill rather than a rate-limit error.
:meth:`DatabentoProvider.estimate_cost` asks what a request would cost before
making it, and ``max_cost_usd`` refuses anything above a ceiling.

Credentials
-----------
::

    DATABENTO_API_KEY=db-...

32 characters, starting ``db-``, from the API Keys page of the user portal.

Install
-------
::

    python3 -m pip install -e ".[databento]"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
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

__all__ = ["DEFAULT_DATASET", "DatabentoProvider", "DatabentoSeriesInfo", "RollEvent"]

#: CME Globex — ES, NQ, CL, GC, ZB, and every other CME/CBOT/NYMEX/COMEX future.
DEFAULT_DATASET = "GLBX.MDP3"

#: Dataset to use when the instrument is not a future.
_EQUITY_DATASET = "XNAS.ITCH"

#: Our timeframe labels mapped onto Databento schemas. Databento aggregates at
#: second, minute, hour and day only — anything else is resampled from a finer
#: schema, which costs more bytes, so the mapping is explicit about the source.
_SCHEMAS: dict[str, str] = {
    "1s": "ohlcv-1s",
    "1m": "ohlcv-1m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}

#: Timeframes Databento has no native schema for, and the schema to build them
#: from. 15m from 1m is 15× the bytes of a native 15m schema that does not
#: exist — worth knowing before requesting ten years of it.
_RESAMPLE_FROM: dict[str, str] = {
    "5m": "1m", "15m": "1m", "30m": "1m", "4h": "1h", "1w": "1d",
}

#: Roll rules, in Databento's continuous-contract symbology.
ROLL_RULES: dict[str, str] = {
    "calendar": "c",   # rolls on expiry — predictable, but can hold a dead month
    "open_interest": "n",
    "volume": "v",
}


@dataclass(frozen=True, slots=True)
class RollEvent:
    """A contract change inside a continuous series.

    ``gap`` is the price discontinuity at the stitch. It is **not** a return —
    nothing was gained or lost as the series moved from one contract to the
    next — which is exactly why it is recorded rather than left to be inferred
    from a suspicious-looking bar.
    """

    timestamp: pd.Timestamp
    from_contract: str
    to_contract: str
    gap: float

    @property
    def gap_pct(self) -> float:
        return self.gap

    def __str__(self) -> str:
        return (
            f"{self.timestamp:%Y-%m-%d}  {self.from_contract} → {self.to_contract}  "
            f"gap {self.gap:+.2f}"
        )


@dataclass(slots=True)
class DatabentoSeriesInfo:
    """What the last fetch actually did, beyond the bars themselves."""

    dataset: str = ""
    schema: str = ""
    symbol_requested: str = ""
    contracts: list[str] = field(default_factory=list)
    rolls: list[RollEvent] = field(default_factory=list)
    resampled_from: str = ""
    cost_usd: float | None = None

    def render(self) -> str:
        lines = [
            f"dataset   : {self.dataset}",
            f"schema    : {self.schema}"
            + (f"  (resampled from {self.resampled_from})" if self.resampled_from else ""),
            f"symbol    : {self.symbol_requested}",
            f"contracts : {len(self.contracts)}  {', '.join(self.contracts[:6])}"
            + (" …" if len(self.contracts) > 6 else ""),
        ]
        if self.cost_usd is not None:
            lines.append(f"cost      : ${self.cost_usd:,.4f}")
        if self.rolls:
            lines.append(
                f"rolls     : {len(self.rolls)} — the price gap at each is NOT a "
                f"return. A backtest that treats it as one is measuring the roll."
            )
            lines += [f"            {r}" for r in self.rolls]
        return "\n".join(lines)


class DatabentoProvider(BaseProvider):
    """Futures and equity bars from Databento.

    Args:
        api_key: falls back to ``DATABENTO_API_KEY``.
        dataset: overrides the per-asset-class default.
        roll_rule: ``calendar``, ``open_interest`` or ``volume``. Only used
            when a futures symbol is resolved as a continuous contract.
        continuous: resolve bare futures roots (``ES``) to continuous contracts
            (``ES.c.0``). Set False to pass explicit contract codes through.
        max_cost_usd: refuse any request Databento prices above this. None
            disables the check — which is a choice, not a default.
    """

    name = "databento"
    #: The only provider here that carries futures. Equities are served
    #: too, from a different dataset.
    asset_classes = frozenset({AssetClass.FUTURES, AssetClass.EQUITY})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        dataset: str = "",
        roll_rule: str = "volume",
        continuous: bool = True,
        max_cost_usd: float | None = 5.0,
    ) -> None:
        self.api_key = api_key or get_credential("DATABENTO_API_KEY")
        self.dataset = dataset
        if roll_rule not in ROLL_RULES:
            raise ValueError(
                f"unknown roll_rule {roll_rule!r}; choose from {sorted(ROLL_RULES)}"
            )
        self.roll_rule = roll_rule
        self.continuous = continuous
        self.max_cost_usd = max_cost_usd
        self.last: DatabentoSeriesInfo = DatabentoSeriesInfo()

    # --- availability -------------------------------------------------------

    @staticmethod
    def _sdk() -> Any:
        try:
            import databento
        except ImportError as exc:  # pragma: no cover - exercised by the message
            raise ProviderUnavailableError(
                "the databento package is not installed. Run:\n"
                '    python3 -m pip install -e ".[databento]"'
            ) from exc
        return databento

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            self._sdk()
        except ProviderUnavailableError:
            return False
        return True

    def _client(self) -> Any:
        if not self.api_key:
            raise ProviderUnavailableError(
                "no Databento API key. Put it in a .env file at the project root:\n"
                "    DATABENTO_API_KEY=db-...\n"
                "Get one at databento.com → Portal → API Keys."
            )
        return self._sdk().Historical(self.api_key)

    # --- request translation ------------------------------------------------

    def dataset_for(self, request: BarRequest) -> str:
        if self.dataset:
            return self.dataset
        if request.instrument.asset_class is AssetClass.FUTURES:
            return DEFAULT_DATASET
        return _EQUITY_DATASET

    def symbol_for(self, request: BarRequest) -> tuple[str, str]:
        """Return ``(symbol, stype_in)`` for this instrument.

        A bare futures root is ambiguous on purpose: ``ES`` names a family of
        contracts, not one of them. Resolving it to ``ES.v.0`` picks the most
        actively traded month and rolls automatically, which is what a
        multi-year backtest needs. An explicit contract like ``ESH6`` is passed
        through untouched — someone who named a month meant that month.
        """
        symbol = request.instrument.symbol.upper()
        if request.instrument.asset_class is not AssetClass.FUTURES:
            return symbol, "raw_symbol"
        if not self.continuous or any(ch.isdigit() for ch in symbol) or "." in symbol:
            return symbol, "raw_symbol"
        return f"{symbol}.{ROLL_RULES[self.roll_rule]}.0", "continuous"

    def schema_for(self, request: BarRequest) -> tuple[str, str]:
        """Return ``(schema, resample_from_label)``."""
        label = request.timeframe.label
        if label in _SCHEMAS:
            return _SCHEMAS[label], ""
        source = _RESAMPLE_FROM.get(label)
        if source is None:
            raise ProviderError(
                f"{self.name}: no route to {label} bars. Native schemas: "
                f"{sorted(_SCHEMAS)}; resampled: {sorted(_RESAMPLE_FROM)}."
            )
        return _SCHEMAS[source], source

    # --- cost ---------------------------------------------------------------

    def estimate_cost(self, request: BarRequest) -> float:
        """What Databento would charge for this request, in USD, before making it.

        Worth calling directly. Databento bills by bytes delivered rather than
        by subscription, so the feedback for asking for too much is an invoice,
        not a 429 — and the gap between "one year of 1-minute bars" and "one
        year of tick data" is about four orders of magnitude.
        """
        symbol, stype_in = self.symbol_for(request)
        schema, _ = self.schema_for(request)
        return float(
            self._client().metadata.get_cost(
                dataset=self.dataset_for(request),
                start=request.start.astimezone(UTC),
                end=request.end.astimezone(UTC),
                symbols=symbol,
                schema=schema,
                stype_in=stype_in,
            )
        )

    # --- diagnostics --------------------------------------------------------

    def check_credentials(self) -> str:
        """Verify the key and report what it can reach, without buying anything.

        Deliberately calls `metadata.list_datasets` rather than fetching a bar.
        Every other provider here can probe itself for free; Databento bills by
        bytes, so a check that pulled one minute of ES would be a check that
        costs money every time someone runs `data-check`. Metadata is free.
        """
        if not self.api_key:
            return (
                "✗ No Databento API key.\n"
                "  Add it to the .env file at the project root:\n"
                "    DATABENTO_API_KEY=db-...\n"
                "  Get one at databento.com → Portal → API Keys."
            )
        if not self.api_key.startswith("db-"):
            return (
                f"✗ DATABENTO_API_KEY does not look like a Databento key "
                f"(got {len(self.api_key)} chars, expected 32 starting 'db-'). "
                f"This is worth checking before blaming the network."
            )
        try:
            self._sdk()
        except ProviderUnavailableError as exc:
            return f"✗ {exc}"

        try:
            datasets = self._client().metadata.list_datasets()
        except Exception as exc:  # the vendor exception surface is wide
            return f"✗ {_explain(exc)}"

        names = sorted(str(d) for d in datasets)
        has_futures = DEFAULT_DATASET in names
        return (
            f"✓ Databento key valid — {len(names)} dataset(s) entitled\n"
            f"  futures ({DEFAULT_DATASET}): "
            + ("yes" if has_futures else "NOT entitled — futures will not work")
            + "\n  cost ceiling  : "
            + (f"${self.max_cost_usd:,.2f} per request" if self.max_cost_usd is not None
               else "none — requests are not priced before they are billed")
        )

    # --- fetch --------------------------------------------------------------

    def _provenance(self, request: BarRequest) -> Provenance:
        detail = f"{self.last.symbol_requested}@{request.timeframe} {self.last.dataset}"
        if self.last.rolls:
            detail += f" rolls={len(self.last.rolls)}"
        return Provenance.real(self.name, detail=detail)

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        client = self._client()
        symbol, stype_in = self.symbol_for(request)
        schema, resample_from = self.schema_for(request)
        dataset = self.dataset_for(request)

        info = DatabentoSeriesInfo(
            dataset=dataset,
            schema=schema,
            symbol_requested=symbol,
            resampled_from=resample_from,
        )

        if self.max_cost_usd is not None:
            try:
                info.cost_usd = self.estimate_cost(request)
            except Exception as exc:
                # Stop rather than proceed. A caller who set a ceiling asked not
                # to be billed above it, and fetching anyway because the *pricer*
                # failed would honour the letter of the request while breaking
                # the only part of it that costs money. Overriding is available,
                # but it has to be chosen.
                info.cost_usd = None
                self.last = info
                raise ProviderError(
                    f"{self.name}: could not price this request before making it "
                    f"({exc}). max_cost_usd={self.max_cost_usd} cannot be enforced "
                    f"without it. Pass max_cost_usd=None to fetch anyway."
                ) from exc
            if info.cost_usd > self.max_cost_usd:
                self.last = info
                raise ProviderError(
                    f"{self.name}: this request costs ${info.cost_usd:,.2f}, over "
                    f"the ${self.max_cost_usd:,.2f} ceiling. Narrow the window, use "
                    f"a coarser timeframe, or raise max_cost_usd deliberately."
                )

        try:
            store = client.timeseries.get_range(
                dataset=dataset,
                start=request.start.astimezone(UTC),
                end=request.end.astimezone(UTC),
                symbols=symbol,
                schema=schema,
                stype_in=stype_in,
            )
            # price_type="float" is the whole reason this adapter uses the SDK:
            # DBN carries prices as int64 scaled by 1e-9, and the conversion is
            # the vendor's to get right, not ours.
            frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
        except ProviderError:
            raise
        except Exception as exc:  # the vendor exception surface is wide
            raise ProviderError(f"{self.name}: {_explain(exc)}") from exc

        if frame is None or frame.empty:
            self.last = info
            raise ProviderError(
                f"{self.name}: no bars for {symbol} {request.timeframe} between "
                f"{request.start.date()} and {request.end.date()} in {dataset}. "
                f"Check the symbol resolves — a futures root needs continuous "
                f"symbology, and {dataset} does not carry equities."
            )

        frame = self._normalise(frame)
        info.contracts = _contracts(frame)
        info.rolls = _detect_rolls(frame)

        if resample_from:
            frame = _resample(frame, request.timeframe.pandas_freq)

        self.last = info
        return frame[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        """Put a Databento frame into this platform's shape.

        `to_df` indexes by ``ts_event`` and carries the resolved contract in a
        ``symbol`` column when ``map_symbols`` is on. Both are kept: the index
        becomes the timestamp, and ``symbol`` is what makes roll detection
        possible at all.
        """
        out = frame.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            for candidate in ("ts_event", "timestamp"):
                if candidate in out.columns:
                    out = out.set_index(pd.to_datetime(out[candidate], utc=True))
                    break
        out.index = pd.to_datetime(out.index, utc=True)
        out = out.sort_index()

        missing = {"open", "high", "low", "close", "volume"} - set(out.columns)
        if missing:
            raise ProviderError(
                f"databento: response is missing {sorted(missing)} — got "
                f"{sorted(out.columns)[:12]}. This usually means the schema was "
                f"not an ohlcv one."
            )
        return out


def _contracts(frame: pd.DataFrame) -> list[str]:
    if "symbol" not in frame.columns:
        return []
    return [str(s) for s in pd.unique(frame["symbol"].dropna())]


def _detect_rolls(frame: pd.DataFrame) -> list[RollEvent]:
    """Find every contract change, and measure the price gap it introduced.

    Detected from the resolved contract name rather than from a price jump,
    because the two are different questions: a 2% overnight move is a return
    and a 2% roll gap is not, and they look identical in a price series. Only
    the symbol distinguishes them.
    """
    if "symbol" not in frame.columns or len(frame) < 2:
        return []

    symbols = frame["symbol"].astype(str)
    changed = symbols.ne(symbols.shift())
    changed.iloc[0] = False  # the first bar starts a contract, it does not roll

    rolls: list[RollEvent] = []
    closes = frame["close"].to_numpy()
    positions = [i for i, flag in enumerate(changed.to_numpy()) if flag]
    for position in positions:
        rolls.append(
            RollEvent(
                timestamp=frame.index[position],
                from_contract=str(symbols.iloc[position - 1]),
                to_contract=str(symbols.iloc[position]),
                gap=float(closes[position] - closes[position - 1]),
            )
        )
    return rolls


#: Left-closed and left-labelled, so a bar stamped 14:30 covers [14:30, 14:45).
#:
#: This must stay identical to :meth:`axiom.core.series.OHLCVSeries.resample`.
#: The convention itself is arbitrary — vendors differ — but having *two* of
#: them in one platform is not: a 15m ES bar built here and a 15m SPY bar built
#: there would be offset by a quarter of an hour while looking interchangeable,
#: and every cross-instrument comparison would quietly be comparing different
#: windows. `test_the_resample_convention_matches_the_platform` pins them
#: together.
#:
#: Note this labelling does *not* by itself prevent lookahead — a bar stamped
#: 14:30 is not complete until 14:45. That is handled by the backtester's
#: origin/confirmed index split, not by relabelling bars.
_RESAMPLE_KWARGS = {"label": "left", "closed": "left"}

_AGGREGATION = {
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}


def _resample(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate finer bars up to the requested timeframe."""
    aggregated = frame.resample(freq, **_RESAMPLE_KWARGS).agg(_AGGREGATION)  # type: ignore[arg-type]
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def _explain(exc: Exception) -> str:
    """Turn a vendor exception into something that names the likely cause."""
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "unauthor" in lowered or "authentication" in lowered:
        return f"{text} — check DATABENTO_API_KEY (32 chars, starts with 'db-')"
    if "402" in text or "payment" in lowered or "credit" in lowered:
        return f"{text} — the account is out of credit, not misconfigured"
    if "403" in text or "forbidden" in lowered:
        return (
            f"{text} — this key may not be entitled to that dataset. A 403 from "
            f"an egress proxy instead means hist.databento.com is blocked, which "
            f"is a network policy and not a key problem."
        )
    if "422" in text or "symbol" in lowered:
        return (
            f"{text} — the symbol probably did not resolve. A futures root like "
            f"'ES' needs continuous symbology ('ES.v.0'); an expired contract "
            f"outside its trading window returns nothing."
        )
    return text
