"""AXIOM web API and single-page operator console.

Design
------
This is a thin HTTP surface over the same engines the CLI drives. It contains no
trading logic of its own — every number it returns was computed by
`ICTEngine`, `Backtester` or `RegimeModel`, so the browser and the terminal can
never disagree about what the platform thinks.

Two rules it does enforce:

**Synthetic data is opt-in, never a fallback.** A failed feed returns HTTP 503
naming the providers that failed. It does not quietly substitute generated bars
— a fabricated equity curve rendered in a polished UI is worse than an error
message, because it looks like a result.

**Every price-bearing response carries provenance.** See `serialise`.

Run it with `axiom web`, or `uvicorn axiom.web.server:app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from axiom.agents.pipeline import AgentPipeline
from axiom.backtest.engine import Backtester
from axiom.core.config import RiskSettings, get_settings
from axiom.core.series import OHLCVSeries
from axiom.core.types import INSTRUMENTS, get_instrument
from axiom.data.base import BarRequest, ProviderError
from axiom.data.registry import default_registry
from axiom.data.synthetic import SyntheticProvider
from axiom.ict.engine import STRICT_CONFIG, ICTConfig, ICTEngine
from axiom.portfolio.positions import Portfolio
from axiom.web.serialise import (
    agent_payload,
    backtest_payload,
    ict_payload,
    instrument_payload,
    provenance_payload,
    ranking_payload,
    series_payload,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - guarded by the CLI
    raise ImportError(
        "the web extra is not installed. Run:\n"
        "    pip install -e '.[web]'"
    ) from exc

log = logging.getLogger("axiom.web")

STATIC_DIR = Path(__file__).parent / "static"

#: Strategies exposed to the UI. Mirrors the CLI's table deliberately: one of
#: them growing a strategy the other cannot run is a bug, not a feature.
from axiom.cli import STRATEGIES  # noqa: E402  (import cycle avoided at runtime)

#: Hard ceiling on a single request's lookback. Without it, one careless
#: `days=100000` on a 1m timeframe pages a provider for twenty minutes while the
#: browser sits on a spinner and the worker is unavailable to everyone else.
MAX_DAYS = 1825

app = FastAPI(
    title="AXIOM",
    description="ICT alpha engine, research terminal, and paper-first execution.",
    version="0.1.0",
)


class SeriesQuery(BaseModel):
    """Common bar-selection parameters."""

    #: SPY, not ES. ES is the instrument this platform is really about, but it
    #: is a *future*, and no free provider carries futures — so defaulting to it
    #: means every first load fails with a 503 before the user has done anything
    #: wrong. It used to "work" only because an equity provider answered ES with
    #: Eversource Energy. A default that needs a paid subscription to render is
    #: not a good first impression; a default that renders the wrong company is
    #: worse. SPY is served by Alpaca, by the HF bucket, and by yfinance.
    symbol: str = Field("SPY", max_length=32)
    timeframe: str = Field("15m", max_length=8)
    days: int = Field(30, ge=1, le=MAX_DAYS)
    synthetic: bool = False
    seed: int = Field(7, ge=0, le=2**31 - 1)


class AnalyseQuery(SeriesQuery):
    strict: bool = Field(False, description="Use the high-conviction ICT preset.")
    max_bars: int = Field(1500, ge=50, le=6000)


class BacktestQuery(SeriesQuery):
    days: int = Field(120, ge=1, le=MAX_DAYS)
    strategy: str = Field("silver-bullet", max_length=64)
    equity: float = Field(250_000.0, gt=0, le=1e12)
    warmup: int = Field(100, ge=0, le=5000)


def _load_series(query: SeriesQuery) -> OHLCVSeries:
    """Fetch bars, or fail loudly.

    The `synthetic` flag is the only path to generated data. There is no
    fallback: see the module docstring.
    """
    instrument = get_instrument(query.symbol)
    request = BarRequest.lookback(instrument, query.timeframe, query.days)

    if query.synthetic:
        return SyntheticProvider(seed=query.seed, start_price=5200.0).fetch_bars(request)

    try:
        return default_registry().fetch_bars(request)
    except ProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "no_market_data",
                "message": str(exc),
                "hint": (
                    "No real data provider could serve this request. Enable "
                    "'Generated bars' to run offline — results from generated "
                    "data are a correctness check, never performance."
                ),
            },
        ) from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "axiom", "version": app.version}


@app.get("/api/instruments")
def instruments() -> dict[str, Any]:
    """The built-in instrument catalogue, plus the strategy and timeframe menus."""
    return {
        "instruments": [instrument_payload(i) for i in INSTRUMENTS.values()],
        "strategies": sorted(STRATEGIES),
        "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Which data sources are credentialed, and which are actually reachable.

    These are different questions — a provider can be fully credentialed and
    still unroutable on a restricted network — so the UI shows both. See
    `docs/DATA_SETUP.md`.
    """
    from axiom.core.credentials import describe_credentials
    from axiom.data import (
        AlpacaProvider,
        CoinbaseProvider,
        DatabentoProvider,
        HuggingFaceProvider,
        MassiveProvider,
    )

    probes = [
        ("alpaca", AlpacaProvider().check_credentials()),
        ("coinbase", CoinbaseProvider().check_reachable()),
        ("databento", DatabentoProvider().check_credentials()),
        ("massive", MassiveProvider().check_credentials()),
        ("huggingface", HuggingFaceProvider("unused/unused").check_token()),
    ]
    sources = [
        {
            "name": name,
            "ok": text.lstrip().startswith("✓"),
            "detail": text.strip(),
        }
        for name, text in probes
    ]

    registry = default_registry()
    reachable = {s["name"] for s in sources if s["ok"]}
    credentialed = [p.name for p in registry.available()]

    return {
        "credentials": describe_credentials(),
        "sources": sources,
        "registry": {
            "order": [p.name for p in registry.providers],
            "credentialed": credentialed,
            "reachable": [n for n in credentialed if n in reachable],
        },
        "settings": {
            "trading_mode": get_settings().trading_mode.value,
            "kill_switch": get_settings().kill_switch,
        },
    }


@app.post("/api/analyse")
def analyse(query: AnalyseQuery) -> dict[str, Any]:
    """Bars plus the full ICT structural read, ready to draw."""
    series = _load_series(query)
    config = STRICT_CONFIG if query.strict else ICTConfig()
    state = ICTEngine(config).analyse(series)

    payload = series_payload(series, max_bars=query.max_bars)
    return {
        "series": payload,
        "ict": ict_payload(state, series, offset=payload["index_offset"]),
        "instrument": instrument_payload(series.instrument),
    }


@app.post("/api/backtest")
def backtest(query: BacktestQuery) -> dict[str, Any]:
    """Run a strategy. Costs and slippage are always applied."""
    if query.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_strategy",
                "message": f"Unknown strategy {query.strategy!r}.",
                "available": sorted(STRATEGIES),
            },
        )

    series = _load_series(query)
    result = Backtester(
        STRATEGIES[query.strategy](),
        risk_settings=RiskSettings(
            account_equity=query.equity, max_gross_exposure_pct=2000.0
        ),
    ).run(series, warmup=query.warmup)

    return {
        "backtest": backtest_payload(result, query.strategy),
        "series": series_payload(series, max_bars=1500),
    }


@app.post("/api/regime")
def regime(query: SeriesQuery) -> dict[str, Any]:
    """Causal HMM regime classification over the window.

    Refits on a rolling window, so it is materially slower than `/api/analyse`
    and is a separate call rather than part of the dashboard payload.
    """
    from axiom.quant.regime import RegimeModel

    series = _load_series(query)
    try:
        regimes = RegimeModel().fit_causal(series)
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "hmm_unavailable",
                "message": str(exc),
                "hint": "pip install -e '.[quant]'",
            },
        ) from exc

    payload = series_payload(series, max_bars=1500)
    offset = payload["index_offset"]
    return {
        "series": payload,
        "regime": {
            "summary": regimes.summary(),
            "labels": [
                regimes.label_at(i).value for i in range(offset, len(series))
            ],
            "confidence": [
                float(regimes.confidence_at(i)) for i in range(offset, len(series))
            ],
            "occupancy": {k.value: v for k, v in regimes.occupancy().items()},
        },
        "provenance": provenance_payload(series.provenance),
    }



class PipelineQuery(SeriesQuery):
    days: int = Field(120, ge=1, le=MAX_DAYS)
    strategy: str = Field("silver-bullet", max_length=64)
    equity: float = Field(250_000.0, gt=0, le=1e12)
    ensemble: bool = Field(
        False,
        description=(
            "Select the strategy from the whole archive by out-of-sample "
            "record instead of using `strategy`."
        ),
    )
    folds: int = Field(3, ge=2, le=8, description="Walk-forward splits, if ensemble.")


class RankQuery(SeriesQuery):
    days: int = Field(250, ge=1, le=MAX_DAYS)
    timeframe: str = Field("1h", max_length=8)
    folds: int = Field(3, ge=2, le=8)
    warmup: int = Field(250, ge=50, le=5000)
    equity: float = Field(5_000_000.0, gt=0, le=1e12)
    top: int = Field(10, ge=1, le=50)


@app.post("/api/pipeline")
def pipeline(query: PipelineQuery) -> dict[str, Any]:
    """Run Research → Debate → Backtest → Risk → Review.

    There is no execute stage and the payload says so (`can_execute: false`).
    The pipeline terminates in an approval request that a human acts on; it
    cannot place an order, and no UI should imply otherwise.

    Measured `facts` and generated `narrative` stay in separate fields all the
    way to the browser. Merging them would destroy the one property that makes
    an LLM safe in a trading loop — that a reader can always tell which numbers
    were computed and which prose was written.
    """
    if not query.ensemble and query.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_strategy",
                "message": f"Unknown strategy {query.strategy!r}.",
                "available": sorted(STRATEGIES),
            },
        )

    series = _load_series(query)
    settings = get_settings()
    risk_settings = RiskSettings(
        account_equity=query.equity, max_gross_exposure_pct=2000.0
    )
    agents = AgentPipeline(settings.agents)
    if query.ensemble:
        result = agents.run_ensemble(
            series, risk_settings=risk_settings, folds=query.folds
        )
    else:
        result = agents.run(
            series, STRATEGIES[query.strategy](), risk_settings=risk_settings
        )
    return {
        "pipeline": agent_payload(result),
        "llm_enabled": settings.agents.enabled,
        "provenance": provenance_payload(series.provenance),
        "series": series_payload(series, max_bars=600),
    }


@app.post("/api/rank")
def rank(query: RankQuery) -> dict[str, Any]:
    """Rank live signals across the archive by out-of-sample evidence.

    Slow by construction — every strategy is scored walk-forward before anything
    is ranked. That is the cost of not ranking on in-sample results, which is
    the fastest way to promote the luckiest strategy in a field of nulls.
    """
    from axiom.research.ensemble import SignalEnsemble

    series = _load_series(query)
    ensemble = SignalEnsemble(
        folds=query.folds,
        warmup=query.warmup,
        risk_settings=RiskSettings(
            account_equity=query.equity, max_gross_exposure_pct=5000.0
        ),
    )
    ranking = ensemble.rank(series, top=query.top)
    return {
        "ranking": ranking_payload(ranking),
        "strategies_scored": len(ensemble.entries),
        "provenance": provenance_payload(series.provenance),
    }


def _console_venue(name: str) -> Any:
    """Build a venue for the console. **Read-only forms only.**

    Deliberately narrower than the CLI's `_build_venue`, which takes a `--live`
    flag. The console has no confirmation step worth the name — a browser tab
    left open is not a deliberate act — so it never constructs a venue in its
    real-money form. Tradovate is built against the demo host; NinjaTrader is
    built with `allow_live=True` because it has no other form, and is only ever
    read from. The console has no order-placing endpoint at all, so the worst a
    misread here produces is a wrong status line, not a wrong trade.
    """
    if name == "tradovate":
        from axiom.execution.tradovate import DEMO_HOST, TradovateVenue

        return TradovateVenue(host=DEMO_HOST, allow_live=False)
    if name == "ninjatrader":
        from axiom.core.credentials import get_credential
        from axiom.execution.ninjatrader import NinjaTraderVenue

        return NinjaTraderVenue(
            account=get_credential("NINJATRADER_ACCOUNT") or "Sim101",
            allow_live=True,
        )
    from axiom.execution.alpaca_paper import AlpacaPaperVenue

    return AlpacaPaperVenue()


@app.get("/api/broker")
def broker(venue_name: str = "alpaca", reconcile: bool = False) -> dict[str, Any]:
    """Broker account status, and optionally a reconciliation.

    Reports `is_live` explicitly. A console that shows broker state without
    saying whether it is real money is a console that will eventually be
    misread.
    """
    venue = _console_venue(venue_name)
    payload: dict[str, Any] = {
        "venue": venue.name,
        "is_live": venue.is_live,
        "available": venue.is_available(),
        "status": venue.check(),
    }
    if not venue.is_available():
        return payload

    try:
        payload["account"] = venue.account() if hasattr(venue, "account") else {}
        payload["positions"] = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_price": p.average_price,
                # Futures venues report size and price, not a marked value —
                # reporting 0.0 would read as a worthless position rather than
                # an unreported one, so the field is omitted instead.
                "market_value": getattr(p, "market_value", None),
                "unrealised_pnl": getattr(p, "unrealised_pnl", None),
            }
            for p in venue.positions().values()
        ]
        if reconcile:
            report = venue.reconcile(
                Portfolio(starting_cash=get_settings().risk.account_equity)
            )
            payload["reconciliation"] = {
                "is_clean": report.is_clean,
                "mismatched": {k: list(v) for k, v in report.mismatched.items()},
                "broker_only": report.broker_only,
                "platform_only": report.platform_only,
                "equity_drift": report.equity_drift,
                "render": report.render(),
            }
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


# ---------------------------------------------------------------------------
# Static console. Mounted last so it cannot shadow an /api route.
# ---------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def console() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(ProviderError)
def provider_error_handler(_request: Any, exc: ProviderError) -> JSONResponse:
    """A data failure is a 503, not a 500 — it is upstream, and it is retryable."""
    return JSONResponse(
        status_code=503,
        content={"detail": {"error": "provider_error", "message": str(exc)}},
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    log_level: Literal["critical", "error", "warning", "info", "debug"] = "info",
) -> None:
    """Run the console. Called by `axiom web`."""
    import uvicorn

    uvicorn.run(
        "axiom.web.server:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
