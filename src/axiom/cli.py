"""AXIOM command line."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import typer
from rich.console import Console

from axiom.agents.pipeline import AgentPipeline
from axiom.backtest.engine import Backtester
from axiom.core.config import RiskSettings, get_settings
from axiom.core.series import OHLCVSeries
from axiom.core.timeframe import Timeframe
from axiom.core.types import get_instrument
from axiom.data.base import BarRequest, ProviderError
from axiom.data.registry import default_registry
from axiom.data.synthetic import SyntheticProvider
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.portfolio.positions import Portfolio
from axiom.quant.regime import RegimeModel
from axiom.research.lab import Candidate, StrategyLab, grid
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Strategy
from axiom.strategy.ict_strategies import LiquidityRaidReversal, SilverBulletStrategy
from axiom.strategy.quant_strategies import (
    MeanReversionZScore,
    TimeSeriesMomentum,
    VolatilityBreakout,
)
from axiom.terminal.app import TerminalState, render

app = typer.Typer(
    add_completion=False,
    help="AXIOM — ICT alpha engine, research terminal, and paper-first execution.",
)
console = Console()

STRATEGIES: dict[str, type[Strategy]] = {
    "silver-bullet": SilverBulletStrategy,
    "liquidity-raid": LiquidityRaidReversal,
    "momentum": TimeSeriesMomentum,
    "mean-reversion": MeanReversionZScore,
    "vol-breakout": VolatilityBreakout,
}


def _make(cls: type[Strategy], params: dict[str, Any]) -> Strategy:
    """Build a strategy from a parameter dict.

    A named function rather than a default-capture lambda so the candidate
    factory has an inferable type.
    """
    return cls(**params)


def _load_series(
    symbol: str, timeframe: str, days: int, synthetic: bool, seed: int
) -> OHLCVSeries:
    """Fetch bars, falling back to synthetic **only** when asked explicitly.

    Never silently substitutes generated data for a failed feed — that is how a
    fabricated backtest happens.
    """
    instrument = get_instrument(symbol)
    request = BarRequest.lookback(instrument, timeframe, days)

    if synthetic:
        return SyntheticProvider(seed=seed, start_price=5200.0).fetch_bars(request)

    try:
        return default_registry().fetch_bars(request)
    except ProviderError as exc:
        console.print(f"[bold red]No market data available.[/bold red]\n{exc}")
        console.print(
            "\n[yellow]Pass --synthetic to run against generated bars instead. "
            "Results from generated data are a correctness check, never "
            "performance.[/yellow]"
        )
        raise typer.Exit(code=1) from exc


@app.command()
def terminal(
    symbol: str = typer.Option("ES", help="Instrument symbol."),
    timeframe: str = typer.Option("15m", help="Bar interval, e.g. 5m, 15m, 1h."),
    days: int = typer.Option(30, help="Lookback window in days."),
    synthetic: bool = typer.Option(False, help="Use generated bars (offline)."),
    seed: int = typer.Option(7, help="Synthetic generator seed."),
) -> None:
    """Render the AXIOM operator terminal."""
    series = _load_series(symbol, timeframe, days, synthetic, seed)
    state = ICTEngine().analyse(series)
    settings = get_settings()
    render(
        TerminalState(
            series=series,
            ict=state,
            portfolio=Portfolio(starting_cash=settings.risk.account_equity),
            risk=RiskManager(
                settings.risk, mode=settings.trading_mode,
                kill_switch=settings.kill_switch,
            ),
        ),
        console,
    )


@app.command()
def analyse(
    symbol: str = typer.Option("ES"),
    timeframe: str = typer.Option("15m"),
    days: int = typer.Option(30),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(7),
    strict: bool = typer.Option(False, help="Use the high-conviction preset."),
) -> None:
    """Print the ICT structural read for a symbol."""
    from axiom.ict.engine import STRICT_CONFIG

    series = _load_series(symbol, timeframe, days, synthetic, seed)
    state = ICTEngine(STRICT_CONFIG if strict else ICTConfig()).analyse(series)

    console.print(f"\n[bold cyan]{series.describe()}[/bold cyan]\n")
    console.print(state.summary())
    console.print(
        f"\n  swings            {len(state.swings)}"
        f"\n  structure events  {len(state.structure_events)}"
        f"\n  fair value gaps   {len(state.fair_value_gaps)}"
        f"\n  order blocks      {len(state.order_blocks)}"
        f"\n  liquidity pools   {len(state.liquidity_pools)}"
        f"\n  sweeps            {len(state.sweeps)}"
    )
    if state.dealing_range is not None:
        dealing = state.dealing_range
        price = float(series.closes[-1])
        console.print(
            f"\n  dealing range     {dealing.low:,.2f} — {dealing.high:,.2f}"
            f"\n  equilibrium       {dealing.equilibrium:,.2f}"
            f"\n  position          {dealing.position_of(price):.1%} "
            f"({'premium' if dealing.is_premium(price) else 'discount'})"
        )


@app.command()
def backtest(
    strategy: str = typer.Option("silver-bullet", help=f"One of: {', '.join(STRATEGIES)}"),
    symbol: str = typer.Option("ES"),
    timeframe: str = typer.Option("15m"),
    days: int = typer.Option(120),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(11),
    equity: float = typer.Option(250_000.0, help="Starting account equity."),
    warmup: int = typer.Option(100, help="Bars excluded before trading starts."),
) -> None:
    """Backtest a strategy. Costs and slippage are always applied."""
    if strategy not in STRATEGIES:
        console.print(f"[red]Unknown strategy {strategy!r}. Choose from: "
                      f"{', '.join(STRATEGIES)}[/red]")
        raise typer.Exit(code=1)

    series = _load_series(symbol, timeframe, days, synthetic, seed)
    result = Backtester(
        STRATEGIES[strategy](),
        risk_settings=RiskSettings(
            account_equity=equity, max_gross_exposure_pct=2000.0
        ),
    ).run(series, warmup=warmup)

    console.print()
    console.print(result.report.render())
    console.print("\n[bold]Signal funnel[/bold]")
    console.print(result.funnel())


@app.command()
def pipeline(
    strategy: str = typer.Option("silver-bullet"),
    symbol: str = typer.Option("ES"),
    timeframe: str = typer.Option("15m"),
    days: int = typer.Option(120),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(11),
    equity: float = typer.Option(250_000.0),
) -> None:
    """Run the Research → Debate → Backtest → Risk → Review agent pipeline.

    Terminates in a human approval request. It cannot place an order.
    """
    if strategy not in STRATEGIES:
        console.print(f"[red]Unknown strategy {strategy!r}.[/red]")
        raise typer.Exit(code=1)

    series = _load_series(symbol, timeframe, days, synthetic, seed)
    settings = get_settings()
    if not settings.agents.enabled:
        console.print(
            "[yellow]No ANTHROPIC_API_KEY set — running in deterministic mode. "
            "All measured facts are still computed; only the narrative is "
            "omitted.[/yellow]\n"
        )

    outcome = AgentPipeline(settings.agents).run(
        series,
        STRATEGIES[strategy](),
        risk_settings=RiskSettings(
            account_equity=equity, max_gross_exposure_pct=2000.0
        ),
    )
    console.print(outcome.render())


@app.command()
def demo() -> None:
    """Offline end-to-end demonstration on generated data."""
    console.print(
        "[bold yellow]Running on SYNTHETIC data. Nothing here is evidence of "
        "profitability.[/bold yellow]\n"
    )
    end = datetime.now(UTC)
    instrument = get_instrument("ES")
    series = SyntheticProvider(seed=11, start_price=5200.0).fetch_bars(
        BarRequest(instrument, Timeframe.parse("15m"), end - timedelta(days=120), end)
    )
    state = ICTEngine().analyse(series)
    settings = get_settings()

    render(
        TerminalState(
            series=series,
            ict=state,
            portfolio=Portfolio(starting_cash=250_000),
            risk=RiskManager(settings.risk, mode=settings.trading_mode),
        ),
        console,
    )

    result = Backtester(
        SilverBulletStrategy(),
        risk_settings=RiskSettings(
            account_equity=250_000, max_gross_exposure_pct=2000.0
        ),
    ).run(series, warmup=100)
    console.print()
    console.print(result.report.render())
    console.print("\n[bold]Signal funnel[/bold]")
    console.print(result.funnel())


@app.command()
def config() -> None:
    """Show effective configuration and the current safety posture."""
    settings = get_settings()
    console.print(f"\n[bold]trading mode[/bold]      {settings.trading_mode.value}")
    console.print(
        f"[bold]kill switch[/bold]       "
        f"{'[red]ENGAGED[/red]' if settings.kill_switch else 'clear'}"
    )
    console.print(f"[bold]orders permitted[/bold]  {settings.orders_permitted}")
    console.print("\n[bold]risk limits[/bold]")
    risk = settings.risk
    console.print(f"  account equity          ${risk.account_equity:,.2f}")
    console.print(
        f"  max risk / trade        {risk.max_risk_per_trade_pct}% "
        f"(${risk.max_risk_per_trade_cash:,.2f})"
    )
    console.print(
        f"  daily loss limit        {risk.daily_loss_limit_pct}% "
        f"(${risk.daily_loss_limit_cash:,.2f})"
    )
    console.print(f"  max gross exposure      {risk.max_gross_exposure_pct}%")
    console.print(f"  max positions           {risk.max_positions}")
    console.print(f"  max consecutive losses  {risk.max_consecutive_losses}")
    console.print("\n[bold]execution assumptions[/bold]")
    console.print(f"  commission / unit       ${settings.execution.commission_per_unit}")
    console.print(f"  slippage                {settings.execution.slippage_ticks} ticks")
    console.print(
        f"\n[bold]agents[/bold]           "
        f"{'enabled (' + settings.agents.agent_model + ')' if settings.agents.enabled else 'deterministic mode (no API key)'}"
    )


@app.command()
def regime(
    symbol: str = typer.Option("ES"),
    timeframe: str = typer.Option("1h"),
    days: int = typer.Option(365),
    states: int = typer.Option(3, help="Number of hidden states."),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(11),
) -> None:
    """Detect market regimes with a causal HMM.

    Fitting uses an expanding window of past data and inference uses the
    forward filter only, so the regime at any bar depends on nothing after it.
    """
    series = _load_series(symbol, timeframe, days, synthetic, seed)
    console.print(f"[cyan]{series.describe()}[/cyan]")
    console.print("[dim]fitting causal HMM (expanding-window refit)...[/dim]")

    regimes = RegimeModel(n_states=states).fit_causal(series)
    console.print(f"\n{regimes.summary()}\n")
    for state, label in sorted(regimes.labels.items()):
        console.print(f"  state {state}  {label.value}")

    last = len(series) - 1
    if regimes.is_known_at(last):
        console.print(
            f"\n[bold]current regime[/bold]  {regimes.label_at(last).value} "
            f"(confidence {regimes.confidence_at(last):.0%})"
        )
    else:
        console.print("\n[yellow]not enough history to assign a regime yet[/yellow]")


@app.command()
def research(
    symbol: str = typer.Option("ES"),
    timeframe: str = typer.Option("1h"),
    days: int = typer.Option(365),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(11),
    equity: float = typer.Option(5_000_000.0, help="Account equity for sizing."),
    folds: int = typer.Option(4, help="Walk-forward folds."),
) -> None:
    """Sweep many strategies walk-forward and rank them honestly.

    Every candidate counts as a trial, and the winner is judged against the
    deflated Sharpe — the bar the best of that many coin flips would clear.
    A search that concludes "nothing survived" has done its job.
    """
    series = _load_series(symbol, timeframe, days, synthetic, seed)
    console.print(f"[cyan]{series.describe()}[/cyan]")

    candidates: list[Candidate] = []
    for params in grid(lookback=[48, 96, 192], stop_atr=[1.5, 2.5]):
        candidates.append(
            Candidate(
                "ts_momentum",
                partial(_make, TimeSeriesMomentum, params),
                params,
            )
        )
    for params in grid(lookback=[48, 96], entry_z=[1.5, 2.0, 2.5]):
        candidates.append(
            Candidate(
                "mean_reversion",
                partial(_make, MeanReversionZScore, params),
                params,
            )
        )
    for params in grid(breakout_lookback=[12, 24]):
        candidates.append(
            Candidate(
                "vol_breakout",
                partial(_make, VolatilityBreakout, params),
                params,
            )
        )

    console.print(f"[dim]evaluating {len(candidates)} candidates over {folds} folds...[/dim]\n")
    outcome = StrategyLab(
        risk_settings=RiskSettings(
            account_equity=equity, max_gross_exposure_pct=2000.0
        ),
        n_folds=folds,
        warmup=200,
    ).search(series, candidates)
    console.print(outcome.render())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
