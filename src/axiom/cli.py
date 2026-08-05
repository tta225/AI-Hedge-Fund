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
from axiom.data.providers import CSVProvider
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


@app.command("data-check")
def data_check(
    dataset: str = typer.Option(
        None,
        "--dataset",
        help="Hugging Face dataset id to verify, e.g. user/my-ohlcv. "
        "Without it only the token is checked, not that any dataset resolves.",
    ),
) -> None:
    """Report which market data sources are reachable and credentialed.

    Run this first. Environment variables are injected at container start, so a
    credential saved mid-session will not appear until a new session begins —
    this command makes that visible instead of surfacing as a 401 later.

    Every source is *probed*, not merely inspected for credentials. A key that is
    present but unusable — wrong value, or a host the network will not route to —
    reports as a failure here rather than as a stack trace inside a backtest.
    """
    from axiom.core.credentials import describe_credentials, misnamed_credentials
    from axiom.data import AlpacaProvider, CoinbaseProvider, HuggingFaceProvider
    from axiom.data.registry import default_registry

    console.print("\n[bold]Credentials[/bold]")
    for line in describe_credentials().splitlines():
        console.print(f"  {line}")

    # A key saved under the wrong name is present but unreadable, and reports
    # identically to one that was never saved. Say which it is.
    if near_misses := misnamed_credentials():
        console.print(
            "\n  [yellow]These are set but nothing reads them — the name is "
            "wrong, not the key:[/yellow]"
        )
        for actual, canonical in near_misses:
            console.print(f"    [yellow]{actual}[/yellow] → rename to [bold]{canonical}[/bold]")

    console.print("\n[bold]Market data sources[/bold]\n")

    # Reachability is only knowable by asking. Record what each probe found so
    # the summary below reports verified state rather than repeating the
    # credential check under a different name.
    reachable: dict[str, bool] = {}

    def report(name: str, status: str) -> None:
        console.print(f"[bold cyan]{name}[/bold cyan]")
        for line in status.splitlines():
            # `[huggingface]` and friends are rich markup; escape so an extra
            # name is not silently swallowed from the output.
            console.print(f"  {line.replace('[', chr(92) + '[')}")
        reachable[name.lower()] = status.lstrip().startswith("✓")

    report("Alpaca", AlpacaProvider().check_credentials())
    console.print()
    report("Coinbase", CoinbaseProvider().check_reachable())
    console.print()

    if dataset:
        report("Hugging Face", HuggingFaceProvider(dataset).check_dataset())
    else:
        # Without a dataset id only the token can be checked. Say that plainly:
        # a valid token pointed at a dataset that does not exist looks identical
        # to a working setup until the first load fails.
        status = HuggingFaceProvider("unused/unused").check_token()
        if status.startswith("✓"):
            status += "\n  ? pass --dataset <id> to verify a dataset resolves"
        report("Hugging Face", status)

    registry = default_registry()
    credentialed = [p.name for p in registry.available()]
    # `available()` tests credentials and imports, never the network. Report the
    # two separately: a provider can be fully credentialed and still unroutable,
    # and collapsing them is how "usable now" ends up contradicting a failure
    # printed four lines above it.
    confirmed = [name for name in credentialed if reachable.get(name)]
    blocked = [name for name in credentialed if reachable.get(name) is False]

    console.print("\n[bold cyan]Registry[/bold cyan]")
    console.print(f"  order        : {[p.name for p in registry.providers]}")
    console.print(f"  credentialed : {credentialed or '[red]none[/red]'}")
    console.print(f"  reachable    : {confirmed or '[red]none[/red]'}")
    if blocked:
        console.print(
            f"  [yellow]credentialed but unreachable: {blocked} — the keys are "
            f"fine; the network is not[/yellow]"
        )
    if not confirmed:
        console.print(
            "\n[yellow]No real data source is usable. Everything still runs with "
            "--synthetic, but synthetic results are a correctness check, never "
            "evidence.[/yellow]\n"
            "[dim]See docs/DATA_SETUP.md.[/dim]"
        )


@app.command()
def setup() -> None:
    """Interactively save your API keys. Run this once.

    Prompts for each credential, writes them to a gitignored `.env` file with
    owner-only permissions, and verifies them immediately. Secrets are never
    echoed to the screen and never appear in shell history.
    """
    import stat
    from pathlib import Path

    from axiom.core.credentials import find_env_file, parse_env_file, reset_cache

    console.print("\n[bold cyan]AXIOM credential setup[/bold cyan]")
    console.print(
        "[dim]Keys are saved to a local .env file that git cannot see.\n"
        "Nothing you type here is displayed or logged.[/dim]\n"
    )

    root = Path.cwd()
    if not (root / "pyproject.toml").exists():
        found = find_env_file()
        root = found.parent if found else root
    env_path = root / ".env"

    existing = parse_env_file(env_path) if env_path.exists() else {}
    if existing:
        console.print(f"[yellow]Found an existing {env_path}[/yellow]")
        console.print("[dim]Press Enter at any prompt to keep the current value.[/dim]\n")

    fields = [
        ("APCA_API_KEY_ID", "Alpaca Key ID", "starts with PK, about 20 characters", False),
        ("APCA_API_SECRET_KEY", "Alpaca Secret Key", "about 40 characters", True),
        ("HF_TOKEN", "Hugging Face token", "starts with hf_ — leave blank if your dataset is public", True),
    ]

    values = dict(existing)
    for name, label, hint, secret in fields:
        current = existing.get(name, "")
        suffix = f" [{len(current)} chars saved]" if current else ""
        console.print(f"[bold]{label}[/bold] [dim]({hint})[/dim]{suffix}")
        entered = typer.prompt("  paste here", default="", hide_input=secret,
                               show_default=False).strip()
        if entered:
            values[name] = entered
        elif current:
            values[name] = current
        console.print()

    lines = [
        "# AXIOM credentials. Gitignored — never commit this file.",
        "# Regenerate with: python -m axiom.cli setup",
        "",
    ]
    lines += [f"{k}={v}" for k, v in values.items() if v]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Owner read/write only: other users on the machine cannot read the keys.
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    console.print(f"[green]✓ Saved to {env_path}[/green] [dim](permissions 600)[/dim]")

    gitignore = root / ".gitignore"
    if gitignore.exists() and ".env" not in gitignore.read_text().splitlines():
        gitignore.write_text(gitignore.read_text().rstrip() + "\n.env\n")
        console.print("[green]✓ Added .env to .gitignore[/green]")

    reset_cache()
    console.print("\n[bold]Checking them now...[/bold]")
    data_check()


@app.command("base-rates")
def base_rates(
    symbol: str = typer.Option("BTC-USD", help="Instrument symbol."),
    timeframe: str = typer.Option("1h"),
    days: int = typer.Option(365),
    horizon: int = typer.Option(48, help="Forward bars over which an outcome is judged."),
    synthetic: bool = typer.Option(False),
    seed: int = typer.Option(7),
    data_root: str = typer.Option("", help="Read from a local CSV cache instead."),
) -> None:
    """Measure how often ICT features actually do what they claim.

    Every rate is reported against a control, because a fill rate without one
    measures volatility and calls it an edge. See docs/REAL_DATA_FINDINGS.md
    for what the controls did to the priors.
    """
    from axiom.research.base_rates import measure_base_rates

    if data_root:
        series = CSVProvider(data_root).fetch(get_instrument(symbol), timeframe, days=days)
    else:
        series = _load_series(symbol, timeframe, days, synthetic, seed)

    console.print(measure_base_rates(series, horizon=horizon).render())


@app.command("cache-data")
def cache_data(
    symbols: str = typer.Option("BTC-USD,ETH-USD,SOL-USD", help="Comma-separated."),
    timeframe: str = typer.Option("1h"),
    days: int = typer.Option(365),
    out: str = typer.Option("data/cache", help="Directory to write CSVs into."),
) -> None:
    """Download real bars to a local CSV cache. No credentials needed.

    Coinbase Exchange serves keyless public candles, which is what makes the
    measurements in docs/REAL_DATA_FINDINGS.md reproducible without anyone
    having to hold an API key. The cache itself is gitignored — market data is
    the venue's to distribute, not this repository's.
    """
    from axiom.data.coinbase import CoinbaseProvider

    provider = CoinbaseProvider(max_requests=600)
    for symbol in (s.strip() for s in symbols.split(",") if s.strip()):
        instrument = get_instrument(symbol)
        try:
            series = provider.fetch(instrument, timeframe, days=days)
        except ProviderError as exc:
            console.print(f"[red]{symbol}: {exc}[/red]")
            continue
        path = CSVProvider.write(series, f"{out}/{symbol}_{timeframe}.csv")
        console.print(
            f"[green]✓[/green] {symbol:<9} {timeframe:<4} "
            f"{len(series):>6,} bars → {path}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
