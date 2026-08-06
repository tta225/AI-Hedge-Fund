"""Multi-instrument backtesting on one clock, with one risk budget.

What makes this a portfolio backtest
------------------------------------
Running the single-instrument backtester once per symbol and adding up the
results is **not** a portfolio backtest, and the difference is not academic. Run
separately, every instrument gets the full risk budget, the full position count,
and the full daily loss allowance. Five symbols each risking 1% is not a 1%-risk
portfolio — it is a 5%-risk portfolio wearing the label of a 1% one, and its
equity curve is correspondingly fictitious.

So this drives every instrument from a **single clock**, against a **single
`Portfolio`** and a **single `RiskManager`**. When ES takes a position, NQ's
signal on the same bar is evaluated against a book that already contains it, and
is rejected if that would breach `max_positions`, gross exposure, or what remains
of the daily loss budget. The competition for the budget is the thing being
simulated.

The clock
---------
Instruments rarely share a bar grid: ES has a session, BTC does not, and a
15-minute bar on one is not aligned with the other. The timeline is the sorted
union of every series' timestamps, and at each point only the instruments that
actually printed a bar are stepped. No instrument is ever advanced onto a bar it
does not have, and none is asked for a price it has not yet printed — that would
be lookahead dressed up as interpolation.

Ordering within a timestamp
---------------------------
When several instruments print at the same moment, they are processed in a fixed
order (the order given) and that order decides who gets a contested last slot.
This is a real bias and it is reported as `contested_slots` rather than hidden.
Nothing about a backtest can remove it; only saying so keeps it honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from axiom.backtest.engine import Backtester, _OpenTrade
from axiom.backtest.metrics import PerformanceReport, Trade, build_report
from axiom.core.config import ExecutionSettings, RiskSettings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.core.types import TradingMode
from axiom.execution.base import Order
from axiom.execution.router import OrderRouter
from axiom.execution.simulated import FillModel, SimulatedVenue
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal, Strategy, StrategyContext


@dataclass(slots=True)
class InstrumentAttribution:
    """What one instrument contributed to the portfolio result."""

    symbol: str
    trades: list[Trade] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    realised_pnl: float = 0.0
    commission: float = 0.0
    risk_rejections: dict[str, int] = field(default_factory=dict)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def win_rate_pct(self) -> float:
        return 100.0 * self.wins / len(self.trades) if self.trades else float("nan")


@dataclass(slots=True)
class PortfolioBacktestResult:
    """Portfolio-level result, plus per-instrument attribution."""

    report: PerformanceReport
    portfolio: Portfolio
    trades: list[Trade]
    attribution: dict[str, InstrumentAttribution]
    ict_states: dict[str, ICTState]
    #: Pearson correlation of instrument close-to-close returns over the run.
    correlations: pd.DataFrame
    #: Signals rejected purely because another instrument already held the slot.
    contested_slots: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    risk_rejections: dict[str, int] = field(default_factory=dict)

    def funnel(self) -> str:
        signals = sum(len(a.signals) for a in self.attribution.values())
        lines = [
            f"instruments       : {len(self.attribution)}",
            f"signals generated : {signals}",
            f"trades completed  : {len(self.trades)}",
            f"contested slots   : {self.contested_slots}"
            + ("  (rejected because another instrument held the budget)"
               if self.contested_slots else ""),
        ]
        if self.risk_rejections:
            lines.append("risk rejections:")
            lines += [f"  {v:>5} × {k}" for k, v in self.risk_rejections.items()]
        return "\n".join(lines)

    def attribution_table(self) -> str:
        rows = [f"{'symbol':<10}{'trades':>8}{'win %':>8}{'realised':>14}{'commission':>12}"]
        for a in sorted(self.attribution.values(), key=lambda x: -x.realised_pnl):
            win = "—" if not a.trades else f"{a.win_rate_pct:.1f}"
            rows.append(
                f"{a.symbol:<10}{len(a.trades):>8}{win:>8}"
                f"{a.realised_pnl:>14,.2f}{a.commission:>12,.2f}"
            )
        return "\n".join(rows)


@dataclass(slots=True)
class _Track:
    """Per-instrument execution state advanced by the shared loop."""

    symbol: str
    series: OHLCVSeries
    strategy: Strategy
    state: ICTState
    venue: SimulatedVenue
    router: OrderRouter
    engine: Backtester
    #: Position of the next bar to process. Advances only when this instrument
    #: actually prints at the current timestamp.
    cursor: int = 0
    orders_by_id: dict[int, Order] = field(default_factory=dict)
    open_trade: _OpenTrade | None = None
    pending_entry: tuple[Order, Signal] | None = None
    budgeted_risk_points: dict[int, float] = field(default_factory=dict)
    attribution: InstrumentAttribution = field(default=None)  # type: ignore[assignment]
    last_close: float = float("nan")


class PortfolioBacktester:
    """Runs several instruments against one shared portfolio and risk budget.

    Args:
        strategies: one strategy for every instrument, or a per-symbol mapping.
            A shared instance would carry state between instruments, so a single
            strategy is copied per track via its class and parameters.
        risk_settings: the portfolio's limits. `max_positions`,
            `max_gross_exposure_pct` and `daily_loss_limit_cash` are what make
            this a portfolio rather than a set of independent runs.
    """

    def __init__(
        self,
        strategies: Strategy | dict[str, Strategy],
        *,
        risk_settings: RiskSettings | None = None,
        execution_settings: ExecutionSettings | None = None,
        ict_config: ICTConfig | None = None,
        starting_equity: float | None = None,
        entry_tolerance_atr: float = 0.35,
        stop_gap_atr: float = 0.75,
    ) -> None:
        self.strategies = strategies
        self.risk_settings = risk_settings or RiskSettings()
        self.execution_settings = execution_settings or ExecutionSettings()
        self.ict_config = ict_config or ICTConfig()
        self.starting_equity = starting_equity or self.risk_settings.account_equity
        self.entry_tolerance_atr = entry_tolerance_atr
        self.stop_gap_atr = stop_gap_atr

    def _strategy_for(self, symbol: str) -> Strategy:
        if isinstance(self.strategies, dict):
            try:
                return self.strategies[symbol]
            except KeyError:
                raise ValueError(
                    f"no strategy supplied for {symbol!r}; have "
                    f"{sorted(self.strategies)}"
                ) from None
        # One instance per track. Sharing would let one instrument's internal
        # state leak into another's decisions.
        return type(self.strategies)()

    def run(
        self,
        series_by_symbol: dict[str, OHLCVSeries],
        *,
        warmup: int = 50,
    ) -> PortfolioBacktestResult:
        if not series_by_symbol:
            raise ValueError("no series supplied")
        for symbol, series in series_by_symbol.items():
            if len(series) <= warmup + 2:
                raise ValueError(
                    f"{symbol}: {len(series)} bars is not enough for a "
                    f"{warmup}-bar warmup"
                )

        portfolio = Portfolio(starting_cash=self.starting_equity)
        risk = RiskManager(self.risk_settings, mode=TradingMode.BACKTEST)

        tracks: dict[str, _Track] = {}
        for symbol, series in series_by_symbol.items():
            venue = SimulatedVenue(
                FillModel(
                    slippage_ticks=self.execution_settings.slippage_ticks,
                    commission_per_unit=self.execution_settings.commission_per_unit,
                )
            )
            engine = Backtester(
                self._strategy_for(symbol),
                risk_settings=self.risk_settings,
                execution_settings=self.execution_settings,
                ict_config=self.ict_config,
                starting_equity=self.starting_equity,
                entry_tolerance_atr=self.entry_tolerance_atr,
                stop_gap_atr=self.stop_gap_atr,
            )
            tracks[symbol] = _Track(
                symbol=symbol,
                series=series,
                strategy=engine.strategy,
                state=ICTEngine(self.ict_config).analyse(series),
                venue=venue,
                # Every track routes into the *same* risk manager. This is the
                # single most important line in the module.
                router=OrderRouter(venue=venue, risk=risk, mode=TradingMode.BACKTEST),
                engine=engine,
                attribution=InstrumentAttribution(symbol=symbol),
            )

        timeline = self._merged_timeline(series_by_symbol)
        # Bar lookup per timestamp, built once — a linear scan per instrument
        # per tick would make the run quadratic in the number of bars.
        positions: dict[str, dict[pd.Timestamp, int]] = {
            symbol: {ts: i for i, ts in enumerate(series.index)}
            for symbol, series in series_by_symbol.items()
        }

        trades: list[Trade] = []
        contested = 0

        for timestamp in timeline:
            risk.roll_day(timestamp)

            for symbol, track in tracks.items():
                index = positions[symbol].get(timestamp)
                if index is None:
                    continue  # this instrument did not print on this tick
                track.cursor = index
                closed, contested_here = self._step(
                    track, index, timestamp, portfolio, risk, warmup
                )
                trades.extend(closed)
                track.attribution.trades.extend(closed)
                contested += contested_here

            # Mark every instrument at its most recent close, not only the ones
            # that printed. A portfolio's equity is defined at every instant,
            # and marking only the movers understates exposure between prints.
            portfolio.mark_to_market(
                {
                    tracks[s].series.instrument.symbol: tracks[s].last_close
                    for s in tracks
                    if np.isfinite(tracks[s].last_close)
                }
            )
            portfolio.record(timestamp)

        for track in tracks.values():
            track.attribution.realised_pnl = sum(t.pnl for t in track.attribution.trades)
            track.attribution.commission = sum(
                t.commission for t in track.attribution.trades
            )

        provenance = self._merged_provenance(series_by_symbol)
        report = build_report(
            trades=trades,
            equity=portfolio.equity_series(),
            starting_equity=self.starting_equity,
            provenance=provenance,
            total_commission=portfolio.total_commission,
            per_trade_budget=(
                self.starting_equity * self.risk_settings.max_risk_per_trade_pct / 100.0
            ),
            notes=self._notes(series_by_symbol, warmup, contested),
        )

        merged_rejections: dict[str, int] = {}
        merged_risk: dict[str, int] = {}
        for track in tracks.values():
            for key, value in track.router.flatten_reason_counts().items():
                merged_rejections[key] = merged_rejections.get(key, 0) + value
            for key, value in track.attribution.risk_rejections.items():
                merged_risk[key] = merged_risk.get(key, 0) + value

        return PortfolioBacktestResult(
            report=report,
            portfolio=portfolio,
            trades=trades,
            attribution={s: t.attribution for s, t in tracks.items()},
            ict_states={s: t.state for s, t in tracks.items()},
            correlations=self.correlation_matrix(series_by_symbol),
            contested_slots=contested,
            rejections=merged_rejections,
            risk_rejections=dict(sorted(merged_risk.items(), key=lambda kv: -kv[1])),
        )

    # --- internals ----------------------------------------------------------

    def _step(
        self,
        track: _Track,
        index: int,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        risk: RiskManager,
        warmup: int,
    ) -> tuple[list[Trade], int]:
        """Advance one instrument by one bar. Mirrors `Backtester.run`'s body."""
        engine = track.engine
        bar = track.series.bar_at(index)
        track.last_close = bar.close
        closed: list[Trade] = []
        contested = 0

        for fill in track.venue.process_bar(bar):
            origin = track.orders_by_id.get(fill.order_id)
            realised = portfolio.apply_fill(
                track.series.instrument, fill.side, fill.quantity, fill.price,
                fill.commission, timestamp,
            )

            is_entry = track.pending_entry is not None and origin is track.pending_entry[0]
            if is_entry and track.pending_entry is not None:
                entry_signal = track.pending_entry[1]
                track.open_trade = _OpenTrade(
                    instrument=track.series.instrument,
                    entry_price=fill.price,
                    entry_time=timestamp,
                    quantity=fill.quantity,
                    side=fill.side,
                    risk_points=track.budgeted_risk_points.get(
                        origin.order_id if origin else -1, entry_signal.risk_points
                    ),
                    strategy=track.strategy.name,
                    tags=entry_signal.tags,
                    commission=fill.commission,
                    realised=realised,
                )
                engine._arm_bracket(
                    track.open_trade, entry_signal, track.router, track.venue,
                    track.orders_by_id, timestamp,
                )
                track.pending_entry = None
            elif track.open_trade is not None:
                track.open_trade.realised += realised
                track.open_trade.commission += fill.commission
                reason = engine._exit_reason(track.open_trade, origin)
                closed.append(engine._close_trade(track.open_trade, fill, timestamp, reason))
                risk.on_trade_closed(track.open_trade.realised)
                engine._cancel_siblings(track.open_trade, origin, track.venue)
                track.open_trade = None

        if (
            track.pending_entry is not None
            and track.pending_entry[0].status.is_terminal
            and track.pending_entry[0].filled_quantity == 0
        ):
            counts = track.attribution.risk_rejections
            counts["entry_expired_gapped"] = counts.get("entry_expired_gapped", 0) + 1
            track.pending_entry = None

        if index >= warmup and track.open_trade is None and track.pending_entry is None:
            context = StrategyContext(
                series=track.series,
                index=index,
                ict=track.state,
                portfolio=portfolio,
                timestamp=timestamp,
            )
            signal = track.strategy.evaluate(context)
            if signal is not None:
                track.attribution.signals.append(signal)
                before = dict(track.attribution.risk_rejections)
                order = engine._submit_entry(
                    signal, context, risk, track.router, portfolio, timestamp,
                    track.attribution.risk_rejections, track.budgeted_risk_points,
                )
                if order is not None:
                    track.orders_by_id[order.order_id] = order
                    track.pending_entry = (order, signal)
                else:
                    # A rejection that only exists because the shared book was
                    # already full is the portfolio effect this class exists to
                    # measure. Counted separately from ordinary rejections.
                    added = {
                        k for k, v in track.attribution.risk_rejections.items()
                        if v > before.get(k, 0)
                    }
                    if added & {"max_positions", "gross_exposure", "daily_loss_limit"}:
                        contested = 1

        return closed, contested

    @staticmethod
    def _merged_timeline(
        series_by_symbol: dict[str, OHLCVSeries],
    ) -> pd.DatetimeIndex:
        """Sorted union of every instrument's bar timestamps."""
        merged: pd.DatetimeIndex | None = None
        for series in series_by_symbol.values():
            merged = series.index if merged is None else merged.union(series.index)
        assert merged is not None
        return merged.sort_values()

    @staticmethod
    def _merged_provenance(series_by_symbol: dict[str, OHLCVSeries]) -> Provenance:
        """Combined provenance. One synthetic input taints the whole result.

        That is the correct behaviour: a portfolio curve built partly from
        generated bars is not evidence about anything, and `merge` enforces it.
        """
        provenance: Provenance | None = None
        for series in series_by_symbol.values():
            provenance = (
                series.provenance if provenance is None
                else provenance.merge(series.provenance)
            )
        assert provenance is not None
        return provenance

    @staticmethod
    def correlation_matrix(series_by_symbol: dict[str, OHLCVSeries]) -> pd.DataFrame:
        """Pearson correlation of close-to-close returns on the shared clock.

        Returns are aligned on the union of timestamps and forward-filled, then
        differenced. Instruments that never print together correlate on nothing
        and come back as NaN rather than a fabricated zero.
        """
        frame = pd.DataFrame(
            {
                symbol: pd.Series(series.closes, index=series.index)
                for symbol, series in series_by_symbol.items()
            }
        ).sort_index()
        returns = frame.ffill().pct_change(fill_method=None)
        return returns.corr()

    def _notes(
        self,
        series_by_symbol: dict[str, OHLCVSeries],
        warmup: int,
        contested: int,
    ) -> list[str]:
        notes = [
            f"Portfolio: {len(series_by_symbol)} instruments "
            f"({', '.join(sorted(series_by_symbol))}) on one shared risk budget.",
            f"Limits: max {self.risk_settings.max_positions} concurrent positions, "
            f"{self.risk_settings.max_risk_per_trade_pct}% risk per trade, "
            f"${self.risk_settings.daily_loss_limit_cash:,.0f} daily loss limit.",
            f"Costs applied to every fill: "
            f"{self.execution_settings.slippage_ticks} tick slippage, "
            f"${self.execution_settings.commission_per_unit}/unit commission.",
            f"Warmup: first {warmup} bars per instrument excluded from trading.",
        ]
        for symbol, series in sorted(series_by_symbol.items()):
            notes.append(f"Data · {symbol}: {series.describe()}")
        if contested:
            notes.append(
                f"{contested} signals were rejected because another instrument "
                f"already held the risk budget. Run separately, those would have "
                f"become trades — that difference is the portfolio effect."
            )
        notes.append(
            "Instruments printing on the same timestamp are processed in a fixed "
            "order, so that order decides who wins a contested last slot. This is "
            "a real bias in any portfolio backtest and cannot be removed."
        )
        return notes


def run_portfolio(
    strategies: Strategy | dict[str, Strategy],
    series_by_symbol: dict[str, OHLCVSeries],
    *,
    warmup: int = 50,
    **kwargs: Any,
) -> PortfolioBacktestResult:
    """Build a `PortfolioBacktester` and run it in one call."""
    return PortfolioBacktester(strategies, **kwargs).run(
        series_by_symbol, warmup=warmup
    )
