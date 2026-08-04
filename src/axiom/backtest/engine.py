"""Event-driven backtester.

Bar loop, in strict order:

1. **fill working orders** against the current bar. Orders were submitted on an
   earlier bar, so nothing fills on the bar that created it;
2. **apply fills** to the portfolio; entry fills arm an OCO bracket, exit fills
   close the trade and update the risk manager's loss streak;
3. **mark to market** on this bar's close;
4. **evaluate the strategy** using only features confirmed at or before this
   bar;
5. **risk-gate and route** any resulting signal, which becomes a working order
   for the *next* bar;
6. **record equity**.

The ICT engine runs exactly once over the whole series. That is safe because
every feature carries ``confirmed_index``, and :class:`StrategyContext` filters
on it — so a single pass gives per-bar answers without ever leaking the future,
and without re-analysing the tape N times.

Bracket handling: stop and target are submitted as an OCO pair. When both could
trigger on the same bar the stop is assumed to fill first. That is pessimistic
and occasionally wrong, but the alternative — assuming the target — inflates
results, and the whole point of the fill model is to err against the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from axiom.backtest.metrics import PerformanceReport, Trade, build_report
from axiom.core.config import ExecutionSettings, RiskSettings
from axiom.core.series import OHLCVSeries
from axiom.core.types import Instrument, OrderType, Side, TradingMode
from axiom.execution.base import Fill, Order
from axiom.execution.router import OrderRouter
from axiom.execution.simulated import FillModel, SimulatedVenue
from axiom.ict.engine import ICTConfig, ICTEngine
from axiom.ict.models import ICTState
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager
from axiom.strategy.base import Signal, Strategy, StrategyContext


@dataclass(slots=True)
class _OpenTrade:
    """Bookkeeping for a position between entry and exit."""

    instrument: Instrument
    entry_price: float
    entry_time: pd.Timestamp
    quantity: float
    side: Side
    risk_points: float
    strategy: str
    tags: tuple[str, ...]
    commission: float
    stop_order: Order | None = None
    target_order: Order | None = None
    realised: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    """Everything a run produced."""

    report: PerformanceReport
    portfolio: Portfolio
    trades: list[Trade]
    signals: list[Signal]
    router: OrderRouter
    ict_state: ICTState
    #: Router-level refusals (kill switch, mode), by reason.
    rejections: dict[str, int] = field(default_factory=dict)
    #: Risk-manager refusals, by reason. Usually the larger number — a strategy
    #: that "takes no trades" is far more often being declined by risk than
    #: failing to generate signals, and conflating the two wastes hours.
    risk_rejections: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return self.report.render()

    def funnel(self) -> str:
        """Signal → order → trade attrition, with the reasons."""
        lines = [
            f"signals generated : {len(self.signals)}",
            f"orders routed     : {self.router.routed_count} (incl. bracket legs)",
            f"trades completed  : {len(self.trades)}",
        ]
        if self.risk_rejections:
            lines.append("risk rejections:")
            lines += [f"  {n:>4} × {r}" for r, n in self.risk_rejections.items()]
        if self.rejections:
            lines.append("router rejections:")
            lines += [f"  {n:>4} × {r}" for r, n in self.rejections.items()]
        return "\n".join(lines)


class Backtester:
    """Runs one strategy over one series."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        risk_settings: RiskSettings | None = None,
        execution_settings: ExecutionSettings | None = None,
        ict_config: ICTConfig | None = None,
        starting_equity: float | None = None,
    ) -> None:
        self.strategy = strategy
        self.risk_settings = risk_settings or RiskSettings()
        self.execution_settings = execution_settings or ExecutionSettings()
        self.ict_config = ict_config or ICTConfig()
        self.starting_equity = starting_equity or self.risk_settings.account_equity

    def run(
        self, series: OHLCVSeries, *, correlate: OHLCVSeries | None = None, warmup: int = 50
    ) -> BacktestResult:
        """Execute the backtest.

        Args:
            warmup: bars skipped before trading begins, so ATR and swing
                detection have stabilised.
        """
        if len(series) <= warmup + 2:
            raise ValueError(
                f"series has {len(series)} bars, not enough for a {warmup}-bar warmup"
            )

        engine = ICTEngine(self.ict_config)
        state = engine.analyse(series, correlate=correlate)

        venue = SimulatedVenue(
            FillModel(
                slippage_ticks=self.execution_settings.slippage_ticks,
                commission_per_unit=self.execution_settings.commission_per_unit,
            )
        )
        risk = RiskManager(self.risk_settings, mode=TradingMode.BACKTEST)
        router = OrderRouter(venue=venue, risk=risk, mode=TradingMode.BACKTEST)
        portfolio = Portfolio(starting_cash=self.starting_equity)

        instrument = series.instrument
        trades: list[Trade] = []
        signals: list[Signal] = []
        orders_by_id: dict[int, Order] = {}
        open_trade: _OpenTrade | None = None
        pending_entry: tuple[Order, Signal] | None = None
        risk_rejections: dict[str, int] = {}

        provenance = series.provenance
        if correlate is not None:
            provenance = provenance.merge(correlate.provenance)

        for i in range(len(series)):
            bar = series.bar_at(i)
            timestamp = series.index[i]
            risk.roll_day(timestamp)

            for fill in venue.process_bar(bar):
                origin = orders_by_id.get(fill.order_id)
                realised = portfolio.apply_fill(
                    instrument, fill.side, fill.quantity, fill.price,
                    fill.commission, timestamp,
                )

                is_entry = (
                    pending_entry is not None and origin is pending_entry[0]
                )
                if is_entry and pending_entry is not None:
                    entry_signal = pending_entry[1]
                    open_trade = _OpenTrade(
                        instrument=instrument,
                        entry_price=fill.price,
                        entry_time=timestamp,
                        quantity=fill.quantity,
                        side=fill.side,
                        risk_points=entry_signal.risk_points,
                        strategy=self.strategy.name,
                        tags=entry_signal.tags,
                        commission=fill.commission,
                        realised=realised,
                    )
                    self._arm_bracket(
                        open_trade, entry_signal, router, venue, orders_by_id, timestamp
                    )
                    pending_entry = None
                elif open_trade is not None:
                    open_trade.realised += realised
                    open_trade.commission += fill.commission
                    reason = self._exit_reason(open_trade, origin)
                    trades.append(
                        self._close_trade(open_trade, fill, timestamp, reason)
                    )
                    risk.on_trade_closed(open_trade.realised)
                    self._cancel_siblings(open_trade, origin, venue)
                    open_trade = None

            portfolio.mark_to_market({instrument.symbol: bar.close})

            if i >= warmup and open_trade is None and pending_entry is None:
                context = StrategyContext(
                    series=series,
                    index=i,
                    ict=state,
                    portfolio=portfolio,
                    timestamp=timestamp,
                )
                signal = self.strategy.evaluate(context)
                if signal is not None:
                    signals.append(signal)
                    order = self._submit_entry(
                        signal, context, risk, router, portfolio, timestamp,
                        risk_rejections,
                    )
                    if order is not None:
                        orders_by_id[order.order_id] = order
                        pending_entry = (order, signal)

            portfolio.record(timestamp)

        report = build_report(
            trades=trades,
            equity=portfolio.equity_series(),
            starting_equity=self.starting_equity,
            provenance=provenance,
            total_commission=portfolio.total_commission,
            notes=self._notes(series, warmup),
        )
        return BacktestResult(
            report=report,
            portfolio=portfolio,
            trades=trades,
            signals=signals,
            router=router,
            ict_state=state,
            rejections=router.flatten_reason_counts(),
            risk_rejections=dict(
                sorted(risk_rejections.items(), key=lambda kv: -kv[1])
            ),
        )

    # --- internals ----------------------------------------------------------

    def _submit_entry(
        self,
        signal: Signal,
        context: StrategyContext,
        risk: RiskManager,
        router: OrderRouter,
        portfolio: Portfolio,
        timestamp: pd.Timestamp,
        risk_rejections: dict[str, int],
    ) -> Order | None:
        decision = risk.evaluate(
            instrument=context.instrument,
            direction=signal.direction,
            entry=signal.entry,
            stop=signal.stop,
            portfolio=portfolio,
            timestamp=timestamp,
        )
        if not decision.approved:
            for reason in decision.reasons or ("unspecified",):
                risk_rejections[reason] = risk_rejections.get(reason, 0) + 1
            return None

        order = Order(
            instrument=context.instrument,
            side=Side.from_direction(signal.direction),
            quantity=decision.quantity,
            order_type=OrderType.MARKET,
            strategy=self.strategy.name,
            stop_loss=signal.stop,
            take_profit=signal.primary_target,
            tag=",".join(signal.tags),
        )
        router.submit(order, timestamp)
        return order if order.is_open else None

    def _arm_bracket(
        self,
        trade: _OpenTrade,
        signal: Signal,
        router: OrderRouter,
        venue: SimulatedVenue,
        orders_by_id: dict[int, Order],
        timestamp: pd.Timestamp,
    ) -> None:
        """Submit the protective stop and the target as an OCO pair."""
        exit_side = trade.side.opposite

        stop_order = Order(
            instrument=trade.instrument,
            side=exit_side,
            quantity=trade.quantity,
            order_type=OrderType.STOP,
            stop_price=signal.stop,
            strategy=self.strategy.name,
            tag="protective_stop",
        )
        router.submit(stop_order, timestamp)
        orders_by_id[stop_order.order_id] = stop_order
        trade.stop_order = stop_order

        target = signal.primary_target
        if target is not None:
            target_order = Order(
                instrument=trade.instrument,
                side=exit_side,
                quantity=trade.quantity,
                order_type=OrderType.LIMIT,
                limit_price=target,
                strategy=self.strategy.name,
                tag="target",
            )
            router.submit(target_order, timestamp)
            orders_by_id[target_order.order_id] = target_order
            trade.target_order = target_order

    @staticmethod
    def _exit_reason(trade: _OpenTrade, origin: Order | None) -> str:
        if origin is None:
            return "unknown"
        if trade.stop_order is not None and origin is trade.stop_order:
            return "stop_loss"
        if trade.target_order is not None and origin is trade.target_order:
            return "take_profit"
        return origin.tag or "manual"

    @staticmethod
    def _cancel_siblings(
        trade: _OpenTrade, filled: Order | None, venue: SimulatedVenue
    ) -> None:
        """OCO: once one leg fills, cancel the other."""
        for leg in (trade.stop_order, trade.target_order):
            if leg is not None and leg is not filled and leg.is_open:
                venue.cancel(leg)

    @staticmethod
    def _close_trade(
        trade: _OpenTrade, fill: Fill, timestamp: pd.Timestamp, reason: str
    ) -> Trade:
        return Trade(
            instrument=fill.instrument,
            side=trade.side,
            quantity=trade.quantity,
            entry_price=trade.entry_price,
            exit_price=fill.price,
            entry_time=trade.entry_time,
            exit_time=timestamp,
            pnl=trade.realised,
            commission=trade.commission,
            strategy=trade.strategy,
            exit_reason=reason,
            risk_points=trade.risk_points,
            tags=trade.tags,
        )

    def _notes(self, series: OHLCVSeries, warmup: int) -> list[str]:
        return [
            f"Strategy: {self.strategy.describe()}",
            f"Data: {series.describe()}",
            (
                f"Costs applied to every fill: {self.execution_settings.slippage_ticks} tick "
                f"slippage, ${self.execution_settings.commission_per_unit}/unit commission."
            ),
            f"Warmup: first {warmup} bars excluded from trading.",
            (
                "Same-bar stop/target ambiguity resolved in favour of the stop "
                "(pessimistic)."
            ),
        ]
