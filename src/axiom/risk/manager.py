"""Risk management — the last gate before anything becomes an order.

Design rules:

* **fail closed.** Any check that cannot be evaluated blocks the trade. An
  unknown state is a rejection, never an approval.
* **every rejection is explained.** :class:`RiskDecision` carries the reasons in
  plain language so an operator reading the log knows precisely which limit bit.
* **the kill switch is checked first and cannot be overridden.** No argument, no
  mode, and no strategy can bypass it.
* **limits are evaluated against the portfolio's live state**, not against what
  the strategy believes it holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from axiom.core.config import RiskSettings
from axiom.core.types import Direction, Instrument, Side, TradingMode
from axiom.portfolio.positions import Portfolio
from axiom.risk.sizing import SizingResult, size_by_risk


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The verdict on a proposed trade."""

    approved: bool
    quantity: float
    reasons: tuple[str, ...] = ()
    sizing: SizingResult | None = None
    #: Stable codes for the reasons, so rejections aggregate across runs.
    #: The reasons carry the numbers; the codes carry the category.
    codes: tuple[str, ...] = ()

    @classmethod
    def reject(cls, *reasons: str, codes: tuple[str, ...] = ()) -> RiskDecision:
        return cls(approved=False, quantity=0.0, reasons=reasons, codes=codes)

    @classmethod
    def approve(cls, sizing: SizingResult, *notes: str) -> RiskDecision:
        return cls(
            approved=True, quantity=sizing.quantity, reasons=notes, sizing=sizing
        )

    def explain(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        body = "; ".join(self.reasons) if self.reasons else "no constraints binding"
        return f"{verdict} qty={self.quantity:g} — {body}"


@dataclass(slots=True)
class RiskState:
    """Mutable counters the limits are evaluated against."""

    consecutive_losses: int = 0
    trades_today: int = 0
    #: Trading day the counters refer to, so they reset on a new day.
    current_day: object | None = None
    #: Set when a limit has halted trading for the remainder of the day.
    halted_reason: str = ""
    closed_trade_pnls: list[float] = field(default_factory=list)

    @property
    def is_halted(self) -> bool:
        return bool(self.halted_reason)


class RiskManager:
    """Enforces sizing and the hard limits in :class:`RiskSettings`."""

    def __init__(
        self,
        settings: RiskSettings | None = None,
        *,
        mode: TradingMode = TradingMode.BACKTEST,
        kill_switch: bool = False,
    ) -> None:
        self.settings = settings or RiskSettings()
        self.mode = mode
        self._kill_switch = kill_switch
        self.state = RiskState()

    # --- kill switch --------------------------------------------------------

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    def engage_kill_switch(self, reason: str = "") -> None:
        """Halt all order generation. Deliberately has no automatic release."""
        self._kill_switch = True
        self.state.halted_reason = reason or "kill switch engaged"

    def release_kill_switch(self, confirmation: str) -> None:
        """Release the kill switch. Requires an explicit human confirmation."""
        if confirmation != "RELEASE":
            raise ValueError(
                "releasing the kill switch requires confirmation='RELEASE'"
            )
        self._kill_switch = False
        self.state.halted_reason = ""

    # --- daily lifecycle ----------------------------------------------------

    def roll_day(self, timestamp: pd.Timestamp) -> None:
        """Reset per-day counters when the New York trading day changes.

        The consecutive-loss counter resets here, making it a **daily circuit
        breaker** rather than a permanent one. Without this reset a strategy
        that takes its first four losses in a row is disabled for the remainder
        of the backtest, which silently truncates the sample and reports metrics
        from a handful of trades as though they were the whole run.

        The daily loss limit remains the real capital backstop; this counter
        only stops a bad session from compounding.
        """
        day = timestamp.tz_convert("America/New_York").normalize()
        if self.state.current_day is None:
            # First observation only establishes which day we are on. Clearing
            # counters here would discard any state recorded before the first
            # bar was seen, which silently erases a loss streak.
            self.state.current_day = day
            return
        if self.state.current_day != day:
            self.state.current_day = day
            self.state.trades_today = 0
            self.state.consecutive_losses = 0
            # A halt caused by a daily limit expires with the day; one caused by
            # the kill switch does not.
            if not self._kill_switch:
                self.state.halted_reason = ""

    def on_trade_closed(self, pnl: float) -> None:
        self.state.closed_trade_pnls.append(pnl)
        if pnl < 0:
            self.state.consecutive_losses += 1
        elif pnl > 0:
            self.state.consecutive_losses = 0

    # --- the gate -----------------------------------------------------------

    def evaluate(
        self,
        *,
        instrument: Instrument,
        direction: Direction,
        entry: float,
        stop: float,
        portfolio: Portfolio,
        timestamp: pd.Timestamp,
        risk_pct: float | None = None,
        entry_tolerance: float = 0.0,
        exit_gap_allowance: float = 0.0,
    ) -> RiskDecision:
        """Approve or reject a proposed entry, and size it if approved.

        Args:
            entry_tolerance: how far the fill may land adverse to ``entry``.
                Size is computed against the worst case, so the returned
                quantity keeps realised risk inside the budget even when the
                fill gaps. The caller must enforce the same bound at the venue
                via :attr:`~axiom.execution.base.Order.max_entry_deviation`.
            exit_gap_allowance: how far the protective stop is expected to fill
                beyond its trigger. Unlike the entry, this cannot be enforced —
                a stop gap happens while the position is already open — so it is
                budgeted for in size and any residual is reported.
        """
        if self._kill_switch:
            return RiskDecision.reject(
                "kill switch is engaged — no orders may be generated", codes=("kill_switch",)
            )
        if direction is Direction.NEUTRAL:
            return RiskDecision.reject("no directional bias; nothing to trade", codes=("no_direction",))

        self.roll_day(timestamp)
        settings = self.settings
        equity = portfolio.equity if portfolio.equity > 0 else settings.account_equity
        blocks: list[str] = []
        block_codes: list[str] = []

        if self.state.is_halted:
            blocks.append(f"trading halted: {self.state.halted_reason}")
            block_codes.append("halted")

        realised_today = portfolio.realised_on(timestamp)
        daily_limit = -abs(settings.daily_loss_limit_cash)
        if realised_today <= daily_limit:
            reason = (
                f"daily loss limit hit: ${realised_today:,.2f} realised today vs "
                f"${daily_limit:,.2f} limit"
            )
            self.state.halted_reason = reason
            blocks.append(reason)
            block_codes.append("daily_loss_limit")

        if self.state.consecutive_losses >= settings.max_consecutive_losses:
            blocks.append(
                f"{self.state.consecutive_losses} consecutive losses ≥ limit of "
                f"{settings.max_consecutive_losses}"
            )
            block_codes.append("consecutive_losses")

        open_count = len(portfolio.open_positions)
        already_open = not portfolio.position(instrument).is_flat
        if not already_open and open_count >= settings.max_positions:
            blocks.append(
                f"{open_count} open positions ≥ limit of {settings.max_positions}"
            )
            block_codes.append("max_positions")

        if blocks:
            return RiskDecision.reject(*blocks, codes=tuple(block_codes))

        # --- size ---
        pct = settings.max_risk_per_trade_pct if risk_pct is None else risk_pct
        if pct > settings.max_risk_per_trade_pct:
            return RiskDecision.reject(
                f"requested risk {pct}% exceeds per-trade cap of "
                f"{settings.max_risk_per_trade_pct}%"
            )

        # Never risk more than the remaining daily budget.
        budget = equity * pct / 100.0
        remaining_daily = settings.daily_loss_limit_cash + min(realised_today, 0.0)
        notes: list[str] = []
        if remaining_daily < budget:
            notes.append(
                f"risk trimmed to ${remaining_daily:,.2f} — the daily loss budget "
                f"remaining, below the ${budget:,.2f} per-trade allowance"
            )
            budget = remaining_daily
        if budget <= 0:
            return RiskDecision.reject("no daily loss budget remaining", codes=("daily_budget_exhausted",))

        # Stop must be on the correct side of entry for the intended direction.
        if direction is Direction.BULLISH and stop >= entry:
            return RiskDecision.reject(
                f"long stop {stop} is at or above entry {entry}"
            )
        if direction is Direction.BEARISH and stop <= entry:
            return RiskDecision.reject(
                f"short stop {stop} is at or below entry {entry}"
            )

        try:
            sizing = size_by_risk(
                instrument, entry, stop, budget,
                entry_tolerance=entry_tolerance,
                exit_gap_allowance=exit_gap_allowance,
            )
        except ValueError as exc:
            return RiskDecision.reject(str(exc), codes=("unsizable",))

        if not sizing.is_tradable:
            return RiskDecision.reject(sizing.rationale, codes=(sizing.code,))

        # --- exposure ---
        added = sizing.quantity * entry * instrument.point_value
        projected = portfolio.gross_exposure + added
        cap = equity * settings.max_gross_exposure_pct / 100.0
        if projected > cap:
            unit_notional = entry * instrument.point_value
            affordable = max(cap - portfolio.gross_exposure, 0.0)
            allowed = int(affordable / unit_notional)
            if allowed <= 0:
                # Distinguish "the book is full" from "one unit is too big to
                # ever fit" — they need completely different fixes, and a single
                # generic message sends you looking in the wrong place.
                if unit_notional > cap:
                    return RiskDecision.reject(
                        f"a single {instrument.symbol} unit carries "
                        f"${unit_notional:,.0f} notional, which alone exceeds the "
                        f"${cap:,.0f} gross exposure cap "
                        f"({settings.max_gross_exposure_pct:g}% of ${equity:,.0f} "
                        f"equity). This instrument is untradable at this account "
                        f"size — raise max_gross_exposure_pct or the account equity"
                    )
                return RiskDecision.reject(
                    f"gross exposure cap reached: ${portfolio.gross_exposure:,.0f} of "
                    f"${cap:,.0f} already used; no room for another "
                    f"${unit_notional:,.0f} unit"
                )
            notes.append(
                f"size cut {sizing.quantity:g} → {allowed} to stay inside the "
                f"${cap:,.0f} gross exposure cap"
            )
            sizing = SizingResult(
                quantity=float(allowed),
                risk_cash=allowed * sizing.risk_per_unit,
                risk_per_unit=sizing.risk_per_unit,
                stop_distance=sizing.stop_distance,
                rationale=sizing.rationale,
            )

        self.state.trades_today += 1
        return RiskDecision.approve(sizing, sizing.rationale, *notes)

    def side_for(self, direction: Direction) -> Side:
        return Side.from_direction(direction)

    def snapshot(self, portfolio: Portfolio, timestamp: pd.Timestamp) -> dict[str, object]:
        """Current risk utilisation — what the terminal's risk panel renders."""
        settings = self.settings
        equity = portfolio.equity if portfolio.equity > 0 else settings.account_equity
        realised = portfolio.realised_on(timestamp)
        return {
            "mode": self.mode.value,
            "kill_switch": self._kill_switch,
            "halted": self.state.halted_reason,
            "equity": equity,
            "realised_today": realised,
            "daily_loss_limit": -settings.daily_loss_limit_cash,
            "daily_budget_used_pct": (
                min(abs(min(realised, 0.0)) / settings.daily_loss_limit_cash * 100, 100.0)
                if settings.daily_loss_limit_cash
                else 0.0
            ),
            "open_positions": len(portfolio.open_positions),
            "max_positions": settings.max_positions,
            "gross_exposure": portfolio.gross_exposure,
            "gross_exposure_cap": equity * settings.max_gross_exposure_pct / 100.0,
            "consecutive_losses": self.state.consecutive_losses,
            "max_consecutive_losses": settings.max_consecutive_losses,
        }
