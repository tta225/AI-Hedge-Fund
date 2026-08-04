"""Performance metrics.

Two rules this module exists to enforce:

* **costs are already in the numbers.** Every P&L here comes from simulated
  fills that paid commission and slippage. There is no "gross" variant to quote.
* **provenance travels with the result.** A :class:`PerformanceReport` built
  from synthetic bars renders a warning banner and answers False to
  :attr:`PerformanceReport.is_evidence`. Nothing here will present generated
  numbers as a track record.

Annualisation uses the observed bar spacing rather than an assumed 252 days, so
a 15-minute futures backtest is not silently scaled as if it were daily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from axiom.core.provenance import Provenance, provenance_warning
from axiom.core.types import Instrument, Side

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(slots=True)
class Trade:
    """A completed round trip."""

    instrument: Instrument
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    commission: float
    strategy: str = ""
    exit_reason: str = ""
    #: Planned risk in points at entry, for R-multiple accounting.
    risk_points: float = 0.0
    tags: tuple[str, ...] = ()

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def duration(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    @property
    def r_multiple(self) -> float | None:
        """P&L expressed in units of the risk taken. None if risk was unstated."""
        if self.risk_points <= 0:
            return None
        risk_cash = self.risk_points * self.quantity * self.instrument.point_value
        return self.pnl / risk_cash if risk_cash else None


@dataclass(slots=True)
class PerformanceReport:
    """Computed metrics plus the provenance that produced them."""

    provenance: Provenance
    starting_equity: float
    ending_equity: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_evidence(self) -> bool:
        """True only when these numbers came from real market data."""
        return self.provenance.is_evidential

    @property
    def warning(self) -> str | None:
        return provenance_warning(self.provenance)

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.metrics.get(key, default)

    def render(self) -> str:
        lines: list[str] = []
        banner = self.warning
        if banner:
            lines += [banner, ""]
        lines.append(f"Data: {self.provenance.describe()}")
        lines.append(
            f"Equity: ${self.starting_equity:,.2f} → ${self.ending_equity:,.2f} "
            f"({self.get('total_return_pct'):+.2f}%)"
        )
        lines.append("")
        order = [
            ("trades", "Trades", "{:.0f}"),
            ("win_rate_pct", "Win rate", "{:.1f}%"),
            ("profit_factor", "Profit factor", "{:.2f}"),
            ("expectancy", "Expectancy/trade", "${:,.2f}"),
            ("avg_r", "Average R", "{:+.2f}R"),
            ("avg_win", "Average win", "${:,.2f}"),
            ("avg_loss", "Average loss", "${:,.2f}"),
            ("max_drawdown_pct", "Max drawdown", "{:.2f}%"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("sortino", "Sortino", "{:.2f}"),
            ("calmar", "Calmar", "{:.2f}"),
            ("cagr_pct", "CAGR", "{:.2f}%"),
            ("total_commission", "Commission paid", "${:,.2f}"),
            ("worst_loss_vs_budget_x", "Worst loss vs budget", "{:.2f}x"),
        ]
        for key, label, fmt in order:
            if key in self.metrics:
                value = self.metrics[key]
                rendered = "n/a" if (isinstance(value, float) and math.isnan(value)) else fmt.format(value)
                lines.append(f"  {label:<20} {rendered}")
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return ``(peak_to_trough_cash, peak_to_trough_pct)``."""
    if equity.empty:
        return 0.0, 0.0
    values = equity.to_numpy(dtype="float64")
    peaks = np.maximum.accumulate(values)
    drawdowns = values - peaks
    i = int(np.argmin(drawdowns))
    cash = float(drawdowns[i])
    pct = float(cash / peaks[i] * 100.0) if peaks[i] else 0.0
    return abs(cash), abs(pct)


#: Daily observations required before a risk-adjusted ratio is reported at all.
MIN_DAILY_OBSERVATIONS = 20


def daily_returns(equity: pd.Series) -> tuple[pd.Series, float]:
    """Daily returns from an equity curve, plus observed periods per year.

    Sharpe and Sortino are computed on **daily** returns, never on the raw bar
    frequency. On a 15-minute curve almost every bar has exactly zero return
    (no position, or no mark change), which collapses the standard deviation
    and multiplies it by ``sqrt(35,000)`` — the arithmetic runs fine and
    produces a Sharpe in the tens, which is pure sampling artefact.

    Periods per year is measured from the curve's own span rather than assumed
    to be 252, so a 24/5 futures curve and an RTH equity curve are each
    annualised on their actual observation count.
    """
    if equity.empty:
        return pd.Series(dtype="float64"), float("nan")

    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    if len(returns) < 2:
        return returns, float("nan")

    span_years = (daily.index[-1] - daily.index[0]).total_seconds() / SECONDS_PER_YEAR
    periods = len(returns) / span_years if span_years > 0 else float("nan")
    return returns, periods


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe from daily returns. NaN when the sample is too short."""
    returns, periods = daily_returns(equity)
    if len(returns) < MIN_DAILY_OBSERVATIONS or not math.isfinite(periods):
        return float("nan")
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return float("nan")
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / sigma * math.sqrt(periods))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but penalising only downside deviation."""
    returns, periods = daily_returns(equity)
    if len(returns) < MIN_DAILY_OBSERVATIONS or not math.isfinite(periods):
        return float("nan")
    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / downside.std(ddof=1) * math.sqrt(periods))


def compute_metrics(
    trades: list[Trade],
    equity: pd.Series,
    starting_equity: float,
    total_commission: float = 0.0,
) -> dict[str, float]:
    """Compute the full metric set. Missing inputs yield NaN, never zero."""
    metrics: dict[str, float] = {}
    ending = float(equity.iloc[-1]) if not equity.empty else starting_equity

    metrics["starting_equity"] = starting_equity
    metrics["ending_equity"] = ending
    metrics["net_pnl"] = ending - starting_equity
    metrics["total_return_pct"] = (
        (ending / starting_equity - 1.0) * 100.0 if starting_equity else float("nan")
    )
    metrics["total_commission"] = total_commission

    dd_cash, dd_pct = max_drawdown(equity)
    metrics["max_drawdown"] = dd_cash
    metrics["max_drawdown_pct"] = dd_pct

    metrics["sharpe"] = sharpe_ratio(equity)
    metrics["sortino"] = sortino_ratio(equity)

    if len(equity) >= 2:
        span = (equity.index[-1] - equity.index[0]).total_seconds()
        years = span / SECONDS_PER_YEAR
        if years > 0 and starting_equity > 0 and ending > 0:
            cagr = (ending / starting_equity) ** (1 / years) - 1.0
            metrics["cagr_pct"] = cagr * 100.0
            metrics["calmar"] = (cagr * 100.0 / dd_pct) if dd_pct else float("nan")

    metrics["trades"] = float(len(trades))
    if not trades:
        for key in ("win_rate_pct", "profit_factor", "expectancy", "avg_win", "avg_loss", "avg_r"):
            metrics[key] = float("nan")
        return metrics

    pnls = np.array([t.pnl for t in trades], dtype="float64")
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    metrics["win_rate_pct"] = float(len(wins) / len(pnls) * 100.0)
    metrics["avg_win"] = float(wins.mean()) if wins.size else 0.0
    metrics["avg_loss"] = float(losses.mean()) if losses.size else 0.0
    metrics["expectancy"] = float(pnls.mean())
    metrics["largest_win"] = float(pnls.max())
    metrics["largest_loss"] = float(pnls.min())

    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = abs(float(losses.sum())) if losses.size else 0.0
    # Profit factor is undefined without a losing trade — reporting it as a
    # large finite number would flatter a sample that simply has not lost yet.
    metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("nan")
    metrics["gross_profit"] = gross_profit
    metrics["gross_loss"] = gross_loss

    r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
    metrics["avg_r"] = float(np.mean(r_values)) if r_values else float("nan")

    durations = [t.duration.total_seconds() for t in trades]
    metrics["avg_duration_hours"] = float(np.mean(durations) / 3600.0)

    return metrics


def build_report(
    trades: list[Trade],
    equity: pd.Series,
    starting_equity: float,
    provenance: Provenance,
    total_commission: float = 0.0,
    notes: list[str] | None = None,
    per_trade_budget: float | None = None,
) -> PerformanceReport:
    metrics = compute_metrics(trades, equity, starting_equity, total_commission)
    collected = list(notes or [])

    # Did any trade actually lose more than it was allowed to? Sizing budgets
    # for entry slippage and expected stop-gap, but a gap larger than the
    # allowance is real tail risk that no order type prevents. Surfacing the
    # worst observed breach keeps that visible instead of buried in the
    # average.
    if per_trade_budget and per_trade_budget > 0 and trades:
        worst = max(-(t.pnl + t.commission) for t in trades)
        overrun = worst / per_trade_budget
        metrics["per_trade_budget"] = per_trade_budget
        metrics["worst_loss"] = max(worst, 0.0)
        metrics["worst_loss_vs_budget_x"] = max(overrun, 0.0)
        if overrun > 1.05:
            collected.append(
                f"Worst trade lost ${worst:,.2f} against a ${per_trade_budget:,.2f} "
                f"per-trade budget ({overrun:.2f}x). The excess is stop-gap risk: "
                "the position was already open when price gapped through the stop, "
                "so no order type could have prevented it. Reduce it by raising "
                "stop_gap_atr (smaller size) or accept it as tail risk."
            )

    if len(trades) < 30:
        collected.append(
            f"Only {len(trades)} trades — too few for the win rate or profit factor "
            "to be statistically meaningful. Treat these as directional, not decisive."
        )
    return PerformanceReport(
        provenance=provenance,
        starting_equity=starting_equity,
        ending_equity=float(equity.iloc[-1]) if not equity.empty else starting_equity,
        trades=trades,
        equity_curve=equity,
        metrics=metrics,
        notes=collected,
    )
