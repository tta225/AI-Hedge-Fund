"""AXIOM Terminal — the operator view.

A Bloomberg-style multi-panel workspace rendered with Rich. Panels are built
from live platform objects, never from mock data, so what the terminal shows is
what the engines actually computed.

Design rule: every panel that displays numbers derived from a series also
displays that series' provenance. An operator must never be able to look at this
screen and be unsure whether they are seeing real market data.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction
from axiom.ict.engine import confluence_score
from axiom.ict.models import ICTState, LiquidityKind
from axiom.portfolio.positions import Portfolio
from axiom.risk.manager import RiskManager

ACCENT = "bright_cyan"
BULL = "bright_green"
BEAR = "bright_red"
MUTED = "grey58"


def _direction_style(direction: Direction) -> str:
    return {Direction.BULLISH: BULL, Direction.BEARISH: BEAR}.get(direction, MUTED)


@dataclass(slots=True)
class TerminalState:
    """Everything the terminal renders in one refresh."""

    series: OHLCVSeries
    ict: ICTState
    portfolio: Portfolio
    risk: RiskManager


def header_panel(state: TerminalState) -> Panel:
    series, ict = state.series, state.ict
    price = float(series.closes[ict.index])
    previous = float(series.closes[max(ict.index - 1, 0)])
    change = price - previous
    pct = (change / previous * 100.0) if previous else 0.0
    style = BULL if change >= 0 else BEAR

    title = Text()
    title.append("AXIOM", style=f"bold {ACCENT}")
    title.append("  │  ", style=MUTED)
    title.append(f"{series.instrument.symbol}", style="bold white")
    title.append(f"  {series.timeframe}  ", style=MUTED)
    title.append(f"{price:,.2f}", style=f"bold {style}")
    title.append(f"  {change:+,.2f} ({pct:+.2f}%)", style=style)
    title.append("   │   ", style=MUTED)
    title.append(f"{ict.session}", style=ACCENT)
    title.append(f"   {ict.as_of.strftime('%Y-%m-%d %H:%M')} UTC", style=MUTED)

    provenance = series.provenance
    banner = Text()
    if not provenance.is_evidential:
        banner.append(
            f" ⚠ {provenance.kind.value.upper()} DATA — NOT MARKET HISTORY ",
            style="bold black on yellow",
        )
    else:
        banner.append(f" {provenance.label} ", style=f"{MUTED}")

    return Panel(Group(title, banner), border_style=ACCENT, padding=(0, 1))


def structure_panel(state: TerminalState) -> Panel:
    ict = state.ict
    price = float(state.series.closes[ict.index])

    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, justify="right")
    table.add_column()

    table.add_row("bias", Text(str(ict.bias).upper(), style=_direction_style(ict.bias)))

    last = ict.last_structure_event
    if last is not None:
        table.add_row(
            "last event",
            Text(
                f"{last.event_type.value.upper()} {last.direction} @ {last.level:,.2f} "
                f"({ict.index - last.confirmed_index} bars ago)",
                style=_direction_style(last.direction),
            ),
        )

    dealing = ict.dealing_range
    if dealing is not None:
        position = dealing.position_of(price)
        zone = "PREMIUM" if position > 0.5 else "DISCOUNT"
        table.add_row("range", f"{dealing.low:,.2f} — {dealing.high:,.2f}")
        table.add_row("equilibrium", f"{dealing.equilibrium:,.2f}")
        table.add_row(
            "position",
            Text(f"{position:.1%}  {zone}", style=BEAR if position > 0.5 else BULL),
        )
        ote_low, ote_high = dealing.optimal_trade_entry(Direction.BULLISH)
        table.add_row("long OTE", f"{ote_low:,.2f} — {ote_high:,.2f}")

    table.add_row("", "")
    table.add_row(
        "confluence L/S",
        Text(
            f"{confluence_score(ict, price, Direction.BULLISH):.2f}  /  "
            f"{confluence_score(ict, price, Direction.BEARISH):.2f}"
        ),
    )
    return Panel(table, title="MARKET STRUCTURE", border_style=ACCENT, title_align="left")


def liquidity_panel(state: TerminalState, limit: int = 8) -> Panel:
    ict = state.ict
    price = float(state.series.closes[ict.index])

    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("side", style=MUTED, width=9)
    table.add_column("level", justify="right")
    table.add_column("dist", justify="right", style=MUTED)
    table.add_column("eq", justify="center", width=3)

    pools = sorted(ict.unswept_pools(), key=lambda p: abs(p.price - price))[:limit]
    for pool in sorted(pools, key=lambda p: -p.price):
        buyside = pool.kind is LiquidityKind.BUYSIDE
        table.add_row(
            Text("BUYSIDE" if buyside else "SELLSIDE", style=BULL if buyside else BEAR),
            f"{pool.price:,.2f}",
            f"{pool.price - price:+,.2f}",
            "◆" if pool.is_equal_cluster else "",
        )
    if not pools:
        table.add_row(Text("no unswept pools", style=MUTED), "", "", "")

    return Panel(table, title="DRAW ON LIQUIDITY", border_style=ACCENT, title_align="left")


def zones_panel(state: TerminalState, limit: int = 8) -> Panel:
    ict = state.ict
    price = float(state.series.closes[ict.index])

    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("type", style=MUTED, width=9)
    table.add_column("dir", width=4)
    table.add_column("zone", justify="right")
    table.add_column("dist", justify="right", style=MUTED)

    rows: list[tuple[str, Direction, float, float, float]] = []
    for block in ict.active_order_blocks():
        rows.append(("OB", block.effective_direction, block.bottom, block.top,
                     abs(block.midpoint - price)))
    for gap in ict.active_fvgs():
        rows.append(("FVG", gap.direction, gap.bottom, gap.top,
                     abs(gap.midpoint - price)))

    for kind, direction, low, high, distance in sorted(rows, key=lambda r: r[4])[:limit]:
        table.add_row(
            kind,
            Text("▲" if direction is Direction.BULLISH else "▼",
                 style=_direction_style(direction)),
            f"{low:,.2f}–{high:,.2f}",
            f"{distance:,.2f}",
        )
    if not rows:
        table.add_row(Text("no live zones", style=MUTED), "", "", "")

    return Panel(table, title="ORDER BLOCKS / IMBALANCE", border_style=ACCENT,
                 title_align="left")


def risk_panel(state: TerminalState) -> Panel:
    snapshot = state.risk.snapshot(state.portfolio, state.ict.as_of)

    def number(key: str) -> float:
        """Read a numeric field from the risk snapshot.

        The snapshot is deliberately ``dict[str, object]`` — it mixes floats,
        ints, bools and strings — so numeric fields are narrowed here rather
        than loosening the snapshot's type for the sake of one renderer.
        """
        value = snapshot[key]
        assert isinstance(value, (int, float)), f"{key} is not numeric: {value!r}"
        return float(value)

    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, justify="right")
    table.add_column()

    kill = bool(snapshot["kill_switch"])
    table.add_row(
        "kill switch",
        Text("ENGAGED" if kill else "clear", style="bold white on red" if kill else BULL),
    )
    table.add_row("mode", Text(str(snapshot["mode"]).upper(), style=ACCENT))
    table.add_row("equity", f"${number('equity'):,.2f}")

    realised = number("realised_today")
    table.add_row(
        "realised today",
        Text(f"${realised:,.2f}", style=BULL if realised >= 0 else BEAR),
    )
    used = number("daily_budget_used_pct")
    table.add_row(
        "daily loss budget",
        Text(f"{used:.0f}% used", style=BEAR if used > 70 else MUTED),
    )
    table.add_row(
        "positions",
        f"{snapshot['open_positions']} / {snapshot['max_positions']}",
    )
    table.add_row(
        "gross exposure",
        f"${number('gross_exposure'):,.0f} / "
        f"${number('gross_exposure_cap'):,.0f}",
    )
    table.add_row(
        "loss streak",
        f"{snapshot['consecutive_losses']} / {snapshot['max_consecutive_losses']}",
    )
    if snapshot["halted"]:
        table.add_row("halted", Text(str(snapshot["halted"]), style=BEAR))

    return Panel(table, title="RISK", border_style=BEAR if kill else ACCENT,
                 title_align="left")


def positions_panel(state: TerminalState) -> Panel:
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("symbol", style="bold")
    table.add_column("qty", justify="right")
    table.add_column("avg", justify="right")
    table.add_column("last", justify="right")
    table.add_column("unrealised", justify="right")

    positions = state.portfolio.open_positions
    for position in positions:
        pnl = position.unrealised_pnl()
        table.add_row(
            position.instrument.symbol,
            Text(f"{position.quantity:+g}",
                 style=BULL if position.is_long else BEAR),
            f"{position.average_price:,.2f}",
            f"{position.last_price:,.2f}",
            Text(f"${pnl:,.2f}", style=BULL if pnl >= 0 else BEAR),
        )
    if not positions:
        table.add_row(Text("flat", style=MUTED), "", "", "", "")

    return Panel(table, title="POSITIONS", border_style=ACCENT, title_align="left")


def sparkline(series: OHLCVSeries, width: int = 60, bars: int = 120) -> Text:
    """A compact close-price sparkline for the header strip."""
    blocks = "▁▂▃▄▅▆▇█"
    closes = series.closes[-bars:]
    if len(closes) < 2:
        return Text("")
    step = max(len(closes) // width, 1)
    sampled = closes[::step][-width:]
    low, high = float(sampled.min()), float(sampled.max())
    span = high - low or 1.0
    text = Text()
    for i, value in enumerate(sampled):
        level = int((value - low) / span * (len(blocks) - 1))
        rising = i == 0 or value >= sampled[i - 1]
        text.append(blocks[level], style=BULL if rising else BEAR)
    return text


def build_layout(state: TerminalState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="chart", size=3),
        Layout(name="body"),
        Layout(name="footer", size=1),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split_column(Layout(name="structure"), Layout(name="risk"))
    layout["right"].split_column(Layout(name="liquidity"), Layout(name="zones"),
                                 Layout(name="positions"))

    layout["header"].update(header_panel(state))
    layout["chart"].update(
        Panel(Align.center(sparkline(state.series)), border_style=MUTED,
              title="PRICE", title_align="left")
    )
    layout["structure"].update(structure_panel(state))
    layout["risk"].update(risk_panel(state))
    layout["liquidity"].update(liquidity_panel(state))
    layout["zones"].update(zones_panel(state))
    layout["positions"].update(positions_panel(state))
    layout["footer"].update(
        Align.center(
            Text(
                f"{state.series.describe()}   ·   paper/backtest only — "
                "no live routing configured",
                style=MUTED,
            )
        )
    )
    return layout


def render(state: TerminalState, console: Console | None = None) -> None:
    """Draw one frame of the terminal."""
    (console or Console()).print(build_layout(state))
