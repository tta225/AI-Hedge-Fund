"""Render a report payload as one self-contained HTML file.

No network, no build step, no external assets: the output is a single file that
opens from disk, from an attachment, or from a static host, years after the run
that produced it.

The typographic rule below is the architecture made visible. Everything the
engines measured is set in the monospaced face with tabular figures; everything
a language model wrote is set in the prose face and carries a generated chip.
A reader can tell which is which from across the room, without reading a word —
which is the same guarantee :class:`~axiom.agents.base.AgentReport` makes in
code by keeping ``facts`` and ``narrative`` in separate fields.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

STAGE_TITLES: dict[str, tuple[str, str]] = {
    "research": ("Research", "What the tape is doing, from the structural engine."),
    "debate": ("Debate", "Both sides argued from identical measurements."),
    "backtest": ("Backtest", "Measured performance. Costs already deducted."),
    "risk": ("Risk", "Sizing against hard limits. Fails closed."),
    "review": ("Review", "Post-hoc scepticism, looking for the flattering artefact."),
}

#: Fact keys whose values are cash amounts. Suffix rules catch most fields; the
#: ones that read as plain nouns are listed so they don't render bare.
_CASH_KEYS = frozenset(
    {
        "equity", "expectancy", "avg_win", "avg_loss", "net_pnl", "gross_profit",
        "gross_loss", "largest_win", "largest_loss", "worst_loss", "entry", "stop",
        "target", "realised_today", "daily_loss_limit", "gross_exposure",
        "gross_exposure_cap", "starting_equity", "ending_equity", "max_drawdown",
        "per_trade_budget", "cash_at_risk", "total_commission", "last_price",
        "range_high", "range_low", "draw_above", "draw_below", "account_equity",
    }
)

_HEADLINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("total_return_pct", "Total return", "pct"),
    ("max_drawdown_pct", "Max drawdown", "pct"),
    ("profit_factor", "Profit factor", "ratio"),
    ("win_rate_pct", "Win rate", "pct"),
    ("avg_r", "Average R", "r"),
    ("sharpe", "Sharpe", "ratio"),
    ("trades", "Trades", "int"),
    ("total_commission", "Costs paid", "cash"),
)


_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?$")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _compact_timestamp(value: str) -> str:
    """``2026-08-10T22:55:23.267489+00:00`` → ``2026-08-10 22:55 UTC``.

    Microseconds on a fifteen-minute bar are noise, and the untruncated string
    is wide enough to force a column past the width of a phone.
    """
    match = _ISO.match(value)
    if not match:
        return value
    zone = match.group(5)
    suffix = " UTC" if zone in {"Z", "+00:00"} else (f" {zone}" if zone else "")
    return f"{match.group(1)} {match.group(2)}{suffix}"


def _fmt(value: Any, kind: str = "auto") -> str:
    """Format one measured value. ``None`` is always ``n/a``, never zero."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return _compact_timestamp(value) or "—"
    if kind == "pct":
        return f"{value:,.2f}%"
    if kind == "cash":
        return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "r":
        return f"{value:+.2f}R"
    if kind == "ratio":
        return f"{value:,.2f}"
    if kind == "x":
        return f"{value:,.2f}×"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.4g}"
    return f"{value:,}"


def _fact_kind(key: str) -> str:
    if key.endswith("_pct"):
        return "pct"
    if key.endswith("_x"):
        return "x"
    if key.endswith("_cash") or key in _CASH_KEYS:
        return "cash"
    if key in {"avg_r"} or key.endswith("_r"):
        return "r"
    return "auto"


def _label(key: str) -> str:
    return key.replace("_", " ")


def _sign_class(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    return " pos" if value > 0 else " neg" if value < 0 else ""


def _tone(key: str, value: Any, kind: str) -> str:
    """Green/red only where the sign genuinely means better or worse.

    A 3.85% win rate is not good news because it is positive, and colouring it
    green says otherwise. Cash and R-multiples carry their meaning in the sign;
    percentages only do when they are a return. Drawdown is a loss magnitude
    and is always the losing colour when non-zero.
    """
    if key.startswith("max_drawdown"):
        return " neg" if isinstance(value, (int, float)) and value else ""
    if kind in {"cash", "r"} or (kind == "pct" and "return" in key):
        return _sign_class(value)
    return ""


def _chip(text: str, tone: str = "") -> str:
    return f'<span class="chip {tone}">{_esc(text)}</span>'


def _meter(label: str, used: float, cap: float, detail: str, danger_at: float = 0.7) -> str:
    """A utilisation bar. Encodes state in form as well as number."""
    ratio = 0.0 if not cap else max(min(used / cap, 1.0), 0.0)
    tone = "critical" if ratio >= 0.9 else "warn" if ratio >= danger_at else "ok"
    return f"""
        <div class="meter">
          <div class="meter-head">
            <span class="meter-label">{_esc(label)}</span>
            <span class="meter-value num">{_esc(detail)}</span>
          </div>
          <div class="meter-track"><i class="meter-fill {tone}" style="width:{ratio * 100:.1f}%"></i></div>
        </div>"""


def _fact_row(key: str, value: Any) -> str:
    kind = _fact_kind(key)
    rendered = _fmt(value, kind)
    if isinstance(value, bool):
        cls = "num"
    elif isinstance(value, str):
        # Free-text facts — a rejection reason, a sizing rationale — are as long
        # as they need to be to name the limit that bit. Let those wrap rather
        # than forcing the numeric column to the width of the longest sentence;
        # short labels stay right-aligned with the values they sit among.
        cls = "text" if len(rendered) > 28 else "str"
    else:
        cls = "num" + _tone(key, value, kind)
    return (
        f'<tr><th scope="row">{_esc(_label(key))}</th>'
        f'<td class="{cls}">{_esc(rendered)}</td></tr>'
    )


def _facts_table(facts: dict[str, Any]) -> str:
    rows = [_fact_row(key, value) for key, value in facts.items()]
    if not rows:
        return '<p class="empty">no measured facts</p>'
    return f'<table class="facts"><tbody>{"".join(rows)}</tbody></table>'


def _facts_split(facts: dict[str, Any]) -> str:
    """Facts across two tables, for stages with no narrative to sit beside.

    Without a model the prose column is empty, and a single tall column of
    values next to whitespace reads as a rendering failure rather than as a
    deterministic run.
    """
    items = list(facts.items())
    if len(items) < 8:
        return f'<div class="facts-split">{_facts_table(facts)}</div>'
    half = (len(items) + 1) // 2
    left = _facts_table(dict(items[:half]))
    right = _facts_table(dict(items[half:]))
    return f'<div class="facts-split two">{left}{right}</div>'


def _warnings(items: list[str], title: str = "Concerns") -> str:
    if not items:
        return ""
    entries = "".join(f"<li>{_esc(w)}</li>" for w in items)
    return (
        f'<div class="concerns"><p class="concerns-title">{_esc(title)}</p>'
        f"<ul>{entries}</ul></div>"
    )


def _narrative(stage: dict[str, Any], deterministic_run: bool = False) -> str:
    if not stage.get("narrative"):
        # When the whole run was deterministic the page says so once, in the
        # panel header. Repeating it under all five stages turns a property of
        # the run into what looks like five separate failures.
        if deterministic_run:
            return ""
        return (
            '<p class="no-narrative">No model narrative for this stage. Every '
            "measured value above was still computed.</p>"
        )
    paragraphs = "".join(
        f"<p>{_esc(block)}</p>"
        for block in str(stage["narrative"]).split("\n\n")
        if block.strip()
    )
    return (
        f'<div class="narrative"><p class="narrative-tag">generated by '
        f'{_esc(stage.get("model") or "model")} · interpretation, not measurement</p>'
        f"{paragraphs}</div>"
    )


def _stage_section(index: int, stage: dict[str, Any], deterministic_run: bool = False) -> str:
    role = str(stage.get("role", ""))
    title, blurb = STAGE_TITLES.get(role, (role.title(), ""))
    concern_count = len(stage.get("warnings", []))
    badge = (
        f'<span class="stage-count">{concern_count} concern'
        f'{"s" if concern_count != 1 else ""}</span>'
        if concern_count
        else '<span class="stage-count clear">clear</span>'
    )
    facts = stage.get("facts", {})
    if stage.get("narrative"):
        body = f"""
        <div class="stage-body split">
          <div class="stage-facts">{_facts_table(facts)}</div>
          <div class="stage-prose">
            {_warnings(stage.get("warnings", []))}
            {_narrative(stage)}
          </div>
        </div>"""
    else:
        body = f"""
        <div class="stage-body">
          {_warnings(stage.get("warnings", []))}
          {_facts_split(facts)}
          {_narrative(stage, deterministic_run)}
        </div>"""

    return f"""
      <section class="stage" id="stage-{_esc(role)}">
        <header class="stage-head">
          <span class="stage-index num">{index:02d}</span>
          <div class="stage-title">
            <h3>{_esc(title)}</h3>
            <p>{_esc(blurb)}</p>
          </div>
          {badge}
        </header>
        {body}
      </section>"""


def _headline_tiles(metrics: dict[str, Any]) -> str:
    tiles = []
    for key, label, kind in _HEADLINE_METRICS:
        value = metrics.get(key)
        rendered = _fmt(value, kind)
        tone = _tone(key, value, kind)
        na = " na" if value is None else ""
        tiles.append(
            f'<div class="tile{na}"><p class="tile-label">{_esc(label)}</p>'
            f'<p class="tile-value num{tone}">{_esc(rendered)}</p></div>'
        )
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _risk_section(risk: dict[str, Any], stage: dict[str, Any] | None) -> str:
    snapshot = risk.get("snapshot", {})
    settings = risk.get("settings", {})

    equity = float(snapshot.get("equity") or 0.0)
    realised = float(snapshot.get("realised_today") or 0.0)
    limit_cash = float(settings.get("daily_loss_limit_cash") or 0.0)
    exposure = float(snapshot.get("gross_exposure") or 0.0)
    exposure_cap = float(snapshot.get("gross_exposure_cap") or 0.0)
    positions = float(snapshot.get("open_positions") or 0.0)
    max_positions = float(snapshot.get("max_positions") or 0.0)
    streak = float(snapshot.get("consecutive_losses") or 0.0)
    max_streak = float(snapshot.get("max_consecutive_losses") or 0.0)

    kill = bool(snapshot.get("kill_switch"))
    halted = str(snapshot.get("halted") or "")

    meters = "".join(
        [
            _meter(
                "Daily loss budget",
                abs(min(realised, 0.0)),
                limit_cash,
                f"${abs(min(realised, 0.0)):,.0f} of ${limit_cash:,.0f}",
            ),
            _meter(
                "Gross exposure",
                exposure,
                exposure_cap,
                f"${exposure:,.0f} of ${exposure_cap:,.0f}",
            ),
            _meter(
                "Open positions",
                positions,
                max_positions,
                f"{positions:,.0f} of {max_positions:,.0f}",
            ),
            _meter(
                "Loss streak",
                streak,
                max_streak,
                f"{streak:,.0f} of {max_streak:,.0f}",
                danger_at=0.5,
            ),
        ]
    )

    state_chips = "".join(
        [
            _chip(
                "kill switch engaged" if kill else "kill switch clear",
                "critical" if kill else "ok",
            ),
            _chip(str(snapshot.get("mode", "")), "neutral"),
            _chip(f"equity ${equity:,.0f}", "neutral"),
            _chip(halted, "critical") if halted else "",
        ]
    )

    decision = ""
    if stage is not None:
        facts = stage.get("facts", {})
        if "decision" in facts:
            approved = bool(facts.get("approved"))
            verdict = "approved" if approved else "rejected"
            decision = f"""
            <div class="decision {verdict}">
              <p class="decision-verdict">Sizing {_esc(verdict)}</p>
              <p class="decision-body num">{_esc(facts.get("decision", ""))}</p>
            </div>"""

    return f"""
      <section class="panel" id="risk">
        <header class="panel-head">
          <h2>Risk state</h2>
          <p>Limits are evaluated against the book, not against what the strategy
             believes it holds. Any check that cannot be evaluated blocks the trade.</p>
        </header>
        <div class="chips">{state_chips}</div>
        <div class="meters">{meters}</div>
        {decision}
      </section>"""


def _structure_section(structure: dict[str, Any]) -> str:
    dealing = structure.get("dealing_range")
    range_block = '<p class="empty">no dealing range established</p>'
    if dealing:
        position = float(dealing.get("position") or 0.0) * 100.0
        zone = str(dealing.get("zone", ""))
        range_block = f"""
          <div class="range">
            <div class="range-scale">
              <span class="num">{_fmt(dealing.get("low"), "cash")}</span>
              <span class="range-zone">{_esc(zone)}</span>
              <span class="num">{_fmt(dealing.get("high"), "cash")}</span>
            </div>
            <div class="range-track">
              <i class="range-eq"></i>
              <i class="range-marker" style="left:{max(min(position, 100.0), 0.0):.2f}%"></i>
            </div>
            <p class="range-caption">price at {position:.1f}% of range ·
               equilibrium {_fmt(dealing.get("equilibrium"), "cash")}</p>
          </div>"""

    pools = "".join(
        f'<tr><td><span class="side {_esc(pool["side"])}">{_esc(pool["side"])}</span></td>'
        f'<td class="num">{_fmt(pool.get("price"), "cash")}</td>'
        f'<td class="num{_sign_class(pool.get("distance"))}">{pool.get("distance", 0):+,.2f}</td>'
        f'<td class="num">{"equal highs/lows" if pool.get("equal_cluster") else ""}</td></tr>'
        for pool in structure.get("pools", [])
    ) or '<tr><td colspan="4" class="empty">no unswept pools</td></tr>'

    zones = "".join(
        f'<tr><td>{_esc(zone["kind"])}</td>'
        f'<td><span class="side {"buyside" if zone["direction"] == "bullish" else "sellside"}">'
        f'{_esc(zone["direction"])}</span></td>'
        f'<td class="num">{_fmt(zone.get("low"), "cash")} – {_fmt(zone.get("high"), "cash")}</td>'
        f'<td class="num">{_fmt(zone.get("distance"), "ratio")}</td></tr>'
        for zone in structure.get("zones", [])
    ) or '<tr><td colspan="4" class="empty">no live zones</td></tr>'

    counts = structure.get("counts", {})
    count_chips = "".join(
        _chip(f"{_label(key)} {value:,}", "neutral") for key, value in counts.items()
    )

    bias = str(structure.get("bias", "neutral"))
    return f"""
      <section class="panel" id="structure">
        <header class="panel-head">
          <h2>Market structure</h2>
          <p>The structural read the research and debate stages were handed.
             Every field is computed by the ICT engine.</p>
        </header>
        <div class="structure-grid">
          <div class="structure-primary">
            <p class="bias-label">Prevailing bias</p>
            <p class="bias {_esc(bias)}">{_esc(bias)}</p>
            <p class="bias-meta num">last {_fmt(structure.get("price"), "cash")} ·
               {_esc(structure.get("session", ""))} session ·
               confluence L {_fmt(structure.get("confluence_long"), "ratio")} /
               S {_fmt(structure.get("confluence_short"), "ratio")}</p>
            {range_block}
          </div>
          <div class="structure-tables">
            <div class="table-wrap">
              <p class="table-title">Draw on liquidity</p>
              <table class="data"><thead><tr><th>side</th><th>level</th>
                <th>distance</th><th></th></tr></thead><tbody>{pools}</tbody></table>
            </div>
            <div class="table-wrap">
              <p class="table-title">Order blocks &amp; imbalance</p>
              <table class="data"><thead><tr><th>type</th><th>direction</th>
                <th>zone</th><th>distance</th></tr></thead><tbody>{zones}</tbody></table>
            </div>
          </div>
        </div>
        <div class="chips wrap">{count_chips}</div>
      </section>"""


def _funnel(funnel: dict[str, Any]) -> str:
    steps = [
        ("signals proposed", funnel.get("signals", 0)),
        ("orders routed", funnel.get("orders_routed", 0)),
        ("trades completed", funnel.get("trades", 0)),
    ]
    bars = "".join(
        f'<div class="funnel-step"><span class="funnel-label">{_esc(label)}</span>'
        f'<span class="funnel-value num">{value:,}</span></div>'
        for label, value in steps
    )
    rejects = {**funnel.get("risk_rejections", {}), **funnel.get("router_rejections", {})}
    reject_rows = "".join(
        f'<tr><th scope="row">{_esc(_label(reason))}</th>'
        f'<td class="num">{count:,}×</td></tr>'
        for reason, count in sorted(rejects.items(), key=lambda kv: -kv[1])
    )
    reject_block = (
        f'<div class="table-wrap"><p class="table-title">Why signals did not become trades</p>'
        f'<table class="facts"><tbody>{reject_rows}</tbody></table></div>'
        if reject_rows
        else ""
    )
    return f'<div class="funnel">{bars}</div>{reject_block}'


def _trades_table(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return '<p class="empty">no completed trades</p>'
    rows = []
    for i, trade in enumerate(trades, start=1):
        won = bool(trade.get("is_win"))
        rows.append(
            f'<tr data-outcome="{"win" if won else "loss"}">'
            f'<td class="num">{i}</td>'
            f'<td><span class="side {"buyside" if trade["side"] == "buy" else "sellside"}">'
            f'{_esc(trade["side"])}</span></td>'
            f'<td class="num">{trade.get("quantity", 0):,.0f}</td>'
            f'<td class="num">{_fmt(trade.get("entry_price"), "cash")}</td>'
            f'<td class="num">{_fmt(trade.get("exit_price"), "cash")}</td>'
            f'<td class="num{" pos" if won else " neg"}">{_fmt(trade.get("pnl"), "cash")}</td>'
            f'<td class="num">{_fmt(trade.get("r_multiple"), "r")}</td>'
            f'<td class="num">{_fmt(trade.get("duration_hours"), "ratio")}h</td>'
            f'<td>{_esc(_label(str(trade.get("exit_reason", ""))))}</td>'
            f'<td class="stamp">{_esc(str(trade.get("exit_time", ""))[:16].replace("T", " "))}</td>'
            "</tr>"
        )
    return f"""
      <div class="filters" role="group" aria-label="Filter trades">
        <button type="button" class="filter is-active" data-filter="all">All</button>
        <button type="button" class="filter" data-filter="win">Winners</button>
        <button type="button" class="filter" data-filter="loss">Losers</button>
      </div>
      <div class="table-wrap scroll">
        <table class="data trades" id="trades">
          <thead><tr><th>#</th><th>side</th><th>qty</th><th>entry</th><th>exit</th>
            <th>P&amp;L</th><th>R</th><th>held</th><th>exit reason</th><th>closed</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </div>"""


def _approval_section(approval: dict[str, Any], evidential: bool) -> str:
    signal = approval.get("signal")
    if signal:
        rr = signal.get("reward_risk")
        candidate = f"""
          <dl class="candidate">
            <div><dt>direction</dt><dd class="num {_esc(signal["direction"])}">{_esc(signal["direction"])}</dd></div>
            <div><dt>entry</dt><dd class="num">{_fmt(signal.get("entry"), "cash")}</dd></div>
            <div><dt>stop</dt><dd class="num">{_fmt(signal.get("stop"), "cash")}</dd></div>
            <div><dt>target</dt><dd class="num">{_fmt(signal.get("primary_target"), "cash")}</dd></div>
            <div><dt>reward : risk</dt><dd class="num">{_fmt(rr, "ratio") if rr else "n/a"}</dd></div>
            <div><dt>confidence</dt><dd class="num">{_fmt(signal.get("confidence"), "ratio")}</dd></div>
          </dl>
          <p class="rationale">{_esc(signal.get("rationale", ""))}</p>"""
    else:
        candidate = '<p class="empty">No candidate signal. Nothing to approve.</p>'

    blocked = (
        '<p class="gate-block">Data is not market history. This candidate must '
        "not be acted on.</p>"
        if not evidential
        else ""
    )
    return f"""
      <section class="panel gate" id="approval">
        <header class="panel-head">
          <h2>Human approval required</h2>
          <p>The pipeline has no execute stage and no route to a venue. It
             terminates here, in a request a person acts on.</p>
        </header>
        {blocked}
        {candidate}
        <pre class="summary num">{_esc(approval.get("summary", ""))}</pre>
      </section>"""


def render_html(payload: dict[str, Any]) -> str:
    """Render a payload from :func:`~axiom.report.payload.build_payload`."""
    meta = payload.get("meta", {})
    provenance = payload.get("provenance", {})
    backtest = payload.get("backtest", {})
    metrics = backtest.get("metrics", {})
    stages = payload.get("stages", [])
    evidential = bool(provenance.get("is_evidential"))

    title = f"AXIOM · {meta.get('symbol', '')} {meta.get('timeframe', '')} · {meta.get('strategy', '')}"

    banner = ""
    if not evidential:
        banner = f"""
      <div class="banner" role="note">
        <p class="banner-title">{_esc(str(provenance.get("kind", "")).upper())} data — not market history</p>
        <p>Every number on this page was produced from generated bars
           ({_esc(provenance.get("label", ""))}). They demonstrate that the platform
           computes; they are not evidence of profitability and must not be read
           as a track record.</p>
      </div>"""

    nav = "".join(
        f'<a href="#{anchor}">{_esc(label)}</a>'
        for anchor, label in (
            ("summary", "Outcome"),
            ("pipeline", "Pipeline"),
            ("risk", "Risk"),
            ("structure", "Structure"),
            ("backtest", "Backtest"),
            ("approval", "Approval"),
        )
    )

    deterministic = not any(stage.get("narrative") for stage in stages)
    stage_html = "".join(
        _stage_section(i, stage, deterministic) for i, stage in enumerate(stages, start=1)
    )
    pipeline_blurb = (
        "Five stages, in order. Every value below was measured by the engines — "
        "no model ran on this report, so there is no generated prose to separate "
        "them from. The pipeline is complete without one."
        if deterministic
        else "Five stages, in order. Monospaced values were measured by the "
        "engines; prose was written by a model and is labelled as such. They "
        "are never merged."
    )
    risk_stage = next((s for s in stages if s.get("role") == "risk"), None)

    notes = "".join(f"<li>{_esc(note)}</li>" for note in backtest.get("notes", []))
    notes_block = (
        f'<div class="notes"><p class="table-title">Run conditions</p><ul>{notes}</ul></div>'
        if notes
        else ""
    )

    embedded = json.dumps(payload, allow_nan=False).replace("</", "<\\/")

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
<div class="shell">
  <header class="masthead">
    <div class="masthead-row">
      <div class="identity">
        <p class="wordmark">AXIOM</p>
        <p class="run">{_esc(meta.get("symbol", ""))} · {_esc(meta.get("timeframe", ""))}
           · {_esc(meta.get("strategy", ""))}</p>
      </div>
      <div class="masthead-meta num">
        <span>{_esc(f'{meta.get("bars", 0):,}')} bars</span>
        <span>{_esc(str(meta.get("start", ""))[:10])} → {_esc(str(meta.get("end", ""))[:10])}</span>
        {_chip(provenance.get("label", ""), "ok" if evidential else "warn")}
      </div>
    </div>
    <nav class="stage-nav">{nav}</nav>
  </header>
  {banner}

  <main>
    <section class="panel" id="summary">
      <header class="panel-head">
        <h2>Measured outcome</h2>
        <p>{_esc(meta.get("strategy", ""))} over {_esc(f'{meta.get("bars", 0):,}')}
           {_esc(meta.get("timeframe", ""))} bars of {_esc(meta.get("symbol", ""))}.
           Slippage and commission are already deducted from every figure. Fields
           the sample cannot support read <span class="num">n/a</span> rather than
           zero.</p>
      </header>
      {_headline_tiles(metrics)}
      <figure class="chart">
        <figcaption>Equity curve · ${_fmt(backtest.get("starting_equity"), "ratio")} start</figcaption>
        <svg id="equity-chart" viewBox="0 0 1000 280" preserveAspectRatio="none"
             role="img" aria-label="Equity curve over the backtest window"></svg>
        <p class="chart-readout num" id="chart-readout"></p>
      </figure>
      {notes_block}
    </section>

    <section class="panel" id="pipeline">
      <header class="panel-head">
        <h2>Agent pipeline</h2>
        <p>{_esc(pipeline_blurb)}</p>
      </header>
      <div class="stages">{stage_html}</div>
    </section>

    {_risk_section(payload.get("risk", {}), risk_stage)}
    {_structure_section(payload.get("structure", {}))}

    <section class="panel" id="backtest">
      <header class="panel-head">
        <h2>Backtest detail</h2>
        <p>Signal attrition and the completed round trips behind the headline
           metrics.</p>
      </header>
      {_funnel(backtest.get("funnel", {}))}
      {_trades_table(backtest.get("trades", []))}
    </section>

    {_approval_section(payload.get("approval", {}), evidential)}
  </main>

  <footer class="colophon">
    <p>Generated {_esc(payload.get("generated_at", ""))} · {_esc(provenance.get("describe", ""))}</p>
    <p>Paper and backtest only. No live routing is configured in this platform.</p>
  </footer>
</div>
<script type="application/json" id="payload">{embedded}</script>
<script>{_JS}</script>
"""


_CSS = """
:root {
  color-scheme: light dark;
  --ground: #f5f7f8;
  --surface: #ffffff;
  --surface-2: #eef1f3;
  --line: #d8dee2;
  --line-soft: #e7ecef;
  --ink: #0f1620;
  --ink-2: #4a5762;
  --ink-3: #78868f;
  --accent: #0d6f7c;
  --accent-soft: rgba(13, 111, 124, 0.12);
  --good: #1c7248;
  --bad: #a83527;
  --warn: #8a5d00;
  --warn-ground: #fbf1d8;
  --warn-line: #e8cd8e;
  --shadow: 0 1px 2px rgba(15, 22, 32, 0.06);
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0b1015;
    --surface: #121a21;
    --surface-2: #18222b;
    --line: #253039;
    --line-soft: #1c262f;
    --ink: #e6edf3;
    --ink-2: #a2b1bd;
    --ink-3: #6d7d8a;
    --accent: #3fb6c4;
    --accent-soft: rgba(63, 182, 196, 0.14);
    --good: #4eb884;
    --bad: #e07461;
    --warn: #e2b45f;
    --warn-ground: #241d0e;
    --warn-line: #4d3d18;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  }
}
:root[data-theme="dark"] {
  --ground: #0b1015;
  --surface: #121a21;
  --surface-2: #18222b;
  --line: #253039;
  --line-soft: #1c262f;
  --ink: #e6edf3;
  --ink-2: #a2b1bd;
  --ink-3: #6d7d8a;
  --accent: #3fb6c4;
  --accent-soft: rgba(63, 182, 196, 0.14);
  --good: #4eb884;
  --bad: #e07461;
  --warn: #e2b45f;
  --warn-ground: #241d0e;
  --warn-line: #4d3d18;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.pos { color: var(--good); }
.neg { color: var(--bad); }
.empty { color: var(--ink-3); font-style: italic; margin: 0; }

.shell { max-width: 1180px; margin: 0 auto; padding: 0 20px 72px; }

.masthead {
  position: sticky; top: 0; z-index: 20;
  background: var(--ground);
  border-bottom: 1px solid var(--line);
  padding: 18px 0 0;
}
.masthead-row { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline; justify-content: space-between; }
.identity { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.wordmark {
  margin: 0; font-family: var(--mono); font-size: 20px; font-weight: 600;
  letter-spacing: 0.24em; color: var(--ink);
}
.run { margin: 0; font-family: var(--mono); font-size: 13px; color: var(--ink-2); }
.masthead-meta { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-3); }
.stage-nav { display: flex; gap: 2px; flex-wrap: wrap; margin-top: 14px; }
.stage-nav a {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--ink-3); text-decoration: none; padding: 8px 12px;
  border-bottom: 2px solid transparent;
}
.stage-nav a:hover, .stage-nav a:focus-visible { color: var(--accent); border-bottom-color: var(--accent); }

.chip {
  display: inline-block; font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em;
  padding: 3px 8px; border-radius: 2px; border: 1px solid var(--line);
  background: var(--surface-2); color: var(--ink-2); white-space: nowrap;
}
.chip.ok { color: var(--good); border-color: color-mix(in srgb, var(--good) 40%, var(--line)); }
.chip.warn { color: var(--warn); background: var(--warn-ground); border-color: var(--warn-line); }
.chip.critical { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 45%, var(--line)); }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chips.wrap { margin-top: 18px; }

.banner {
  margin: 20px 0 0; padding: 14px 18px;
  background: var(--warn-ground); border: 1px solid var(--warn-line);
  border-left: 3px solid var(--warn);
}
.banner p { margin: 0; color: var(--ink-2); font-size: 14px; max-width: 78ch; }
.banner-title {
  font-family: var(--mono); font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--warn); margin-bottom: 6px !important;
}

main { display: flex; flex-direction: column; gap: 28px; margin-top: 28px; }
.panel {
  background: var(--surface); border: 1px solid var(--line);
  box-shadow: var(--shadow); padding: 26px 26px 30px;
  scroll-margin-top: 110px;
}
.panel-head { margin-bottom: 22px; }
.panel-head h2 {
  margin: 0 0 6px; font-size: 13px; font-family: var(--mono); font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent);
}
.panel-head p { margin: 0; color: var(--ink-2); max-width: 72ch; font-size: 14px; }

.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line-soft); border: 1px solid var(--line-soft); }
.tile { background: var(--surface); padding: 14px 16px; }
.tile-label { margin: 0 0 4px; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.tile-value { margin: 0; font-size: 24px; font-weight: 500; letter-spacing: -0.01em; }
.tile.na .tile-value { color: var(--ink-3); font-size: 20px; }

.chart { margin: 26px 0 0; }
.chart figcaption { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 8px; }
#equity-chart { width: 100%; height: 260px; display: block; background: var(--surface-2); border: 1px solid var(--line-soft); }
.chart-readout { margin: 8px 0 0; font-size: 12px; color: var(--ink-2); min-height: 1.4em; }

.notes { margin-top: 24px; }
.notes ul { margin: 0; padding-left: 18px; color: var(--ink-2); font-size: 13px; }
.notes li { margin-bottom: 4px; }
.table-title { margin: 0 0 8px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }

.stages { display: flex; flex-direction: column; }
.stage { border-top: 1px solid var(--line); padding: 22px 0; scroll-margin-top: 110px; }
.stage:first-child { border-top: none; padding-top: 0; }
.stage-head { display: flex; align-items: baseline; gap: 16px; }
.stage-index { color: var(--accent); font-size: 13px; letter-spacing: 0.08em; }
.stage-title { flex: 1; }
.stage-title h3 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
.stage-title p { margin: 2px 0 0; color: var(--ink-2); font-size: 13px; }
.stage-count {
  font-family: var(--mono); font-size: 11px; color: var(--bad); white-space: nowrap;
  border: 1px solid color-mix(in srgb, var(--bad) 35%, var(--line)); padding: 3px 8px;
}
.stage-count.clear { color: var(--good); border-color: color-mix(in srgb, var(--good) 35%, var(--line)); }
.stage-body { margin-top: 18px; }
.stage-body.split { display: grid; grid-template-columns: minmax(260px, 400px) 1fr; gap: 28px; }
.facts-split { display: grid; gap: 0 40px; }
.facts-split.two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }

table.facts { width: 100%; border-collapse: collapse; table-layout: fixed; }
table.facts th {
  text-align: left; font-weight: 400; color: var(--ink-3); font-size: 12px;
  padding: 5px 12px 5px 0; vertical-align: top; letter-spacing: 0.02em;
  width: 45%; word-break: break-word;
}
table.facts td.num { text-align: right; white-space: nowrap; }
table.facts td {
  padding: 5px 0; font-size: 13px; vertical-align: top;
  font-family: var(--mono); font-variant-numeric: tabular-nums;
}
table.facts td.text { text-align: left; white-space: normal; overflow-wrap: anywhere; color: var(--ink-2); font-size: 12.5px; }
table.facts td.str { text-align: right; white-space: normal; overflow-wrap: anywhere; }
table.facts tr + tr th, table.facts tr + tr td { border-top: 1px solid var(--line-soft); }

.concerns { border-left: 2px solid var(--warn); background: var(--warn-ground); padding: 12px 16px; margin-bottom: 18px; max-width: 92ch; }
.concerns-title { margin: 0 0 6px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--warn); }
.concerns ul { margin: 0; padding-left: 16px; }
.concerns li { color: var(--ink-2); font-size: 13.5px; margin-bottom: 5px; }

.narrative { max-width: 68ch; }
.narrative-tag { margin: 0 0 8px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; color: var(--ink-3); }
.narrative p + p { margin-top: 10px; }
.no-narrative { color: var(--ink-3); font-size: 13.5px; max-width: 62ch; margin: 0; }

.meters { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px 32px; margin-top: 20px; }
.meter-head { display: flex; flex-direction: column; gap: 2px; }
.meter-label { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.meter-value { font-size: 13px; color: var(--ink-2); }
.meter-track { margin-top: 6px; height: 6px; background: var(--surface-2); border: 1px solid var(--line-soft); }
.meter-fill { display: block; height: 100%; background: var(--accent); }
.meter-fill.warn { background: var(--warn); }
.meter-fill.critical { background: var(--bad); }

.decision { margin-top: 24px; border: 1px solid var(--line); border-left: 3px solid var(--good); padding: 14px 18px; background: var(--surface-2); }
.decision.rejected { border-left-color: var(--bad); }
.decision-verdict { margin: 0 0 4px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--good); }
.decision.rejected .decision-verdict { color: var(--bad); }
.decision-body { margin: 0; font-size: 13px; color: var(--ink-2); }

.structure-grid { display: grid; grid-template-columns: minmax(260px, 360px) 1fr; gap: 32px; }
.bias-label { margin: 0; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.bias { margin: 2px 0 6px; font-family: var(--mono); font-size: 30px; font-weight: 600; letter-spacing: -0.01em; text-transform: uppercase; }
.bias.bullish { color: var(--good); }
.bias.bearish { color: var(--bad); }
.bias.neutral { color: var(--ink-3); }
.bias-meta { margin: 0 0 20px; font-size: 12px; color: var(--ink-2); }
.range-scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; color: var(--ink-2); }
.range-zone { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); }
.range-track { position: relative; height: 10px; margin-top: 6px; background: var(--surface-2); border: 1px solid var(--line-soft); }
.range-eq { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--ink-3); }
.range-marker { position: absolute; top: -4px; width: 2px; height: 16px; background: var(--accent); }
.range-caption { margin: 8px 0 0; font-size: 12px; color: var(--ink-3); font-family: var(--mono); }

.structure-tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; }
table.data { width: 100%; border-collapse: collapse; font-size: 13px; }
table.data th {
  text-align: left; font-family: var(--mono); font-weight: 400; font-size: 11px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3);
  border-bottom: 1px solid var(--line); padding: 0 10px 6px 0;
}
table.data td { padding: 5px 10px 5px 0; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
table.data td.num, table.data th.num { text-align: right; }
.side { font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }
.side.buyside { color: var(--good); }
.side.sellside { color: var(--bad); }
.table-wrap { overflow-x: auto; }
/* Grid children default to min-width:auto, which lets a wide table stretch its
   track instead of scrolling inside it — and then the whole page scrolls
   sideways on a phone. */
.structure-grid > *, .structure-tables > *, .stage-body.split > *, .facts-split > * { min-width: 0; }
.table-wrap.scroll { max-height: 460px; overflow-y: auto; border: 1px solid var(--line-soft); }
.table-wrap.scroll table { min-width: 720px; }
.table-wrap.scroll thead th { position: sticky; top: 0; background: var(--surface); }
.trades td { padding-left: 12px; }
.stamp { font-family: var(--mono); font-size: 12px; color: var(--ink-3); }

.funnel { display: flex; gap: 1px; background: var(--line-soft); border: 1px solid var(--line-soft); margin-bottom: 24px; }
.funnel-step { flex: 1; background: var(--surface); padding: 12px 16px; display: flex; flex-direction: column; gap: 2px; }
.funnel-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.funnel-value { font-size: 22px; font-family: var(--mono); }

.filters { display: flex; gap: 6px; margin: 24px 0 10px; }
.filter {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 6px 12px; background: var(--surface-2); color: var(--ink-2);
  border: 1px solid var(--line); cursor: pointer;
}
.filter:hover { color: var(--accent); }
.filter.is-active { background: var(--accent-soft); color: var(--accent); border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }

.gate { border-left: 3px solid var(--accent); }
.gate-block { color: var(--bad); font-family: var(--mono); font-size: 13px; margin: 0 0 16px; }
.candidate { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1px; background: var(--line-soft); border: 1px solid var(--line-soft); margin: 0; }
.candidate > div { background: var(--surface); padding: 12px 16px; }
.candidate dt { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.candidate dd { margin: 2px 0 0; font-size: 18px; }
.candidate dd.bullish { color: var(--good); }
.candidate dd.bearish { color: var(--bad); }
.rationale { color: var(--ink-2); font-size: 14px; max-width: 76ch; margin: 16px 0 0; }
.summary { margin: 20px 0 0; padding: 14px 16px; background: var(--surface-2); border: 1px solid var(--line-soft); font-size: 12.5px; color: var(--ink-2); overflow-x: auto; white-space: pre-wrap; }

.colophon { margin-top: 36px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--ink-3); font-size: 12px; font-family: var(--mono); }
.colophon p { margin: 0 0 4px; }

a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

@media (max-width: 860px) {
  .stage-body.split, .structure-grid { grid-template-columns: 1fr; }
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .funnel { flex-direction: column; }
  .masthead { position: static; }
  .tile-value { font-size: 20px; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


_JS = """
(function () {
  var node = document.getElementById('payload');
  if (!node) return;
  var data = JSON.parse(node.textContent);

  var curve = (data.backtest && data.backtest.equity_curve) || [];
  var svg = document.getElementById('equity-chart');
  var readout = document.getElementById('chart-readout');

  function draw() {
    if (!svg || curve.length < 2) {
      if (readout) readout.textContent = 'Not enough equity observations to plot.';
      return;
    }
    var W = 1000, H = 280, padL = 8, padR = 8, padT = 14, padB = 14;
    var values = curve.map(function (p) { return p.equity; });
    var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    if (hi === lo) { hi = lo + 1; }
    var x = function (i) { return padL + (i / (curve.length - 1)) * (W - padL - padR); };
    var y = function (v) { return padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB); };

    var line = '', area = '';
    for (var i = 0; i < curve.length; i++) {
      line += (i ? 'L' : 'M') + x(i).toFixed(2) + ' ' + y(values[i]).toFixed(2) + ' ';
    }
    area = line + 'L' + x(curve.length - 1).toFixed(2) + ' ' + H + ' L' + padL + ' ' + H + ' Z';

    var start = values[0];
    var grid = '';
    [0.25, 0.5, 0.75].forEach(function (f) {
      var gy = (padT + f * (H - padT - padB)).toFixed(1);
      grid += '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + gy + '" y2="' + gy +
              '" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>';
    });
    var baseline = '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + y(start).toFixed(2) +
                   '" y2="' + y(start).toFixed(2) +
                   '" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="4 4" stroke-width="1"/>';

    svg.innerHTML =
      '<defs><linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="currentColor" stop-opacity="0.18"/>' +
      '<stop offset="100%" stop-color="currentColor" stop-opacity="0"/>' +
      '</linearGradient></defs>' +
      '<g color="var(--ink-3)">' + grid + baseline + '</g>' +
      '<path d="' + area + '" fill="url(#eqfill)" color="var(--accent)"/>' +
      '<path d="' + line + '" fill="none" stroke="var(--accent)" stroke-width="1.6" ' +
      'vector-effect="non-scaling-stroke" stroke-linejoin="round"/>' +
      '<circle cx="' + x(curve.length - 1).toFixed(2) + '" cy="' + y(values[values.length - 1]).toFixed(2) +
      '" r="3.5" fill="var(--accent)"/>' +
      '<line id="eq-cursor" x1="0" x2="0" y1="' + padT + '" y2="' + (H - padB) +
      '" stroke="var(--accent)" stroke-opacity="0" stroke-width="1"/>';

    var cursor = svg.querySelector('#eq-cursor');
    function fmtMoney(v) { return '$' + v.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
    readout.textContent = curve[0].t.slice(0, 10) + ' → ' + curve[curve.length - 1].t.slice(0, 10) +
      '  ·  ' + fmtMoney(start) + ' → ' + fmtMoney(values[values.length - 1]);

    svg.addEventListener('mousemove', function (event) {
      var box = svg.getBoundingClientRect();
      var ratio = (event.clientX - box.left) / box.width;
      var i = Math.max(0, Math.min(curve.length - 1, Math.round(ratio * (curve.length - 1))));
      cursor.setAttribute('x1', x(i).toFixed(2));
      cursor.setAttribute('x2', x(i).toFixed(2));
      cursor.setAttribute('stroke-opacity', '0.6');
      var change = ((values[i] / start) - 1) * 100;
      readout.textContent = curve[i].t.slice(0, 16).replace('T', ' ') + '  ·  ' +
        fmtMoney(values[i]) + '  ·  ' + (change >= 0 ? '+' : '') + change.toFixed(2) + '% from start';
    });
    svg.addEventListener('mouseleave', function () {
      cursor.setAttribute('stroke-opacity', '0');
      readout.textContent = curve[0].t.slice(0, 10) + ' → ' + curve[curve.length - 1].t.slice(0, 10) +
        '  ·  ' + fmtMoney(start) + ' → ' + fmtMoney(values[values.length - 1]);
    });
  }

  draw();

  var table = document.getElementById('trades');
  document.querySelectorAll('.filter').forEach(function (button) {
    button.addEventListener('click', function () {
      document.querySelectorAll('.filter').forEach(function (other) {
        other.classList.toggle('is-active', other === button);
      });
      var want = button.getAttribute('data-filter');
      if (!table) return;
      table.querySelectorAll('tbody tr').forEach(function (row) {
        var outcome = row.getAttribute('data-outcome');
        row.style.display = (want === 'all' || outcome === want) ? '' : 'none';
      });
    });
  });
})();
"""
