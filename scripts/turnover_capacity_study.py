"""Does trading less, or costing better, rescue anything?

The cross-sectional campaign found one dominant pattern: every candidate above
30× annual turnover lost money, and the only positives turned 7× and 15×.
Direction was not the problem — the book was rebuilt too often. Two things
follow, and this measures both.

**Was the cost model wrong?** A flat 10 bp per unit traded is size-independent,
which is false in the direction that matters: it charges a thousand-dollar
trade and a hundred-million-dollar trade identically. The square-root impact
model in :mod:`axiom.execution.costs` uses the panel's own volume history, so
cost becomes a function of participation — and therefore of capital. If the
flat charge was simply too high for the sizes involved, the earlier nulls were
partly an artefact of the model.

**Can turnover be cut without cutting signal?** Rebalancing less often throws
away information along with the trading. Hysteresis does not: it holds a name
until it falls out of a wider band, suppressing the round trips that come from
a name moving from rank 20 to rank 21 and back.

Four experiments, in increasing order of what they can establish:

1. **Cost-model comparison** — the same book under flat and impact costs, so a
   change in result is attributable to one or the other.
2. **Turnover grid** — hysteresis widths × rebalance frequencies, measuring
   what each buys in turnover and costs in gross return.
3. **Capacity curve** — Sharpe against capital deployed. The question a
   size-independent model cannot ask.
4. **Full re-run** — the whole 22-candidate campaign under the better cost
   model and the best turnover configuration, so the trial accounting stays
   honest.

The universe remains survivorship-biased, so every figure is an upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from axiom.alpha.agents import MomentumAgent
from axiom.execution.costs import (
    PARTICIPATION_WARN,
    FlatBpsCost,
    SpreadImpactCost,
    average_dollar_volume,
)
from axiom.research.panel_lab import backtest_panel, capacity_curve
from axiom.research.turnover import HysteresisBands

#: The configuration the discovery run selected. Frozen, so this study varies
#: only cost and turnover mechanics and never re-tunes the signal.
LOOKBACK = 252
REBALANCE = 5
TOP_FRACTION = 0.2
WARMUP = 300


def _agents() -> list[MomentumAgent]:
    return [MomentumAgent({"lookback": LOOKBACK})]


def cost_model_comparison(panel: Any, aum: float) -> dict[str, Any]:
    """The same book priced two ways."""
    print(f"{'=' * 78}\n1. COST MODEL — same book, two models, ${aum:,.0f}\n{'=' * 78}")
    rows: dict[str, Any] = {}
    models = {
        "flat 10bp": FlatBpsCost(10.0),
        "flat 5bp": FlatBpsCost(5.0),
        "impact": SpreadImpactCost(),
    }
    for label, model in models.items():
        outcome = backtest_panel(
            _agents(), panel, start=WARMUP, rebalance_every=REBALANCE,
            top_fraction=TOP_FRACTION, cost_model=model, aum=aum,
        )
        gross = backtest_panel(
            _agents(), panel, start=WARMUP, rebalance_every=REBALANCE,
            top_fraction=TOP_FRACTION, cost_model=FlatBpsCost(0.0),
        )
        drag = gross.sharpe - outcome.sharpe
        rows[label] = {
            "sharpe": outcome.sharpe,
            "gross_sharpe": gross.sharpe,
            "sharpe_drag": drag,
            "turnover": outcome.annual_turnover,
        }
        print(
            f"  {label:<12} net {outcome.sharpe:+.3f}  gross {gross.sharpe:+.3f}  "
            f"drag {drag:.3f}  turnover {outcome.annual_turnover:.1f}x"
        )
    print()
    return rows


def turnover_grid(panel: Any, aum: float) -> dict[str, Any]:
    """Hysteresis widths against rebalance frequencies."""
    print(f"{'=' * 78}\n2. TURNOVER — hysteresis x rebalance, impact cost at "
          f"${aum:,.0f}\n{'=' * 78}")
    print(f"  {'config':<28}{'net':>8}{'gross':>8}{'turn/yr':>9}{'cost%/yr':>10}")
    print("  " + "-" * 63)

    model = SpreadImpactCost()
    rows: dict[str, Any] = {}
    configs: list[tuple[str, HysteresisBands | None, int]] = [
        ("no buffer, rebal 5", None, 5),
        ("no buffer, rebal 21", None, 21),
        ("buffer .2/.3, rebal 5", HysteresisBands(0.2, 0.3), 5),
        ("buffer .2/.4, rebal 5", HysteresisBands(0.2, 0.4), 5),
        ("buffer .2/.5, rebal 5", HysteresisBands(0.2, 0.5), 5),
        ("buffer .2/.4, rebal 1", HysteresisBands(0.2, 0.4), 1),
        ("buffer .2/.4, rebal 21", HysteresisBands(0.2, 0.4), 21),
        ("buffer .1/.3, rebal 5", HysteresisBands(0.1, 0.3), 5),
    ]
    for label, bands, rebalance in configs:
        common = {
            "start": WARMUP, "rebalance_every": rebalance,
            "top_fraction": TOP_FRACTION, "bands": bands,
        }
        net = backtest_panel(
            _agents(), panel, cost_model=model, aum=aum, **common  # type: ignore[arg-type]
        )
        gross = backtest_panel(
            _agents(), panel, cost_model=FlatBpsCost(0.0), **common  # type: ignore[arg-type]
        )
        # Cost drag in return terms, annualised on the panel's own frequency.
        cost_pct = (
            (gross.returns.mean() - net.returns.mean()) * net.periods_per_year * 100
        )
        rows[label] = {
            "sharpe": net.sharpe,
            "gross_sharpe": gross.sharpe,
            "turnover": net.annual_turnover,
            "cost_pct_yr": cost_pct,
        }
        print(
            f"  {label:<28}{net.sharpe:>+8.3f}{gross.sharpe:>+8.3f}"
            f"{net.annual_turnover:>9.1f}{cost_pct:>10.2f}"
        )
    print()
    return rows


def capacity(panel: Any, bands: HysteresisBands | None) -> dict[str, Any]:
    """Sharpe against capital deployed, with participation diagnostics."""
    label = "with buffer" if bands is not None else "no buffer"
    print(f"{'=' * 78}\n3. CAPACITY — Sharpe vs capital ({label})\n{'=' * 78}")
    print(f"  {'AUM':>14}{'net Sharpe':>13}{'cost%/yr':>11}{'max partic':>12}")
    print("  " + "-" * 50)

    aums = [1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
    curve = capacity_curve(
        _agents(), panel, aums=aums, start=WARMUP,
        rebalance_every=REBALANCE, top_fraction=TOP_FRACTION, bands=bands,
    )
    gross = backtest_panel(
        _agents(), panel, start=WARMUP, rebalance_every=REBALANCE,
        top_fraction=TOP_FRACTION, bands=bands, cost_model=FlatBpsCost(0.0),
    )

    model = SpreadImpactCost()
    adv = average_dollar_volume(panel.closes, panel.volumes, len(panel) - 1)
    rows: dict[str, Any] = {}
    crossing: float | None = None
    for aum, outcome in curve.items():
        cost_pct = (
            (gross.returns.mean() - outcome.returns.mean())
            * outcome.periods_per_year * 100
        )
        # Participation of a representative rebalance: the mean traded weight
        # spread over the names actually held.
        typical_delta = float(np.mean(outcome.turnover[outcome.turnover > 0]))
        held = max(int(panel.n_symbols * TOP_FRACTION * 2), 1)
        per_name = np.full(panel.n_symbols, typical_delta / held)
        max_participation = float(
            model.participation(per_name, aum=aum, adv_notional=adv).max()
        )
        rows[f"{aum:.0f}"] = {
            "sharpe": outcome.sharpe,
            "cost_pct_yr": cost_pct,
            "max_participation": max_participation,
        }
        if crossing is None and outcome.sharpe <= 0:
            crossing = aum
        flag = "  ! extrapolating" if max_participation > PARTICIPATION_WARN else ""
        print(
            f"  ${aum:>13,.0f}{outcome.sharpe:>+13.3f}{cost_pct:>11.2f}"
            f"{max_participation:>12.1%}{flag}"
        )

    print(
        f"\n  Gross Sharpe (no costs): {gross.sharpe:+.3f}"
        + (
            f"\n  Crosses zero between ${crossing:,.0f} and the level below it."
            if crossing
            else "\n  Still positive at the largest size tested."
        )
    )
    print()
    return {"curve": rows, "gross_sharpe": gross.sharpe, "zero_crossing": crossing}


def main() -> None:
    from scripts.cross_sectional_study import candidates, load_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/equities")
    parser.add_argument("--confirm-root", default="data/equities_confirm")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--aum", type=float, default=1e7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    panel = load_panel(args.data_root, args.timeframe)
    print(
        f"Panel: {panel.n_symbols} symbols x {len(panel):,} bars "
        f"({panel.index[0].date()} -> {panel.index[-1].date()})\n"
        f"Frozen signal: {LOOKBACK}-bar momentum, top/bottom "
        f"{TOP_FRACTION:.0%}. Only cost and turnover mechanics vary.\n"
        "Universe is survivorship-biased; every figure is an upper bound.\n"
    )

    summary: dict[str, Any] = {
        "costs": cost_model_comparison(panel, args.aum),
        "turnover": turnover_grid(panel, args.aum),
        "capacity_no_buffer": capacity(panel, None),
        "capacity_buffered": capacity(panel, HysteresisBands(0.2, 0.4)),
    }

    # 4. The full campaign again, under the better model, on the universe the
    # signal was NOT selected on. Trial accounting stays at 22.
    print(f"{'=' * 78}\n4. FULL CAMPAIGN — impact cost, buffered, confirmation "
          f"universe\n{'=' * 78}")
    from axiom.research.panel_lab import PanelLab

    confirm = load_panel(args.confirm_root, args.timeframe)
    lab = PanelLab(
        n_folds=4,
        warmup=WARMUP,
        top_fraction=TOP_FRACTION,
        cost_model=SpreadImpactCost(),
        aum=args.aum,
        bands=HysteresisBands(0.2, 0.4),
    )
    lab_result = lab.search(confirm, candidates())
    print(lab_result.render(top=10))
    summary["campaign_verdict"] = lab_result.verdict
    summary["campaign_survivors"] = [r.candidate.label for r in lab_result.survivors]

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
