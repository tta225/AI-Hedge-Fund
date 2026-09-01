"""Confirm one pre-specified hypothesis on a universe the search never saw.

The discovery run swept 22 cross-sectional candidates over 65 US large caps and
found nothing that cleared a 22-trial deflated-Sharpe bar of 1.63. The best was
12-month momentum rebalanced monthly, at a Sharpe of 0.48.

That number cannot be promoted by re-testing it as a single hypothesis on the
same data. **Selecting a configuration after looking and then claiming one
trial is the oldest way to manufacture significance**, and the deflated Sharpe
exists precisely to prevent it.

What *is* legitimate is testing the selected configuration on data that played
no part in selecting it. This script does that: the same rule, frozen, run on
70 different US large caps over the same window. Nothing about the
configuration is tuned here — a single trial, one number, and the bar it has to
clear is correspondingly low.

Two things this still cannot fix, and neither is a footnote:

*The window is shared.* Both universes span 2020-07 to 2026-08, so the same
macro regime drives both. This is a cross-sectional replication, not a
temporal one, and a result that holds only because both samples lived through
the same rate cycle is not a durable one.

*Both universes are survivorship-biased.* Names liquid today, applied backward.
The measured figure is an upper bound in both cases.

A confirmation that fails is decisive. A confirmation that passes is
suggestive and still owes an out-of-period test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from axiom.alpha.agents import MomentumAgent
from axiom.research.panel_lab import PanelCandidate, PanelLab, backtest_panel
from axiom.research.reference import assess_plausibility, sharpe_confidence_interval

#: Frozen before this universe was loaded. Taken from the discovery run, and
#: matching the published Time Series Momentum specification (12-month
#: formation, monthly rebalance) rather than being tuned to it.
LOOKBACK = 252
REBALANCE = 21
TOP_FRACTION = 0.2
COST_BPS = 10.0


def main() -> None:
    from scripts.cross_sectional_study import load_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-root", default="data/equities")
    parser.add_argument("--confirm-root", default="data/equities_confirm")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print(
        f"Hypothesis, frozen before loading the confirmation data:\n"
        f"  cross-sectional momentum, {LOOKBACK}-bar formation, rebalanced "
        f"every {REBALANCE} bars,\n"
        f"  dollar-neutral long-short on the top/bottom "
        f"{TOP_FRACTION:.0%}, {COST_BPS:.0f} bp per unit traded.\n"
    )

    summary: dict[str, Any] = {"lookback": LOOKBACK, "rebalance": REBALANCE}

    for label, root in (
        ("DISCOVERY (selected on this)", args.discovery_root),
        ("CONFIRMATION (never seen)", args.confirm_root),
    ):
        panel = load_panel(root, args.timeframe)
        lab = PanelLab(
            n_folds=args.folds,
            warmup=args.warmup,
            cost_bps=COST_BPS,
            top_fraction=TOP_FRACTION,
        )
        result = lab.search(
            panel,
            [
                PanelCandidate(
                    name="momentum",
                    factory=lambda: [MomentumAgent({"lookback": LOOKBACK})],
                    params={"lookback": LOOKBACK, "rebal": REBALANCE},
                    rebalance_every=REBALANCE,
                )
            ],
        )
        row = result.results[0]
        sharpe = row.result.sharpe
        observations = int(row.result.returns.size)
        low, high = sharpe_confidence_interval(sharpe, max(observations, 2))

        print(f"{'=' * 74}\n{label} — {panel.n_symbols} symbols\n{'=' * 74}")
        print(
            f"  Sharpe            {sharpe:+.3f}  "
            f"(95% CI [{low:+.2f}, {high:+.2f}] over {observations} obs)"
        )
        print(f"  total return      {row.result.total_return_pct:+.2f}%")
        print(f"  max drawdown      {row.result.max_drawdown_pct:.2f}%")
        print(f"  annual turnover   {row.result.annual_turnover:.1f}x")
        print(f"  cost drag         {row.result.annual_turnover * COST_BPS / 100:.2f}%/yr")
        print(f"  {assess_plausibility(sharpe, max(observations, 2)).verdict}")
        print()

        summary[label.split()[0].lower()] = {
            "symbols": panel.n_symbols,
            "sharpe": sharpe,
            "ci": [low, high],
            "observations": observations,
            "total_return_pct": row.result.total_return_pct,
            "max_drawdown_pct": row.result.max_drawdown_pct,
            "annual_turnover": row.result.annual_turnover,
        }

        # Cost sensitivity, on the confirmation set only. A result that exists
        # only at an optimistic cost assumption is a statement about the cost
        # model rather than about the market.
        if "CONFIRMATION" in label:
            print("  Cost sensitivity (confirmation universe):")
            curve: dict[str, float] = {}
            for bps in (0.0, 5.0, 10.0, 20.0, 40.0):
                outcome = backtest_panel(
                    [MomentumAgent({"lookback": LOOKBACK})],
                    panel,
                    start=args.warmup,
                    rebalance_every=REBALANCE,
                    cost_bps=bps,
                    top_fraction=TOP_FRACTION,
                )
                curve[f"{bps:g}bp"] = outcome.sharpe
                print(f"    {bps:>5.0f} bp  Sharpe {outcome.sharpe:+.3f}")
            summary["cost_sensitivity"] = curve
            print()

            # Phase sensitivity. A monthly rebalance over six years is only
            # ~58 decisions, and a strategy whose Sharpe swings with which day
            # of the month it trades has not found an effect, it has found a
            # particular sequence of coin flips. This diagnostic exists because
            # an earlier version of the harness anchored rebalance phase to
            # each fold's start rather than to the panel, and that alone moved
            # the measured Sharpe from 0.07 to 0.15.
            print("  Phase sensitivity (offset the rebalance day):")
            phases: dict[str, float] = {}
            for offset in range(0, REBALANCE, 3):
                outcome = backtest_panel(
                    [MomentumAgent({"lookback": LOOKBACK})],
                    panel,
                    start=args.warmup,
                    rebalance_every=REBALANCE,
                    cost_bps=COST_BPS,
                    top_fraction=TOP_FRACTION,
                    phase=offset,
                )
                phases[f"+{offset}"] = outcome.sharpe
                print(f"    offset {offset:>2}  Sharpe {outcome.sharpe:+.3f}")
            values = [v for v in phases.values() if np.isfinite(v)]
            spread = max(values) - min(values) if values else 0.0
            print(
                f"    spread {spread:.3f} across phases"
                + (
                    "  — larger than the effect itself; this is noise"
                    if values and spread > abs(np.mean(values))
                    else ""
                )
            )
            summary["phase_sensitivity"] = phases
            summary["phase_spread"] = spread
            print()

    discovery = summary["discovery"]["sharpe"]
    confirmation = summary["confirmation"]["sharpe"]
    holds = confirmation > 0 and confirmation > discovery * 0.5
    print("=" * 74)
    print(
        f"VERDICT: discovery {discovery:+.3f} vs confirmation {confirmation:+.3f} — "
        + (
            "the effect survives a universe it was not selected on."
            if holds
            else "the effect does not replicate on fresh names."
        )
    )
    if summary["confirmation"]["ci"][0] <= 0:
        print(
            "  The confirmation interval still includes zero, so this is "
            "consistent with the effect being real and also consistent with it "
            "being nothing. It is not yet evidence for trading."
        )
    summary["replicates"] = bool(holds)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
