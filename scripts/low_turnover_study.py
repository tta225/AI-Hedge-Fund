"""The sixth campaign: signals that are slow because of what they measure.

Five campaigns have produced nothing that clears a deflated-Sharpe bar. The
turnover study was the exception in a narrow but specific way: it found that
*trading less* was worth roughly 0.45 Sharpe and about five times the capacity,
and — uniquely in this project — the finding replicated on a universe that had
played no part in selecting it.

That result has an implication the earlier campaigns did not act on. Every
signal tested so far was fast by construction and then slowed down afterwards
with hysteresis and rebalance spacing. If the cost structure genuinely favours
low turnover, the place to look is families that are slow *natively*, where a
quarterly holding period is the signal's own frequency rather than a constraint
imposed on it.

Five families, from :mod:`axiom.alpha.factors`: 12-1 momentum, residual
momentum, low volatility, Amihud illiquidity, and long-term reversal.

Three things about the design, each of which is there to stop a specific way
this study could flatter itself:

**Trial accounting is honest.** Every configuration evaluated counts, including
the ones added because an earlier one looked promising. The deflated-Sharpe bar
rises with the count, which is the mechanism that makes searching harder rather
than easier.

**The buffer and the cost model are fixed in advance** at the values the
turnover study settled on — ``HysteresisBands(0.2, 0.4)`` and
:class:`~axiom.execution.costs.SpreadImpactCost` — and are not tuned here.
Re-tuning execution mechanics inside a signal search is how a null result
becomes a positive one.

**Discovery and confirmation are separate universes.** The full campaign runs on
the discovery set; anything that survives is then re-run, unchanged, on 70
names that took no part in selecting it. A result that does not survive that
transfer is not a result.

With ``--pit-root`` the universes are point-in-time and contain the names that
stopped trading; without it they are the older fixed ticker lists, which are
survivorship-biased and make every figure an upper bound. The banner printed at
the top of a run says which was used, because the two are not comparable and
mislabelling them would be the most consequential error this script could make.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from axiom.alpha.base import AlphaAgent
from axiom.alpha.factors import (
    QUARTER,
    Illiquidity,
    LongHorizonMomentum,
    LongTermReversal,
    LowVolatility,
    ResidualMomentum,
)
from axiom.execution.costs import FlatBpsCost, SpreadImpactCost
from axiom.research.panel_lab import PanelCandidate, PanelLab, backtest_panel
from axiom.research.turnover import HysteresisBands

#: Frozen from the turnover study. Not re-tuned here — see the module docstring.
BANDS = HysteresisBands(0.2, 0.4)
TOP_FRACTION = 0.2
#: Long-term reversal needs three years of formation plus a year of skip, so the
#: warm-up is set by the slowest family rather than the fastest.
WARMUP = 800


def candidates() -> list[PanelCandidate]:
    """Every low-turnover family at a small number of horizons.

    Deliberately few configurations per family. A grid of twenty variants per
    family would raise the trial count — and therefore the bar — without adding
    structural diversity, because twenty lookbacks of the same factor are one
    hypothesis tested twenty times, not twenty hypotheses.

    Rebalance frequencies are quarterly and monthly. Nothing here is rebalanced
    every five days, because the whole premise is that these families do not
    need to be.
    """
    out: list[PanelCandidate] = []

    def add(name: str, factory: Any, params: dict[str, object], rebalance: int) -> None:
        out.append(
            PanelCandidate(
                name=name,
                factory=factory,
                params={**params, "rebal": rebalance},
                rebalance_every=rebalance,
                top_fraction=TOP_FRACTION,
            )
        )

    for rebalance in (QUARTER, 21):
        for skip in (21, 0):
            add(
                "mom12_1",
                lambda s=skip: [LongHorizonMomentum({"lookback": 252, "skip": s})],
                {"skip": skip},
                rebalance,
            )
        add(
            "residual_mom",
            lambda: [ResidualMomentum({"lookback": 252, "skip": 21})],
            {"lookback": 252},
            rebalance,
        )
        add(
            "low_vol",
            lambda: [LowVolatility({"lookback": 252})],
            {"lookback": 252},
            rebalance,
        )
        add(
            "illiquidity",
            lambda: [Illiquidity({"lookback": 252})],
            {"lookback": 252},
            rebalance,
        )
        add(
            "lt_reversal",
            lambda: [LongTermReversal({"lookback": 756, "skip": 252})],
            {"lookback": 756},
            rebalance,
        )

    return out


def turnover_profile(panel: Any, aum: float) -> dict[str, Any]:
    """What each family actually turns over, before anything is judged on return.

    Run first and reported separately, because it tests the study's *premise*
    rather than its hypothesis. If these families turn over 40× a year like the
    fast ones did, the entire rationale for the campaign is wrong and that
    should be visible before any Sharpe is quoted.
    """
    print(f"{'=' * 78}\n1. TURNOVER — is the premise true?\n{'=' * 78}")
    print(f"  {'family':<22}{'rebal':>7}{'turn/yr':>10}{'gross':>9}{'net':>9}{'cost%/yr':>10}")
    print("  " + "-" * 67)

    model = SpreadImpactCost()
    families: list[tuple[str, list[AlphaAgent], int]] = [
        ("mom12_1", [LongHorizonMomentum()], QUARTER),
        ("residual_mom", [ResidualMomentum()], QUARTER),
        ("low_vol", [LowVolatility()], QUARTER),
        ("illiquidity", [Illiquidity()], QUARTER),
        ("lt_reversal", [LongTermReversal()], QUARTER),
    ]
    rows: dict[str, Any] = {}
    for label, agents, rebalance in families:
        common: dict[str, Any] = {
            "start": WARMUP,
            "rebalance_every": rebalance,
            "top_fraction": TOP_FRACTION,
            "bands": BANDS,
        }
        net = backtest_panel(agents, panel, cost_model=model, aum=aum, **common)
        gross = backtest_panel(agents, panel, cost_model=FlatBpsCost(0.0), **common)
        cost_pct = (gross.returns.mean() - net.returns.mean()) * net.periods_per_year * 100
        rows[label] = {
            "turnover": net.annual_turnover,
            "gross_sharpe": gross.sharpe,
            "sharpe": net.sharpe,
            "cost_pct_yr": cost_pct,
        }
        print(
            f"  {label:<22}{rebalance:>7}{net.annual_turnover:>10.1f}"
            f"{gross.sharpe:>+9.3f}{net.sharpe:>+9.3f}{cost_pct:>10.2f}"
        )
    print()
    return rows


def pit_panels(root: str, top_n: int) -> tuple[Any, Any, str]:
    """Discovery and confirmation panels from the point-in-time cache.

    Split by liquidity rank rather than at random: the top ``top_n`` names form
    the discovery universe and the next ``top_n`` the confirmation set, so the
    two are disjoint and the second is not a resample of the first. Both are
    selected **as of the start of trading**, using only volume observed before
    it, and both retain the names that later delisted.
    """
    from axiom.alpha.panel import Panel
    from axiom.core.provenance import DataKind, Provenance
    from scripts.survivorship_study import load

    closes, volumes, universe = load(root)
    as_of = closes.index[min(WARMUP, len(closes) - 1)]
    provenance = Provenance(
        source="alpaca:pit", kind=DataKind.REAL, detail="point-in-time US equities"
    )

    first = universe.select_at(as_of, closes=closes, volumes=volumes, top_n=top_n)
    # Same size as discovery, and disjoint from it: excluding the first
    # selection and taking the next `top_n` yields ranks top_n+1..2*top_n.
    # Equal sizes matter because a Sharpe estimated on a wider cross-section is
    # not directly comparable to one estimated on a narrower one.
    second = universe.select_at(
        as_of, closes=closes, volumes=volumes, top_n=top_n, exclude=first.symbols
    )
    discovery = Panel.point_in_time(
        closes[list(first.symbols)], volumes[list(first.symbols)], universe, provenance
    )
    confirm = Panel.point_in_time(
        closes[list(second.symbols)], volumes[list(second.symbols)], universe, provenance
    )
    note = (
        f"Point-in-time universes selected as of {as_of.date()}: "
        f"{len(first.symbols)} discovery ({first.n_later_delisted} delisted within a "
        f"year), {len(second.symbols)} confirmation ({second.n_later_delisted}). "
        "Names that stopped trading are held while they lived and liquidated at "
        "their last close."
    )
    return discovery, confirm, note


def main() -> None:
    from scripts.cross_sectional_study import load_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/equities")
    parser.add_argument("--confirm-root", default="data/equities_confirm")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--aum", type=float, default=1e7)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--pit-root", default="",
        help="Run against the survivorship-free cache in this directory instead "
             "of the fixed ticker lists.",
    )
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    pit_confirm = None
    if args.pit_root:
        panel, pit_confirm, note = pit_panels(args.pit_root, args.top_n)
        print(note + "\n")
    else:
        panel = load_panel(args.data_root, args.timeframe)
    print(
        f"Discovery panel: {panel.n_symbols} symbols x {len(panel):,} bars "
        f"({panel.index[0].date()} -> {panel.index[-1].date()})\n"
        f"Warm-up {WARMUP} bars, set by long-term reversal's 756+252 formation.\n"
        f"Buffer {BANDS.entry_fraction}/{BANDS.exit_fraction} and the impact cost "
        "model are frozen from the turnover study, not tuned here.\n"
        + (
            "Universe is point-in-time: names that stopped trading are held while "
            "they lived\nand liquidated at their last close.\n"
            if args.pit_root
            else "Universes are survivorship-biased; every figure is an upper bound.\n"
        )
    )

    summary: dict[str, Any] = {"turnover": turnover_profile(panel, args.aum)}

    print(f"{'=' * 78}\n2. CAMPAIGN — walk-forward, discovery universe\n{'=' * 78}")
    lab = PanelLab(
        n_folds=args.folds,
        warmup=WARMUP,
        top_fraction=TOP_FRACTION,
        cost_model=SpreadImpactCost(),
        aum=args.aum,
        bands=BANDS,
    )
    trials = candidates()
    discovery = lab.search(panel, trials)
    print(discovery.render(top=12))
    summary["discovery_verdict"] = discovery.verdict
    summary["discovery_survivors"] = [r.candidate.label for r in discovery.survivors]

    # 3. Whatever survived, on names that did not select it. Runs even when
    # nothing survived, because the best candidate's behaviour out of universe
    # is informative regardless of whether it cleared a bar.
    n_confirm = pit_confirm.n_symbols if pit_confirm is not None else 70
    print(
        f"\n{'=' * 78}\n3. CONFIRMATION — same configs, {n_confirm} names that "
        f"selected nothing\n{'=' * 78}"
    )
    if pit_confirm is not None:
        confirm = pit_confirm
    else:
        try:
            confirm = load_panel(args.confirm_root, args.timeframe)
        except SystemExit:
            print("  No confirmation universe cached; skipping. Run scripts/cache_equities.py.")
            confirm = None

    if confirm is not None:
        ranked = sorted(
            (r for r in discovery.results if r.is_viable),
            key=lambda r: discovery.deflated.get(r.candidate.label, 0.0),
            reverse=True,
        )[:5]
        print(f"  {'candidate':<34}{'discovery':>11}{'confirm':>10}{'turn/yr':>10}")
        print("  " + "-" * 65)
        transfers: dict[str, Any] = {}
        for entry in ranked:
            candidate = entry.candidate
            agents = candidate.factory()
            outcome = backtest_panel(
                agents,
                confirm,
                start=WARMUP,
                rebalance_every=candidate.rebalance_every or QUARTER,
                top_fraction=candidate.top_fraction or TOP_FRACTION,
                bands=BANDS,
                cost_model=SpreadImpactCost(),
                aum=args.aum,
            )
            transfers[candidate.label] = {
                "discovery_sharpe": entry.result.sharpe,
                "confirm_sharpe": outcome.sharpe,
                "confirm_turnover": outcome.annual_turnover,
            }
            print(
                f"  {candidate.label:<34}{entry.result.sharpe:>+11.3f}"
                f"{outcome.sharpe:>+10.3f}{outcome.annual_turnover:>10.1f}"
            )
        summary["confirmation"] = transfers

        # Sign *agreement*, not "is the confirmation positive". Counting the
        # latter reports a candidate that was -0.37 on discovery and +0.12 on
        # confirmation as having survived, when it did the opposite of
        # surviving: it inverted. On a universe where most candidates are
        # negative, that error turns a table full of sign flips into an
        # apparent replication rate of 80%.
        agreed = [
            v
            for v in transfers.values()
            if np.sign(v["discovery_sharpe"]) == np.sign(v["confirm_sharpe"])
            and v["discovery_sharpe"] > 0
        ]
        flipped = [
            v
            for v in transfers.values()
            if np.sign(v["discovery_sharpe"]) != np.sign(v["confirm_sharpe"])
        ]
        summary["n_sign_agreed"] = len(agreed)
        summary["n_sign_flipped"] = len(flipped)
        print(
            f"\n  {len(agreed)} of {len(transfers)} were positive on discovery AND kept "
            f"that sign out of universe; {len(flipped)} inverted.\n"
            "  A sign that does not survive the transfer is not a finding, and a "
            "negative\n  candidate turning positive elsewhere is noise rather than "
            "a discovery."
        )

    print(
        f"\n{'=' * 78}\nWhat this campaign counted: {len(trials)} trials. The "
        f"deflated-Sharpe bar rises with that number,\nwhich is the point — a "
        "wider search has to clear a higher bar, not a lower one.\n" + "=" * 78
    )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
