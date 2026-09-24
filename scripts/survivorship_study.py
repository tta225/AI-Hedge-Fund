"""How much were six campaigns of results flattered?

Every backtest in this project carried the sentence "the universe is
survivorship-biased, so every figure is an upper bound." That is an honest
caveat and a useless one, because it gives no idea whether the correction is
worth 0.05 Sharpe or 0.5.

This measures it. The design is a controlled comparison and the control is the
point:

**Arm A — point-in-time.** At each rebalance date the universe is the most
liquid names *trading on that date*, ranked by volume observed strictly before
it. Names that later delisted are included while they lived, and are liquidated
at their last close when they stop.

**Arm B — survivors only.** The identical procedure, over the identical dates,
with one change: names that stopped trading before the end of the sample are
deleted from history entirely. This reconstructs exactly what the old caches
were — a list of tickers liquid today, applied backward.

Everything else is held fixed: same strategy, same parameters, same cost model,
same buffer, same dates. The difference between the two arms is therefore
attributable to survivorship and nothing else, which is what makes it a
measurement rather than an argument.

The expected sign is unambiguous — deleting failures should flatter the result —
so the interesting quantity is the magnitude, and whether it is large enough to
change any conclusion reached so far.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from axiom.alpha.base import AlphaAgent
from axiom.alpha.factors import (
    QUARTER,
    Illiquidity,
    LongHorizonMomentum,
    LongTermReversal,
    LowVolatility,
    ResidualMomentum,
)
from axiom.alpha.panel import Panel
from axiom.core.provenance import DataKind, Provenance
from axiom.data.universe import PointInTimeUniverse
from axiom.execution.costs import FlatBpsCost, SpreadImpactCost
from axiom.research.panel_lab import backtest_panel
from axiom.research.turnover import HysteresisBands

BANDS = HysteresisBands(0.2, 0.4)
TOP_FRACTION = 0.2
WARMUP = 800

REAL = Provenance(
    source="alpaca:pit", kind=DataKind.REAL, detail="point-in-time US equities"
)


def load(root: str) -> tuple[pd.DataFrame, pd.DataFrame, PointInTimeUniverse]:
    path = Path(root)
    if not (path / "closes.csv.gz").exists():
        raise SystemExit(
            f"No point-in-time cache in {root}. Run scripts/cache_pit_universe.py first."
        )
    closes = pd.read_csv(path / "closes.csv.gz", index_col=0, parse_dates=True)
    volumes = pd.read_csv(path / "volumes.csv.gz", index_col=0, parse_dates=True)
    universe = PointInTimeUniverse.load(str(path / "listings.csv"))
    return closes, volumes, universe


def build_panel(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    universe: PointInTimeUniverse,
    *,
    as_of: pd.Timestamp,
    top_n: int,
    exclude: tuple[str, ...] = (),
) -> tuple[Panel, tuple[str, ...], int]:
    """A panel of the ``top_n`` most liquid names as of ``as_of``.

    One selection date rather than a rolling one, deliberately. A universe that
    re-selects every quarter is more realistic and confounds this particular
    experiment: membership changes would then differ between the two arms for
    reasons other than survivorship, and the comparison would no longer isolate
    the thing it is trying to measure.
    """
    snapshot = universe.select_at(
        as_of, closes=closes, volumes=volumes, top_n=top_n, exclude=exclude
    )
    members = list(snapshot.symbols)
    panel = Panel.point_in_time(closes[members], volumes[members], universe, REAL)
    return panel, snapshot.symbols, snapshot.n_later_delisted


def _families() -> list[tuple[str, list[AlphaAgent]]]:
    return [
        ("mom12_1", [LongHorizonMomentum()]),
        ("residual_mom", [ResidualMomentum()]),
        ("low_vol", [LowVolatility()]),
        ("illiquidity", [Illiquidity()]),
        ("lt_reversal", [LongTermReversal()]),
    ]


def run(panel: Panel, agents: list[AlphaAgent], aum: float) -> dict[str, float]:
    common: dict[str, Any] = {
        "start": WARMUP,
        "rebalance_every": QUARTER,
        "top_fraction": TOP_FRACTION,
        "bands": BANDS,
    }
    net = backtest_panel(agents, panel, cost_model=SpreadImpactCost(), aum=aum, **common)
    gross = backtest_panel(agents, panel, cost_model=FlatBpsCost(0.0), **common)
    return {
        "sharpe": net.sharpe,
        "gross_sharpe": gross.sharpe,
        "turnover": net.annual_turnover,
        "total_return_pct": net.total_return_pct,
        "max_drawdown_pct": net.max_drawdown_pct,
        "n_bars": int(net.returns.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pit")
    parser.add_argument("--aum", type=float, default=1e7)
    parser.add_argument(
        "--top-n", type=int, nargs="+", default=[100, 500, 1000],
        help="Universe breadths to compare. Attrition is near zero among the very "
             "largest names, so a single size cannot show the effect.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    closes, volumes, universe = load(args.root)
    print(
        f"Cache: {closes.shape[0]:,} bars x {closes.shape[1]:,} symbols "
        f"({closes.index[0].date()} -> {closes.index[-1].date()})\n"
        f"Listings: {len(universe):,}, of which {len(universe.delisted):,} stopped "
        "trading in sample.\n"
    )

    # 1. How much attrition is there to correct for at all?
    print(f"{'=' * 78}\n1. ATTRITION — what a today-only universe deletes\n{'=' * 78}")
    dates = [d for d in pd.date_range(closes.index[0], closes.index[-1], freq="YE")
             if d > closes.index[0] + pd.Timedelta(days=400)]
    # Reported at the widest breadth under test: attrition among the very
    # largest names is close to zero, so a narrow cut would make the table look
    # like there is nothing here to correct.
    widest = max(args.top_n)
    report = universe.survivorship_report(
        dates, closes=closes, volumes=volumes, top_n=widest
    )
    print(f"  (of the {widest} most liquid names on each date)")
    print(report.to_string(index=False))
    total_deleted = int(report["n_later_delisted"].sum())
    print(
        f"\n  Across these snapshots, {total_deleted} member-slots were occupied by "
        "names\n  that stopped trading within a year. A survivorship-biased "
        "universe contains\n  none of them, at any point in their history.\n"
    )

    # 2. The controlled comparison, at several universe breadths.
    #
    # One size cannot answer the question. The hundred most liquid US equities
    # almost never delist — mega-caps get acquired, not wound up — so a top-100
    # universe shows an attrition of roughly zero and would suggest survivorship
    # bias does not exist. Widen the universe and the failures appear. The
    # relationship between breadth and bias *is* the finding, so it is measured
    # rather than assumed at one arbitrary cut.
    as_of = closes.index[WARMUP] if len(closes) > WARMUP else closes.index[len(closes) // 2]
    survivors = universe.survivor_only(closes.index[-1])

    summary: dict[str, Any] = {
        "attrition": report.to_dict(orient="records"),
        "as_of": str(as_of.date()),
        "by_universe_size": {},
    }

    print(f"{'=' * 78}\n2. THE COMPARISON — same strategy, two universes\n{'=' * 78}")
    print(f"  Universe selected as of {as_of.date()}, by trailing dollar volume.\n")

    for top_n in args.top_n:
        pit_panel, members, _ = build_panel(
            closes, volumes, universe, as_of=as_of, top_n=top_n
        )
        kept = [s for s in members if s in survivors.listings]
        n_dead = len(members) - len(kept)
        if not kept or n_dead == 0:
            print(
                f"  top {top_n}: {len(members)} names, none later delisted — nothing "
                "to measure at this breadth.\n"
            )
            summary["by_universe_size"][str(top_n)] = {
                "n_members": len(members), "n_later_delisted": 0, "families": {}
            }
            continue

        biased_panel = Panel.point_in_time(closes[kept], volumes[kept], survivors, REAL)
        print(
            f"  top {top_n}: {len(members)} names, {n_dead} later delisted "
            f"({n_dead / len(members):.1%})"
        )
        print(
            f"  {'family':<16}{'PIT':>9}{'survivor':>10}{'bias':>9}"
            f"{'PIT gross':>11}{'turn':>8}"
        )
        print("  " + "-" * 63)

        entry: dict[str, Any] = {
            "n_members": len(members), "n_later_delisted": n_dead, "families": {}
        }
        biases = []
        for label, agents in _families():
            pit = run(pit_panel, agents, args.aum)
            biased = run(biased_panel, [type(a)(a.config) for a in agents], args.aum)
            bias = biased["sharpe"] - pit["sharpe"]
            biases.append(bias)
            entry["families"][label] = {"pit": pit, "survivor_only": biased, "bias": bias}
            print(
                f"  {label:<16}{pit['sharpe']:>+9.3f}{biased['sharpe']:>+10.3f}"
                f"{bias:>+9.3f}{pit['gross_sharpe']:>+11.3f}{pit['turnover']:>8.1f}"
            )
        finite = [b for b in biases if np.isfinite(b)]
        if finite:
            entry["mean_bias"] = float(np.mean(finite))
            print(f"  mean bias at this breadth: {np.mean(finite):+.3f} Sharpe\n")
        summary["by_universe_size"][str(top_n)] = entry

    all_biases = [
        family["bias"]
        for entry in summary["by_universe_size"].values()
        for family in entry["families"].values()
        if np.isfinite(family["bias"])
    ]
    if all_biases:
        mean_bias = float(np.mean(all_biases))
        summary["mean_bias"] = mean_bias
        print(
            f"  Mean survivorship bias across {len(all_biases)} "
            f"(family, breadth) pairs: {mean_bias:+.3f} Sharpe.\n"
            "  Positive means deleting the failures flattered the result, which is\n"
            "  the direction every earlier campaign's caveat was pointing at."
        )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
