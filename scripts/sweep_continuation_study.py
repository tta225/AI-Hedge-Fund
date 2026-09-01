"""Walk-forward test of the sweep-continuation reading, against its control.

``docs/REAL_DATA_FINDINGS.md`` measured a directional tilt: after a liquidity
sweep, price continued with the raid more often than it reversed. That is a base
rate over a fixed horizon. It is not a strategy, and this script is the
difference between the two — it asks whether the tilt survives an entry rule, a
stop, a target, commission and slippage, out of sample, after correcting for the
number of configurations tried.

The control matters more than the leaderboard. ``SweepReversalControl`` takes
the *same* population of sweeps under the *same* gates and trades the opposite
direction with mirrored levels. If continuation and reversal both lose, the
entry machinery is the problem, not the direction. If they both win, something
is wrong with the cost model. Only a split result says anything about direction.

Usage::

    python scripts/sweep_continuation_study.py --symbols BTC-USD,ETH-USD,SOL-USD

Requires a local cache; populate it with ``python -m axiom.cli cache-data``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from axiom.core.series import OHLCVSeries
from axiom.core.types import get_instrument
from axiom.data.providers import CSVProvider
from axiom.research.lab import Candidate, StrategyLab, grid
from axiom.strategy.base import Strategy
from axiom.strategy.sweep_continuation import (
    SweepContinuationStrategy,
    SweepReversalControl,
)

#: The axes swept. Deliberately small. Every combination is a hypothesis test
#: and the deflated Sharpe charges for all of them, so a wide grid does not buy
#: a better answer — it buys a higher bar and a worse one.
#:
#: ``require_hold`` is deliberately absent. Under the shipped
#: :class:`~axiom.ict.engine.ICTConfig` the detector only emits raids that
#: closed back inside, so True produces zero trades on every symbol. Including
#: it would double the trial count — and therefore the deflated-Sharpe bar —
#: while contributing nothing but empty rows.
AXES: dict[str, list[Any]] = {
    "entry_window": [1, 3, 6],
    "min_penetration_atr": [0.0, 0.25],
    "max_penetration_atr": [1.0, 3.0],
    "min_rr": [1.2, 2.0],
}


def _make(cls: type[Strategy], params: dict[str, Any]) -> Strategy:
    """Named factory rather than a lambda, so the closure captures by value."""
    return cls(**params)


def _candidates() -> list[Candidate]:
    """Treatment and control over an identical parameter grid.

    Both families are counted as trials. Charging the deflated Sharpe only for
    the treatment would be selecting on the control's behalf.
    """
    out: list[Candidate] = []
    for params in grid(**AXES):
        for cls, tag in (
            (SweepContinuationStrategy, "continuation"),
            (SweepReversalControl, "control"),
        ):
            out.append(
                Candidate(
                    name=tag,
                    factory=lambda c=cls, p=params: _make(c, p),  # type: ignore[misc]
                    params=params,
                )
            )
    return out


def _summarise(result: Any, symbol: str) -> dict[str, Any]:
    """Treatment-vs-control comparison, which the leaderboard does not show."""
    families: dict[str, list[Any]] = {"continuation": [], "control": []}
    for row in result.viable:
        families.setdefault(row.candidate.name, []).append(row)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "n_trials": result.n_trials,
        "skill_threshold_sharpe": result.skill_threshold_sharpe,
        "pbo": None if result.pbo is None else result.pbo.probability,
    }
    for family, rows in families.items():
        if not rows:
            summary[family] = {"viable": 0}
            continue
        best = max(rows, key=lambda r: result.deflated.get(r.candidate.label, 0.0))
        summary[family] = {
            "viable": len(rows),
            "median_sharpe": float(sorted(r.sharpe for r in rows)[len(rows) // 2]),
            "median_avg_r": float(sorted(r.avg_r for r in rows)[len(rows) // 2]),
            "best_label": best.candidate.label,
            "best_sharpe": best.sharpe,
            "best_dsr": result.deflated.get(best.candidate.label, float("nan")),
            "best_trades": best.trades,
        }
    summary["survivors"] = [r.candidate.label for r in result.survivors]
    summary["verdict"] = result.verdict
    return summary


def run(symbol: str, timeframe: str, data_root: str, folds: int) -> dict[str, Any]:
    series: OHLCVSeries = CSVProvider(data_root).fetch(
        get_instrument(symbol), timeframe, days=10_000
    )
    print(f"\n{'=' * 78}\n{symbol} {timeframe} — {len(series):,} bars, "
          f"provenance {series.provenance.label}\n{'=' * 78}")
    if not series.provenance.is_evidential:
        print("REFUSING: this series is not evidential. Nothing measured here "
              "would describe a market.")
        return {"symbol": symbol, "error": "non-evidential"}

    candidates = _candidates()
    started = time.monotonic()
    result = StrategyLab(n_folds=folds).search(series, candidates)
    print(result.render(top=10))
    print(f"\nElapsed: {time.monotonic() - started:.0f}s")

    summary = _summarise(result, symbol)
    cont, ctrl = summary.get("continuation", {}), summary.get("control", {})
    if cont.get("viable") and ctrl.get("viable"):
        print(
            f"\nTreatment vs control — median Sharpe "
            f"{cont['median_sharpe']:+.2f} vs {ctrl['median_sharpe']:+.2f}, "
            f"median avg R {cont['median_avg_r']:+.2f} vs {ctrl['median_avg_r']:+.2f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--data-root", default="data/cache")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    summaries = [
        run(symbol.strip(), args.timeframe, args.data_root, args.folds)
        for symbol in args.symbols.split(",")
        if symbol.strip()
    ]
    if args.out:
        Path(args.out).write_text(json.dumps(summaries, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
