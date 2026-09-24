"""Test the highest-Sharpe claim in the published survey: 22:00 UTC in BTC.

"Overnight Seasonality in Bitcoin" reports Sharpe 0.892 for going long BTC at
22:00 UTC and holding two hours. It is the best number among the 61 strategies
surveyed in ``paperswithbacktest/awesome-systematic-trading``, and the only one
testable with data this repository already holds.

The design is a sweep over **all 24 hours**, not a test of hour 22 alone. This
matters more than any other choice here. Testing one hour in isolation and
finding it profitable is nearly uninformative — with 24 hours available,
something had to come first, and the paper reporting hour 22 is itself the
outcome of a search whose trial count nobody charged for. Sweeping every hour
puts the published hour in a field of 23 nulls that differ from it in nothing
but the claim attached, and lets the deflated Sharpe charge for the whole
search at once.

The question is therefore not "is hour 22 profitable" but **"is hour 22
distinguishable from the other 23?"**

Usage::

    python scripts/seasonality_study.py --symbols BTC-USD,ETH-USD,SOL-USD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from axiom.core.series import OHLCVSeries
from axiom.core.types import Direction, get_instrument
from axiom.data.providers import CSVProvider
from axiom.research.lab import Candidate, StrategyLab
from axiom.strategy.seasonality import (
    PUBLISHED_ENTRY_HOUR,
    PUBLISHED_HOLD_HOURS,
    TimeOfDaySeasonality,
)


def _candidates(hold_hours: int) -> list[Candidate]:
    """One candidate per UTC hour. The published hour gets no special status."""
    return [
        Candidate(
            name="seasonality",
            factory=lambda h=hour: TimeOfDaySeasonality(  # type: ignore[misc]
                entry_hour=h, hold_hours=hold_hours, direction=Direction.BULLISH
            ),
            params={"entry_hour": hour},
        )
        for hour in range(24)
    ]


def run(symbol: str, timeframe: str, data_root: str, folds: int, hold: int) -> dict[str, Any]:
    series: OHLCVSeries = CSVProvider(data_root).fetch(
        get_instrument(symbol), timeframe, days=10_000
    )
    print(f"\n{'=' * 78}\n{symbol} {timeframe} — {len(series):,} bars, "
          f"provenance {series.provenance.label}\n{'=' * 78}")
    if not series.provenance.is_evidential:
        print("REFUSING: not evidential data.")
        return {"symbol": symbol, "error": "non-evidential"}

    result = StrategyLab(n_folds=folds).search(series, _candidates(hold))
    print(result.render(top=24))

    ranked = result.leaderboard
    by_hour = {r.candidate.params["entry_hour"]: r for r in result.viable}
    published = by_hour.get(PUBLISHED_ENTRY_HOUR)
    rank = next(
        (i + 1 for i, r in enumerate(ranked)
         if r.candidate.params["entry_hour"] == PUBLISHED_ENTRY_HOUR),
        None,
    )
    print(
        f"\nPublished hour {PUBLISHED_ENTRY_HOUR:02d}:00 UTC — "
        + (
            f"rank {rank} of {len(ranked)} viable hours, "
            f"Sharpe {published.sharpe:+.2f}, avg R {published.avg_r:+.2f}, "
            f"{published.trades} trades"
            if published is not None and rank is not None
            else "not viable (too few trades)"
        )
    )
    return {
        "symbol": symbol,
        "n_trials": result.n_trials,
        "published_hour_rank": rank,
        "published_hour_sharpe": None if published is None else published.sharpe,
        "published_hour_avg_r": None if published is None else published.avg_r,
        "viable_hours": len(result.viable),
        "hour_sharpes": {
            int(h): r.sharpe for h, r in sorted(by_hour.items())
        },
        "survivors": [r.candidate.label for r in result.survivors],
        "plausibility": None if result.plausibility is None else result.plausibility.verdict,
        "verdict": result.verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--data-root", default="data/cache")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--hold", type=int, default=PUBLISHED_HOLD_HOURS)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    summaries = [
        run(s.strip(), args.timeframe, args.data_root, args.folds, args.hold)
        for s in args.symbols.split(",")
        if s.strip()
    ]
    if args.out:
        Path(args.out).write_text(json.dumps(summaries, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
