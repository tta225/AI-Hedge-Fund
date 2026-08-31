"""Search the cross-sectional agents for something that clears the bar.

Three prior campaigns found nothing. Each was limited in the same three ways,
and this one changes all three deliberately rather than sweeping the same
ground more finely:

*Different data.* One year of hourly bars on three correlated crypto majors is
roughly 1.2 effective independent bets. This runs six years of daily bars over
65 US large caps across every major sector — the asset class and horizon where
the replicated anomalies actually live.

*Different shape.* Everything tested so far was single-instrument and
pattern-driven. Momentum, reversal and statistical arbitrage are **relative**
statements: they rank names against each other and are meaningless on one
series. That machinery existed in :mod:`axiom.alpha` and had never been
measured, because no harness could evaluate it.

*A heterogeneous candidate set.* The seasonality run showed that a homogeneous
grid *lowers* the deflated-Sharpe bar, because the bar scales with dispersion
across trials. Twenty-four near-identical hours made one outlier look
extraordinary. This set spans structurally different families — trend against
mean-reversion against factor-residual — so the dispersion, and therefore the
bar, is real.

**The universe is survivorship-biased and that inflates every number here.**
It is a list of names liquid *today*, applied to a window starting six years
ago, so every constituent survived. The honest reading is that these results
are an upper bound: a strategy failing here with that advantage would fail
harder without it.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from axiom.alpha.agents import MomentumAgent, StatArbAgent, VolumePressureAgent
from axiom.alpha.ensemble import AlphaEnsemble
from axiom.alpha.panel import Panel
from axiom.core.types import get_instrument
from axiom.data.providers import CSVProvider
from axiom.research.panel_lab import PanelCandidate, PanelLab


def load_panel(root: str, timeframe: str) -> Panel:
    """Build an aligned panel from every cached symbol in ``root``."""
    paths = sorted(glob.glob(f"{root}/*_{timeframe}.csv"))
    if not paths:
        raise SystemExit(
            f"No cached bars in {root}. Run scripts/cache_equities.py first."
        )
    symbols = [Path(p).name.split("_")[0] for p in paths]
    provider = CSVProvider(root)
    series = {
        symbol: provider.fetch(get_instrument(symbol), timeframe, days=100_000)
        for symbol in symbols
    }
    return Panel.from_series(series)


def candidates() -> list[PanelCandidate]:
    """Structurally different families, each at a few horizons.

    Momentum and reversal are the *same* agent with opposite signs — the
    inversion is expressed by reading the bottom of the ranking as the long
    side, which the lab does by flipping conviction. Including both is what
    makes the dispersion honest: a set containing only trend variants would
    understate how much a good result could owe to luck.
    """
    out: list[PanelCandidate] = []

    for lookback in (21, 63, 126, 252):
        for rebalance in (5, 21):
            out.append(
                PanelCandidate(
                    name="momentum",
                    factory=lambda lb=lookback: [  # type: ignore[misc]
                        MomentumAgent({"lookback": lb})
                    ],
                    params={"lookback": lookback, "rebal": rebalance},
                    rebalance_every=rebalance,
                )
            )

    # Short-horizon reversal: the literature's highest-Sharpe equity anomaly
    # (0.816) is short-term reversal, which is momentum's ranking read upside
    # down at a short lookback.
    for lookback in (2, 5, 10):
        for rebalance in (1, 5):
            out.append(
                PanelCandidate(
                    name="reversal",
                    factory=lambda lb=lookback: [  # type: ignore[misc,arg-type]
                        _Inverted(MomentumAgent({"lookback": lb}))
                    ],
                    params={"lookback": lookback, "rebal": rebalance},
                    rebalance_every=rebalance,
                )
            )

    for lookback in (60, 120):
        for entry_z in (1.0, 2.0):
            out.append(
                PanelCandidate(
                    name="statarb",
                    factory=lambda lb=lookback, z=entry_z: [  # type: ignore[misc]
                        StatArbAgent({"lookback": lb, "entry_z": z})
                    ],
                    params={"lookback": lookback, "z": entry_z},
                    rebalance_every=5,
                )
            )

    for lookback in (10, 20, 40):
        out.append(
            PanelCandidate(
                name="volume",
                factory=lambda lb=lookback: [  # type: ignore[misc]
                    VolumePressureAgent({"lookback": lb})
                ],
                params={"lookback": lookback},
                rebalance_every=5,
            )
        )

    out.append(
        PanelCandidate(
            name="ensemble",
            factory=lambda: AlphaEnsemble(
                [MomentumAgent({"lookback": 126}), StatArbAgent(), VolumePressureAgent()]
            ),
            params={"agents": 3},
            rebalance_every=5,
        )
    )
    return out


class _Inverted:
    """A momentum agent read upside down — the reversal reading of a ranking.

    A wrapper rather than a flag on the agent: the sign flip is a *portfolio*
    decision about which end of the ranking to buy, not a claim about what the
    agent measured, and burying it inside the agent would make two different
    hypotheses share one name in the leaderboard.
    """

    def __init__(self, inner: MomentumAgent) -> None:
        self.inner = inner
        self.agent_id = f"reversal_{inner.agent_id}"
        self.min_history = inner.min_history

    def generate_checked(self, panel: Panel, index: int) -> list[Any]:
        from dataclasses import replace

        # The agent_id is rewritten as well as the sign. The ensemble keys its
        # prior weights by roster agent_id and looks them up by signal
        # agent_id, so renaming only the wrapper leaves every signal
        # unattributable.
        return [
            replace(signal, signal=-signal.signal, agent_id=self.agent_id)
            for signal in self.inner.generate_checked(panel, index)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/equities")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    panel = load_panel(args.data_root, args.timeframe)
    print(
        f"Panel: {panel.n_symbols} symbols x {len(panel):,} bars "
        f"({panel.index[0].date()} -> {panel.index[-1].date()}), "
        f"provenance {panel.provenance.kind.value}"
    )
    print(
        "Universe is survivorship-biased by construction; every number below "
        "is an upper bound.\n"
    )

    lab = PanelLab(
        n_folds=args.folds, warmup=args.warmup, cost_bps=args.cost_bps
    )
    result = lab.search(panel, candidates())
    print(result.render(top=20))

    if args.out:
        payload: dict[str, Any] = {
            "n_trials": result.n_trials,
            "skill_threshold_sharpe": result.skill_threshold_sharpe,
            "pbo": None if result.pbo is None else result.pbo.probability,
            "verdict": result.verdict,
            "survivors": [r.candidate.label for r in result.survivors],
            "rows": [
                {
                    "label": r.candidate.label,
                    "sharpe": r.result.sharpe,
                    "dsr": result.deflated.get(r.candidate.label),
                    "total_return_pct": r.result.total_return_pct,
                    "max_drawdown_pct": r.result.max_drawdown_pct,
                    "annual_turnover": r.result.annual_turnover,
                    "observations": int(r.result.returns.size),
                }
                for r in result.leaderboard
            ],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
