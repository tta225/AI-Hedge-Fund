# Pointing the machinery at equities

*Run: 2026-08-27. Source: Alpaca IEX daily bars.
Reproduce with `python scripts/cache_equities.py`, then
`python scripts/cross_sectional_study.py` and
`python -m scripts.momentum_confirmation`.*

Three previous campaigns found nothing. Each tested single-instrument,
pattern-driven ideas on one year of hourly bars across three correlated crypto
majors — about **1.2 effective independent bets**. This one changed all three
constraints at once rather than sweeping the same ground more finely.

**Result: still nothing clears the bar — and this time the reason is precise
enough to be useful.**

---

## What changed

**Different data.** Six years of daily bars over 65 US large caps across every
major sector. This is the asset class and horizon where the replicated
anomalies actually live.

**Different shape.** Momentum, reversal and statistical arbitrage are
*relative* statements — they rank names against each other and are meaningless
on a single series. That machinery already existed in `axiom.alpha` and had
**never been measured**, because no harness could evaluate a `Panel`.
`axiom.research.panel_lab` is that harness.

**A heterogeneous candidate set.** The seasonality run showed a homogeneous
grid *lowers* the deflated-Sharpe bar, since the bar scales with dispersion
across trials. This set spans trend, mean-reversion and factor-residual
families, so the dispersion — and the bar — is real.

## The discovery run

22 candidates, 4 anchored walk-forward folds, dollar-neutral long-short, 10 bp
per unit of notional traded.

| candidate | Sharpe | ret% | maxDD% | turn/yr |
|---|---|---|---|---|
| momentum, 252d, rebal 5 | **+0.43** | +22.9 | 16.4 | 15.5 |
| momentum, 252d, rebal 21 | +0.39 | +19.9 | 14.1 | 7.0 |
| momentum, 126d, rebal 21 | +0.01 | −2.3 | 14.4 | 8.6 |
| ensemble (3 agents) | −0.10 | −7.1 | 15.8 | 35.8 |
| statarb, 60d, z=1.0 | −0.57 | −33.7 | 43.6 | 94.2 |
| volume, 20d | −0.76 | −42.3 | 46.0 | 88.8 |
| reversal, 10d, rebal 1 | −1.36 | −48.8 | 55.3 | 145.1 |

**Skill bar: Sharpe > 1.63.** Best deflated Sharpe: **0.00**. Nothing survived.

**One variable explains almost the whole table: turnover.** Everything above
30×/year loses money; the two positive candidates turn over 7× and 15×. At
10 bp, 90× turnover is a 9% annual drag — larger than any edge on offer. The
cross-sectional agents are not mostly wrong about direction; they are mostly
too expensive to trade.

## The confirmation test, and the trap it avoids

The best candidate — 12-month momentum, monthly rebalance — sits close to the
published Time Series Momentum figure of 0.576. It would be easy to re-test
that single configuration, claim one trial instead of 22, and clear a much
lower bar.

**That is the oldest way to manufacture significance.** The configuration was
selected *after* looking at the results, so claiming a single trial for it is
not honest. The deflated Sharpe exists to prevent exactly this.

The legitimate move is to test the frozen configuration on data that played no
part in selecting it: **70 different US large caps, disjoint from the first
universe**.

| | discovery (selected on this) | confirmation (never seen) |
|---|---|---|
| Symbols | 65 | 70 |
| Sharpe | **+0.389** | **+0.040** |
| 95% CI | [+0.33, +0.45] | **[−0.02, +0.10]** |
| Total return | +19.9% | −0.7% |
| Annual turnover | 7.0× | 7.3× |
| Vs literature | 54th percentile | 19th percentile |

**The effect does not replicate.** It falls to roughly a tenth, and the
confirmation interval includes zero.

### It dies on costs

| cost | 0 bp | 5 bp | 10 bp | 20 bp | 40 bp |
|---|---|---|---|---|---|
| Sharpe | +0.129 | +0.096 | +0.063 | −0.003 | −0.135 |

Even at *zero* cost the confirmation Sharpe is 0.13. There is nothing for costs
to eat into.

### It dies on the rebalance day

The decisive diagnostic. Holding everything fixed and changing only *which day
of the month* the book rebalances — a choice carrying no information whatsoever:

| phase | 0 | 3 | 6 | 9 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|---|
| Sharpe | +0.063 | −0.023 | +0.069 | +0.119 | +0.161 | +0.006 | +0.244 |

**Spread 0.267 against a mean of 0.091 — three times the effect.** A monthly
rebalance over six years is only ~58 decisions, and the arbitrary choice of
which day to make them moves the answer more than the strategy does. That is
what noise looks like.

## Three bugs found in my own harness

Found by checking output against reality, not by the tests.

**The leaderboard sorted on deflated Sharpe alone.** In a failed search every
candidate scores exactly 0.00, so the order fell to insertion — and the first
run reported a −0.16 Sharpe candidate as "best" while +0.48 sat further down.
Ties now break on Sharpe.

**Walk-forward folds changed the trading days.** Rebalance phase was anchored
to each fold's `start` rather than to the panel, so a folded run traded on
different days than a continuous run over the same bars. That alone moved a
measured Sharpe from **0.07 to 0.15**, and swung `statarb[60, z=2.0]` from
**+0.47 to −0.90**. Phase is now anchored to the panel index and exposed as an
explicit parameter, which is what made the phase-sensitivity diagnostic above
possible.

**`AlphaEnsemble` had an unstated contract.** It keys prior weights by roster
`agent_id` and looks them up by *signal* `agent_id`. A wrapper that renamed
itself without renaming its signals raised a bare `KeyError` deep inside the
backtest loop, silently voiding 6 of 22 candidates. It now names the mismatch
and says what to do about it.

## What this does not establish

Both universes are **survivorship-biased by construction** — names liquid
today, applied backward six years — so every figure here is an upper bound.
Both share the 2020–2026 window, so this is a cross-sectional replication and
not a temporal one. And the agents tested are this repository's
implementations; a null result is evidence about them, not about momentum as a
published phenomenon.

## What follows

Four campaigns, no survivors. The measurement apparatus is now considerably
better than when this started — a cross-sectional harness that can be shown to
detect an edge when one exists (the oracle test), a phase diagnostic, a
plausibility prior, and three fewer bugs. What there still is not, is an edge.

The infrastructure work continues on its own merits, because it is what makes
*any* future edge deployable. But nothing here is a candidate for capital, and
the honest summary after four campaigns is that the ideas tested so far do not
survive costs, and the cost problem is quantitative rather than incidental:
turnover, not direction, is what has killed every candidate.
