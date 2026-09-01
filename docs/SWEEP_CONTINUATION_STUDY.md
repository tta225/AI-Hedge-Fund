# Does the sweep tilt survive being traded?

*Run: 2026-08-26. Source: Coinbase Exchange, keyless public candles.
Reproduce with `python -m axiom.cli cache-data` then
`python scripts/sweep_continuation_study.py`.*

[REAL_DATA_FINDINGS.md](REAL_DATA_FINDINGS.md) measured a directional tilt.
After a liquidity sweep, price continued **with** the raid more often than it
reversed against it — on all four series, by −17.8, −6.1, −6.4 and −7.1
percentage points of "reversal lift".

That is a base rate over a fixed horizon. It says nothing about where to put a
stop, what to target, or whether anything is left after commission and
slippage. This document is the gap between the two.

**Answer: the direction holds up. The tradeability does not.**

---

## What was tested

`SweepContinuationStrategy` trades with the raid.
`SweepReversalControl` takes the same sweeps under the same gates and trades
the opposite direction with stop and target mirrored about entry. Both were
swept over an identical 24-point grid — `entry_window` × `min_penetration_atr`
× `max_penetration_atr` × `min_rr` — giving 48 trials per symbol, all of them
counted against the deflated Sharpe.

Evaluation is 4-fold anchored walk-forward. Every reported number comes from
test windows that follow the data used to reach them.

## Results

| | BTC-USD 1h | ETH-USD 1h | SOL-USD 1h |
|---|---|---|---|
| Trials | 48 | 48 | 48 |
| Skill bar (Sharpe) | 3.65 | 3.74 | 3.33 |
| PBO | 21% | 9% | 1% |
| **Continuation** — median Sharpe | −0.62 | **+1.74** | −7.36 |
| **Control** — median Sharpe | −2.80 | −1.49 | −8.40 |
| **Continuation** — median avg R | −0.20 | **+0.51** | −1.31 |
| **Control** — median avg R | −0.24 | −0.12 | −1.44 |
| Best deflated Sharpe | 0.00 | 0.00 | 0.00 |
| **Survivors** | **none** | **none** | **none** |

### The direction claim survives

Continuation beat its own reversal control on **all three symbols**, on both
median Sharpe and median average R. That is the base-rate finding reproducing
under a completely different measurement — a full backtest with stops, targets,
commission and slippage, rather than a fixed-horizon hit rate. Two independent
methodologies agreeing on direction is the strongest statement in this document.

### The tradeability claim does not

Nothing cleared the bar. The best candidate on each symbol scored a deflated
Sharpe probability of **0.00** against a 0.95 threshold. With 48 trials and the
dispersion observed, a raw Sharpe of roughly 3.3–3.7 would have been needed to
be distinguishable from the best of that many coin flips; the best observed was
1.84.

And the profile across symbols is not one of a thin edge. It is one of no edge:

* ETH is genuinely positive (+1.74 median Sharpe, +0.51 average R over 40 trades).
* BTC is mildly negative.
* SOL is a disaster — −7.36 median Sharpe, −1.31 average R, 25% drawdown.

A real effect degrades. It does not invert. One profitable symbol out of three,
with the worst one losing more than a full R per trade, is what a coin looks
like when you flip it three times.

The low PBO figures are worth reading correctly. PBO asks whether the
*selection procedure* generalises — whether picking the in-sample winner finds
something that holds up out of sample. 1–21% says the procedure is sound. It
does not say the thing it selected is any good, and here it wasn't.

## Two things this run found that were not the question

**The detector only emits rejection-type sweeps.**
`ICTConfig.sweep_require_close_back` defaults True, so all 297 sweeps on BTC-USD
had `closed_back_inside=True`. The strategy shipped with `require_hold=True`,
intended to isolate breakout-type raids — which silently filtered its entire
population away and produced **zero trades on a year of real bars**. The default
is now False, the flag is plumbed through `ICTConfig` so the other population is
reachable at all, and a test pins the default. This is also the clearest
statement of what the base rates measured: the continuation tilt is a tilt in
raids that *rejected*, not in breakouts that stuck.

**A mirrored control is not a matched control.**
Continuation took 40–117 trades where the control took 78–134 on the same
sweeps. The signals are emitted at identical bars, but the trades that follow
are not: a position that runs to a 2R target occupies bars during which no new
signal can be taken, so the two arms desynchronise after their first divergent
outcome. The comparison above is still informative — it is the same setups
traded both ways — but it is a comparison of two strategies, not of one
population split in half. A per-signal paired study, holding position state
fixed, would be the stronger test.

## What this does not establish

Crypto, one venue, one year, three correlated majors, hourly bars. The 2025-08
to 2026-08 window is one regime. Nothing here transfers to equities or futures
without being re-measured there, and nothing here is a statement about ICT in
general — only about this one claim, on this data, traded this way.

## What follows from it

The gate stated when this work started was: nothing goes live until something
clears deflated Sharpe out of sample. Nothing did.

So `SweepContinuationStrategy` is a research object. It is registered in the
CLI, it is tested, and it is not a candidate for capital. Infrastructure work
downstream of this — persistence, broker reconciliation, portfolio risk,
ops — proceeds on its own merits, because it is what makes *any* future edge
safely deployable and none of it depends on this one existing. Automation
multiplies expectancy. Multiplying this expectancy would be a way to lose money
faster.
