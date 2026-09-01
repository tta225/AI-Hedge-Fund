# Testing the best claim in the literature

*Run: 2026-08-27. Source: Coinbase Exchange, keyless public candles.
Reproduce with `python -m axiom.cli cache-data` then
`python scripts/seasonality_study.py`.*

`paperswithbacktest/awesome-systematic-trading` surveys 61 published systematic
strategies. The highest Sharpe in the whole survey is **0.892**, for
"Overnight Seasonality in Bitcoin": go long BTC at 22:00 UTC, hold two hours,
flatten. It is also the only entry in the survey testable with data this
repository already holds.

**It did not reproduce. And the run produced a false positive that is more
interesting than the original claim.**

---

## The design

The study sweeps **all 24 hours**, not hour 22 alone. This is the whole
methodology.

Testing one nominated hour and finding it profitable is nearly uninformative.
With 24 hours available something had to come first, and a paper reporting hour
22 is itself the output of a search whose trial count nobody charged for.
Sweeping every hour puts the published hour in a field of 23 nulls identical to
it in every respect except the claim attached, and lets the deflated Sharpe pay
for the entire search at once.

The question is not *"is hour 22 profitable"*. It is **"is hour 22
distinguishable from the other 23?"**

## Result 1 — the published claim does not reproduce

| | BTC-USD | ETH-USD | SOL-USD |
|---|---|---|---|
| Hour 22 rank (of 24) | **23rd** | 4th | **23rd** |
| Hour 22 Sharpe | −0.04 | +0.84 | −9.30 |

On BTC — the asset the paper actually studied — 22:00 UTC ranked **second to
last of 24 hours**. The claimed 0.892 became −0.04.

ETH is the interesting near-miss: +0.84 there, close to the published 0.892,
and 4th of 24. That is what a coincidence looks like. If 22:00 UTC captured
something real about global crypto flow, it would not be near-best on ETH and
near-worst on BTC and SOL in the same twelve months.

## Result 2 — a false positive, and the mechanism behind it

On BTC, hour 13:00 cleared the bar: raw Sharpe 1.30, **deflated Sharpe
probability 1.00**, 109 trades. That is the first candidate in this project to
survive the lab's own honesty controls.

It is not real. Two independent checks caught it.

**It doesn't replicate.** Hour 13 scores 1.30 on BTC, 0.21 on ETH, and −9.29 on
SOL. Three highly correlated majors over the same twelve months. Whatever hour
13 is, it is not a property of crypto markets.

**The plausibility check flagged it on sight.** A Sharpe of 1.30 exceeds the
best of 61 published systematic strategies, none of which clears 1.0. The new
`axiom.research.reference` module said so automatically, in the same render
that reported the DSR of 1.00 — two of the lab's own controls disagreeing with
each other in print.

### Why the deflated Sharpe was fooled

This is worth stating precisely, because it is a real limitation of a control
this repository relies on heavily.

The deflated Sharpe's bar scales with the **dispersion of results across
trials**. The `StrategyLab` notes have always said so. What this run shows is
the consequence: a *homogeneous* candidate set has low dispersion, which
**lowers** the bar rather than raising it.

| Symbol | Variance of Sharpes across 24 hours | Best DSR | Survivor |
|---|---|---|---|
| BTC-USD | **0.21** | 1.00 | hour 13 |
| ETH-USD | 0.31 | 0.58 | none |
| SOL-USD | 0.49 | 0.00 | none |

BTC had the lowest dispersion and was the only symbol to produce a survivor.
Twenty-four near-identical rules clustered near zero make one outlier look
extraordinary. The same 1.30 Sharpe submitted alongside a structurally diverse
candidate set would not have cleared.

**A researcher can therefore lower their own significance bar by submitting a
homogeneous grid** — not by cheating, just by sweeping one parameter finely
instead of testing several different ideas. That is a footgun, and the outside
view in `reference.py` is now the backstop for it.

## Result 3 — long-only strategies inherit drift

Every one of SOL's 24 hours was negative. Not because SOL has bad seasonality,
but because SOL fell over the window and this is a long-only strategy.

This is why the within-asset ranking is the informative statistic and the
cross-asset Sharpe comparison is not. All 24 hours in a given symbol share that
symbol's drift, so ranking them against each other controls for it. Comparing
hour 22 on SOL against hour 22 on BTC does not.

## What was built

**`axiom.research.reference`** — the outside view. The deflated Sharpe and PBO
are both *internal* to a trial set: they ask whether a result beats its own
peers, and neither has any idea what a plausible Sharpe ratio is. This module
supplies the reference distribution (61 published anomalies: max 0.892, median
0.354, 69% below 0.5, **15% negative**, none above 1.0), Lo's standard error for
a Sharpe estimate, and a confidence interval. `SearchResult` now carries a
`plausibility` check on whichever candidate the leaderboard would promote.

It flags, it never rejects. High-frequency and capacity-constrained strategies
sit outside this reference class honestly and can carry a Sharpe of 3 without
anything being wrong.

The most useful thing it prints is the interval. A Sharpe from 40 observations
has an error bar wide enough to swallow the entire table above, and quoting the
point estimate hides that.

**`axiom.strategy.seasonality`** — `TimeOfDaySeasonality` and
`SeasonalityControl`, implemented from the paper's description. Two deviations
from the published rule, both of which cost the strategy something rather than
flattering it: a wide stop exists because the risk manager sizes from one, and
costs are charged on roughly 365 round trips a year.

## What this does not establish

One venue, one year, three correlated crypto majors, hourly bars, long-only.
The paper studied Gemini data over a different and longer window; this is not a
refutation of their sample, it is a failure to reproduce on ours. That is a
weaker claim than "the effect is not real" and it is the only one the data
supports.

## What follows

Nothing here is a candidate for capital, including hour 13 — especially hour
13. The gate is unchanged.

The lasting output of this run is not a strategy but a control: the lab now
knows what a plausible number looks like, which is the one thing its internal
statistics could never tell it.
