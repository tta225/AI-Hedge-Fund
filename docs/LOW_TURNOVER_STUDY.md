# Slow signals: the premise held, the edge did not

*Run: 2026-08-28. Source: Alpaca IEX daily bars, 2020-07 to 2026-08.
Reproduce with `python -m scripts.low_turnover_study`.*

The turnover study found the only result in this project that replicated out of
sample: trading less was worth ~0.45 Sharpe and ~5× capacity on a weak signal.
That pointed somewhere specific — search families that are slow *because of what
they measure*, rather than fast families slowed down afterwards.

This is that campaign. Five families, twelve configurations, one clean
confirmation universe.

**The headline: the premise is confirmed and the conclusion is unchanged. These
families turn over 0.8–2.5× a year against 15–25× for the fast ones, and cost
0.15–0.39%/yr against 2.74%. None of them produced an edge. The best result on
discovery — a Sharpe of 1.02 — went to −0.11 on names that had not selected it.**

---

## 1. The premise was worth testing, and it is true

Turnover, gross and net Sharpe, and annual cost drag at $10M, quarterly
rebalance with the frozen 0.2/0.4 buffer:

| family | turn/yr | gross | net | cost %/yr |
|---|---|---|---|---|
| mom12_1 | 2.3 | +0.715 | +0.679 | 0.36 |
| residual_mom | 2.5 | +0.622 | +0.583 | 0.39 |
| lt_reversal | 1.7 | −0.104 | −0.141 | 0.28 |
| low_vol | 1.1 | −0.740 | −0.754 | 0.16 |
| illiquidity | 0.8 | −0.297 | −0.315 | 0.15 |

For comparison, the cross-sectional campaign's fast candidates ran at 15–25×
turnover and paid **2.74%/yr**. The cost structure genuinely does favour this
regime: the drag is roughly seven to eighteen times smaller, and for the two
positive families it is now a rounding error against the gross figure rather
than the thing that consumes it.

That is a real finding about *mechanism*. It is not a finding about edge.

## 2. The campaign: nothing survived, again

Twelve trials, three anchored walk-forward folds, 800-bar warm-up.

| candidate | Sharpe | DSR | ret% | maxDD% | turn/yr |
|---|---|---|---|---|---|
| mom12_1[rebal=21,skip=21] | **1.02** | 0.24 | 31.40 | 7.82 | 4.1 |
| mom12_1[rebal=21,skip=0] | 0.84 | 0.00 | 25.79 | 8.19 | 3.9 |
| mom12_1[rebal=63,skip=21] | 0.80 | 0.00 | 23.93 | 8.64 | 2.7 |
| residual_mom[rebal=63] | 0.62 | 0.00 | 17.11 | 8.87 | 2.9 |
| lt_reversal[rebal=63] | 0.09 | 0.00 | 1.06 | 17.47 | 2.4 |
| illiquidity[rebal=63] | −0.39 | 0.00 | −10.09 | 16.37 | 1.5 |
| low_vol[rebal=63] | −0.76 | 0.00 | −23.71 | 27.04 | 1.8 |

The skill bar for twelve trials is a Sharpe of 1.06. The best candidate reached
1.02 — *below* what the best of twelve coin flips would be expected to produce.

## 3. The plausibility prior earned its keep

The reference-class check flagged the 1.02 immediately: it exceeds the best of
61 published systematic strategies (0.892), and **none of them clears 1.0**. The
verdict text says such a claim "needs an explanation" and lists the usual ones —
lookahead, survivorship, unmodelled cost, too short a sample.

The confirmation universe supplied the explanation:

| candidate | discovery | **confirmation** |
|---|---|---|
| mom12_1[rebal=21,skip=21] | +1.024 | **−0.112** |
| mom12_1[rebal=21,skip=0] | +0.845 | **−0.090** |
| mom12_1[rebal=63,skip=21] | +0.797 | **−0.075** |
| mom12_1[rebal=63,skip=0] | +0.647 | **−0.173** |
| residual_mom[rebal=63] | +0.624 | **+0.016** |

Every momentum variant flipped sign. Residual momentum survived as
indistinguishable from zero.

Worth stating plainly, because it is the methodological lesson of this run:
**PBO said 24% — "selection generalises acceptably" — and the transfer still
failed completely.** PBO is measured by reshuffling folds *within one universe*,
so it detects overfitting to a time window and is blind to overfitting to a set
of names. The two are different failure modes and only the second was operating
here. A held-out universe is not an optional extra alongside PBO; it tests
something PBO structurally cannot see.

## 4. A bug this campaign caught

`ResidualMomentum` originally scored on the residuals of an OLS fit *with* an
intercept. Those residuals sum to exactly zero by construction, so the score was
ranking floating-point noise. The fix is the construction the literature
actually uses: fit the intercept so beta is unbiased, then subtract only the
market component, leaving the idiosyncratic drift in the residual.

The correction moved gross Sharpe from +0.078 to +0.622. Every residual-momentum
figure above is post-fix. A unit test now pins the property — a panel where one
name has a known private uptrend must rank that name first — because the broken
version passed every contract test it had.

## What this does not rescue

Nothing. Six campaigns, no survivors. The low-turnover thesis was correct about
costs and produced no edge, which is a more specific null than the previous
five: it rules out "the earlier failures were a cost artefact" a second time,
now from the opposite direction.

## Caveats that are not footnotes

**Survivorship, worst here.** Long-term reversal is the family most damaged by
it: a three-year loser that subsequently delisted is absent from a universe
built out of names liquid today, so the long leg is drawn from a sample with the
worst outcomes deleted. Its −0.14 is an upper bound and a generous one.

**No value, no quality.** The two strongest low-turnover families in the
literature need fundamentals, which this platform has no source for. Their
absence is the largest single gap in the search, and closing it needs a data
feed rather than more code.

**One window.** 2020–2026 is a single macro regime, and a six-year sample of a
quarterly-rebalanced strategy contains very few independent observations.

## What follows

The two directions this points at are not more price-based factor searching:

1. **A fundamentals feed**, which would open value and quality — the families
   with the strongest published evidence and the lowest natural turnover.
2. **A point-in-time universe**, without which every number in this document
   remains an upper bound of unknown size.

Both are data problems. Six campaigns is enough evidence that the bottleneck is
no longer the search machinery.
