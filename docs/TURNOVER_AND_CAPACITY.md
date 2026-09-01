# Trading less, and costing better

*Run: 2026-08-27. Source: Alpaca IEX daily bars, 2020-07 to 2026-08.
Reproduce with `python -m scripts.turnover_capacity_study`.*

The cross-sectional campaign found one dominant pattern: every candidate above
30× annual turnover lost money, and the only positives turned 7× and 15×.
Direction was not the problem — the book was rebuilt too often. Two questions
follow, and this answers both.

**The headline: hysteresis flips the sign of the strategy on held-out data,
and roughly quintuples its capacity. The signal underneath is still too weak to
trade.**

---

## 1. Was the cost model wrong? Yes — it was too *generous*

A flat 10 bp per unit traded is size-independent, which charges a
thousand-dollar trade and a hundred-million-dollar trade identically. The new
model in `axiom.execution.costs` prices commission, half-spread, and
square-root market impact using the panel's own volume history:

    impact ≈ η · σ · √(Q / ADV)

Same book, $10M, 15.2× turnover:

| model | net Sharpe | Sharpe drag |
|---|---|---|
| gross (no costs) | +0.531 | — |
| flat 5 bp | +0.465 | 0.066 |
| flat 10 bp | +0.399 | 0.132 |
| **impact model** | **+0.293** | **0.239** |

The impact model is **harsher** than flat 10 bp, not kinder. The earlier nulls
were not an artefact of an overly punitive cost assumption; if anything the
flat charge was understating the true cost at this turnover. That closes off
the more comfortable explanation for four failed campaigns.

The linear part is only 1.2 bp. Impact supplies the rest, which is the point:
cost is dominated by the term that scales with size, and a model without that
term cannot see it.

## 2. Can turnover be cut without cutting signal? Yes, decisively

Hysteresis holds a name until it falls out of a *wider* exit band rather than
selling the moment it leaves the entry band. A name moving from rank 20 to 21
and back has said nothing, and round-tripping it pays a spread for noise.

On the discovery universe, impact cost at $10M:

| config | net | gross | turn/yr | cost %/yr |
|---|---|---|---|---|
| no buffer, rebal 5 | +0.293 | +0.531 | 15.2 | 2.74 |
| no buffer, rebal 21 | +0.326 | +0.434 | 6.6 | 1.21 |
| buffer .2/.4, rebal 5 | +0.325 | +0.396 | 5.5 | 0.67 |
| buffer .2/.5, rebal 5 | +0.273 | +0.326 | 4.3 | 0.47 |
| **buffer .1/.3, rebal 5** | **+0.474** | +0.556 | 6.6 | 1.03 |

Note that `buffer .1/.3` changes *two* things — it buffers **and** concentrates
into the top 10% instead of 20%. Reporting it as a buffer result would confound
the two, so both were isolated and then tested on the universe that selected
neither.

## 3. It replicates — which nothing else in this project has

| config | discovery | **confirmation** | turnover (confirm) |
|---|---|---|---|
| top 20%, no buffer | +0.293 | **−0.353** | 16.9× |
| top 10%, no buffer | +0.301 | **−0.342** | 25.0× |
| top 20%, buffer .2/.4 | +0.325 | **+0.061** | 5.8× |
| top 10%, buffer .1/.3 | +0.474 | **+0.139** | 7.2× |

On 70 names that played no part in selecting anything, **the unbuffered book is
decisively negative and the buffered book is positive.** That is a swing of
roughly 0.45 Sharpe attributable to trading less, and it is out-of-sample.

Two things make this more than a cost saving:

*Gross Sharpe improves too* — 0.051 to 0.176 on the confirmation universe.
Buffering removes trades that were losing money *before* costs, because chasing
mean-reverting rank noise is negative-expectancy on its own.

*The mechanism is general.* Hysteresis is a property of how a ranking is turned
into a book, not of momentum. It should transfer to any cross-sectional signal,
and the machinery is now in `axiom.research.turnover` for exactly that.

## 4. Capacity — the question a flat model cannot ask

Sharpe against capital deployed, impact costs, discovery universe:

| AUM | no buffer | with buffer |
|---|---|---|
| $1M | +0.445 | +0.369 |
| $10M | +0.293 | +0.325 |
| $50M | +0.018 | +0.247 |
| $100M | −0.186 | **+0.189** |
| $500M | −1.022 | −0.057 |
| $1B | −1.610 | −0.239 |

**Zero crossing moves from roughly $50–100M to roughly $100–500M.** Hysteresis
does not merely improve the Sharpe at one size; it extends the frontier by
about five times. That is the practical form of the result: the same signal
supports several times more capital when the book is not churned.

Participation is reported alongside, and anything above 10% is flagged as
extrapolating beyond where the square-root law was fitted — the $500M+ rows are
directional, not estimates.

## What this does not rescue

The full 22-candidate campaign was re-run on the **confirmation** universe with
the impact model and the buffer applied. Nothing survived. Best was
`momentum[252, rebal 5]` at a Sharpe of **0.04** against a 22-trial bar of 3.34.

So the honest summary is a split verdict:

* The **execution machinery** improved materially and the improvement
  replicates: ~0.45 Sharpe from turnover reduction, ~5× capacity, on held-out
  data.
* The **signal** is still not there. +0.139 is below the published median of
  0.354, nowhere near a deflated-Sharpe threshold, and was itself selected by
  looking at a grid of eight configurations.

Trading less turned a losing implementation of a weak signal into a
break-even one. It did not manufacture an edge, and nothing here is a
candidate for capital.

## Caveats that are not footnotes

**Survivorship.** Both universes are names liquid today applied backward six
years. Every figure is an upper bound.

**One window.** 2020–2026 is a single macro regime; this is a cross-sectional
replication, not a temporal one.

**η = 1.0 is a choice.** Published estimates of the impact coefficient cluster
around 0.3–0.6. The conservative value was chosen deliberately, and a friendlier
one would move every capacity number to the right. The *shape* of the curve —
and the fact that buffering shifts it — does not depend on that choice.

## What follows

The turnover finding is the first thing in this project that replicated out of
sample, and it points somewhere specific: if trading less is worth ~0.45 Sharpe
on a weak signal, the next campaign should search **low-turnover signal
families** rather than re-testing fast ones with better plumbing. Value,
quality, and long-horizon factor exposures rebalance quarterly by construction,
which is where the cost structure now measurably favours the desk.
