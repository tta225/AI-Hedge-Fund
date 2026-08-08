# What the real data said

*First run: 2026-08-05. Source: Coinbase Exchange, keyless public candles.*

Until this run, every number this repository had ever produced came from a
generator that was written to contain the patterns the engine looks for. That is
a correctness check. It is not evidence, and the README has said so from the
first commit.

This is the first measurement on bars that actually traded.

---

## The dataset

| Symbol | Timeframe | Bars | Span |
|---|---|---|---|
| BTC-USD | 1h | 8,749 | 2025-08-05 → 2026-08-05 |
| ETH-USD | 1h | 8,749 | 2025-08-05 → 2026-08-05 |
| SOL-USD | 1h | 8,749 | 2025-08-05 → 2026-08-05 |
| BTC-USD | 15m | 11,494 | 2026-04-07 → 2026-08-05 |

~38,700 bars. Provenance `real`, `is_evidential = True`.

**This is crypto, and that is a real limitation, not a footnote.** See
[What this does not establish](#what-this-does-not-establish).

---

## Everything here turns on the control

"84% of fair value gaps get filled" sounds like an edge. It is not one, and the
reason is the only methodological point in this document that matters.

Price at short horizons revisits nearly everything nearby. *Any* band of similar
height near the current price gets touched most of the time. A rate quoted
without a control measures the market's general volatility and attributes it to
the pattern.

So every rate here is paired with a control:

* **For zones** (gaps, order blocks) — the same geometry, same height, same
  distance from the confirming bar's close, **mirrored to the other side**. A
  band that no ICT rule identifies and that exists nowhere in the methodology.
* **For directional claims** (sweeps, structure breaks) — the **same bar, the
  opposite direction**. Same volatility, same ATR threshold, same horizon. This
  isolates the only thing being claimed: direction.

### The control was wrong once, and the null test caught it

The first version controlled directional claims against a *randomly chosen bar*
elsewhere in the series. It failed `test_no_lift_is_significant` — the
random-walk null case, where by construction nothing is predictable — by
reporting a significant edge on pure noise.

The bug: sweeps are detected at moments of elevated volatility, so the 1-ATR
threshold at a sweep bar is systematically larger than at an average bar.
Comparing the two measured that difference, not the sweep. Anchoring the control
to the same bar fixed it.

It is worth being clear about what happened: **the first version of this module
reported BOS continuation at +5.4pp, and that number was an artefact of a broken
control.** With the control corrected it is +1.0pp. The null test is the only
reason that was caught rather than published.

---

## Results

Run `python3 -m axiom.cli base-rates --data-root data/cache`.

| Measurement | BTC 1h | ETH 1h | SOL 1h | BTC 15m |
|---|---|---|---|---|
| FVG reaches CE | −0.8pp | +0.0pp | −0.2pp | +1.8pp |
| Order block revisited | +1.5pp | +0.4pp | −0.7pp | +2.4pp |
| **Sweep reverses 1 ATR** | **−17.8pp\*** | **−6.1pp** | **−6.4pp** | **−7.1pp** |
| BOS continues 1 ATR | +1.0pp | −1.5pp | +0.7pp | −2.0pp |
| MSS continues 1 ATR | −5.2pp | +0.9pp | −1.9pp | +1.5pp |

`*` = 95% interval excludes zero.

### 1. Fair value gaps and order blocks show no edge at all

Raw rates look spectacular — 84% of gaps reach consequent encroachment, 89% fill
completely, 78% of order blocks get revisited. The mirrored control band, which
is not a gap and not an order block and carries no methodological meaning, gets
touched **85.2%** of the time.

Eight measurements, all within ±2.5pp of control, none significant. Not weakly
positive. Absent.

The fair value gap does not have a fill rate. It has the ambient revisit rate of
any nearby price band, and that has been mistaken for a property of imbalance.

### 2. Liquidity sweeps point the wrong way — consistently

This is the one robust finding in the dataset.

Negative in **all four** datasets — −17.8, −6.1, −6.4, −7.1 — and significant in
one. After a sweep, price is *more* likely to continue in the direction of the
sweep than to reverse against it.

The methodology's claim is the opposite: a sweep marks exhaustion and precedes
reversal. Measured on this data, over this horizon, it precedes continuation.

Consistency of sign across four independent series is worth more than the single
significance star. This is the finding most likely to survive.

### 3. Structure breaks are noise, and MSS does not beat BOS

BOS: +1.0, −1.5, +0.7, −2.0. MSS: −5.2, +0.9, −1.9, +1.5. Both straddle zero and
neither is significant anywhere.

The methodology holds that MSS — a change of character delivered with
displacement — is the higher-conviction signal. It does not measure that way.
The displacement filter costs roughly half the sample and buys nothing.

---

## What this means for the confluence weights

`ICTConfig`'s confluence weights were always labelled as priors. They now have
their first contact with evidence, and the evidence does not support them as
unconditional signals.

**They have not been changed.** Two reasons:

1. These are *unconditional* base rates. ICT's own claim is that these features
   work **in confluence and in context** — a gap inside a discount array, in a
   killzone, after a sweep, aligned with higher-timeframe bias. Measuring each
   feature in isolation and finding nothing does not refute the conditional
   claim. It refutes the idea that the features are individually informative,
   which is a weaker and different statement.
2. Re-fitting weights to crypto and applying them to ES would be worse than
   leaving honest guesses in place. A number tuned on BTC and shipped for
   equities is a guess wearing a lab coat.

What has changed is the standard of proof. Any future weight change must cite a
measurement, and the measurement must carry a control that passes the null test.

---

## What this does not establish

* **It is crypto.** BTC trades 24/7. There is no cash open, no cash close, no
  09:30 auction. Killzones, judas swings, Power of Three, the Silver Bullet
  window and everything else anchored to the equities session are **not tested
  here at all**. Those session concepts are the load-bearing half of the
  methodology and they remain entirely unmeasured.
* **It is one year, one regime.**
* **Outcome definitions are choices.** "Reverses 1 ATR within 48 bars" is
  defensible and arbitrary. A different horizon or threshold moves these
  numbers. The controls move with them — that is the point of having them — but
  the specific figures are not universal constants.
* **No conditioning.** No confluence filter, no session filter, no HTF
  alignment. Deliberate: establish the marginal rate first, then measure what
  conditioning adds. Conditioning first is how you find a 70%-win-rate setup
  with twelve observations.

---

## What to do next

1. **Get equities or futures data.** The session-anchored concepts are
   untestable on crypto and they are the ones the methodology rests on. Needs
   Alpaca keys or an equivalent feed. Highest-value unblock by a wide margin.
2. **Measure conditional rates.** Does a gap the confluence scorer tags beat its
   control by more than an untagged one? That is the question ICT actually makes
   a claim about, and the engine already computes the tag.
3. **Investigate sweep continuation.** The consistent negative lift says the
   reversal reading is backwards on this data. Treat with suspicion: hunting for
   a sign flip is exactly where overfitting begins. Any strategy built on it
   goes through `StrategyLab` and gets read on the deflated Sharpe, not the raw
   one.

---

## The first real backtest

`silver-bullet` on BTC-USD 15m, 30 days:

```
Equity: $250,000.00 → $247,179.56 (-1.13%)
  Trades               7
  Win rate             0.0%
  Worst loss vs budget 0.49x
```

Seven trades is not a result and the report says so itself.

The number that matters is the last one: **worst loss 0.49× the per-trade risk
budget.** The sizing fix was built and verified against generated bars, where it
moved the worst trade from 7.20× to 0.57×. It holds on real data. That is the
one claim in this document that transfers.
