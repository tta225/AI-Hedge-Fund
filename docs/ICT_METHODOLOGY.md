# ICT methodology, as formalised in AXIOM

ICT concepts are taught discretionarily — "the last down candle before the move", "a clean
sweep of the highs". Trading them systematically means committing to precise definitions.
This document states the definition AXIOM uses for each concept, where judgement was
required, and how to change it.

Every threshold below is a **prior, not an estimate**. They are starting points chosen to be
defensible, not values fitted to data. Replace them with measured hit rates once you have
run real market data through the backtester.

> **Reconciliation note.** These definitions follow standard ICT methodology. Where your own
> stratagem differs — different killzone bounds, different displacement thresholds, extra
> confirmation requirements — those differences belong in `ICTConfig` and in the strategy
> classes, and this document should be updated alongside them.

---

## 1. Swing points

**Definition.** A swing high of strength *n* is a bar whose high strictly exceeds the *n*
bars to its left and is not exceeded by the *n* bars to its right.

**Judgement call.** Strict on the left, non-strict on the right. A flat double top therefore
registers once, at its first bar — rather than twice, or not at all.

**Confirmation.** `origin + n`. Configurable via `ICTConfig.swing_strength` (default 2;
`STRICT_CONFIG` uses 3).

---

## 2. Market structure — BOS, CHoCH, MSS

A structure event fires when a **close** breaks a prior confirmed swing.

| Condition | Event | Meaning |
|---|---|---|
| Break agrees with prevailing bias | **BOS** | Continuation |
| Break opposes bias, no displacement | **CHoCH** | Change of character — first evidence of a flip |
| Break opposes bias, with displacement | **MSS** | Market structure shift — higher conviction |

**Judgement calls:**

* **Close, not wick.** A wick beyond a swing is a liquidity raid, not a structural break.
  Wick-based breaks are handled by the liquidity module instead.
* **Displacement = close-through distance ÷ ATR.** Default threshold `0.25`. This is what
  separates an energetic reversal from price drifting one tick past a pivot.
* **The first break with no established bias is a BOS**, not a CHoCH — it defines a trend
  rather than reversing one.

---

## 3. Fair value gaps

**Definition.** Three bars where the first and third do not overlap.

* Bullish: `low[i+1] > high[i-1]` — the band between them is the gap
* Bearish: `high[i+1] < low[i-1]`

**Life cycle** (all tracked):

1. **formed** — third bar closes; the gap becomes visible
2. **CE touched** — price reaches the 50% level (consequent encroachment), which ICT treats
   as the meaningful fill rather than a full close
3. **mitigated** — price trades fully through
4. **inverted** — price *closes* decisively beyond, then respects it from the far side; the
   gap flips polarity and becomes an inverse FVG

**Judgement calls:**

* Steps 3 and 4 are distinct because a mitigated gap is spent, while an inverted one is a
  live level with the *opposite* bias.
* Inversion requires a **close** beyond the gap, not a wick.
* Minimum gap height `0.08 × ATR` (default) filters the one-tick noise gaps that make both
  the chart and the backtest meaningless.

---

## 4. Order blocks and breakers

**Definition.** The last candle closing *against* the direction of a displacement leg that
broke structure. Bullish order block = the final down-close candle before the up leg. Zone
spans that candle's full high-to-low range.

**Quality gates, in order of importance:**

1. The leg **must break structure**. A displacement that resolves nothing is not an order block.
2. The leg **should leave an FVG**. Blocks without one are kept but flagged
   (`has_imbalance=False`); `require_order_block_imbalance=True` drops them entirely.
3. The block is invalidated when price **closes** through its far edge, becoming a
   **breaker** that trades with inverted polarity.

**Judgement calls:**

* Search runs **backwards from the break**, up to `order_block_lookback` bars (default 12),
  so the block anchors to the leg that actually did the work regardless of its length.
* Zone uses the **full candle range**, not just the body. The body-only convention is
  defensible; it is a one-line change in `find_order_blocks`.
* `entry_edge` is the proximal side (first engagement); `protective_edge` is the distal side
  (natural stop anchor).

---

## 5. Liquidity pools and sweeps

**Pools.** Swing highs are buyside liquidity (resting buy stops above); swing lows are
sellside. Extremes within `equality_atr` (default `0.12 × ATR`) collapse into a single
**equal highs/lows** cluster — the highest-quality pool, because every chart reader has
drawn the same line.

**Judgement call.** A pool sits at the **extreme** of its members, not their mean. Buy stops
rest above the highest of a set of equal highs.

**Sweeps.** Price trades beyond a pool and then **closes back inside**. The rejection is what
distinguishes a stop raid from a genuine breakout — liquidity was taken to fill orders, not
to continue.

* Minimum penetration `0.05 × ATR` (noise filter)
* `require_close_back=False` also returns clean breakouts, which is useful for *measuring*
  how often a raid is not a reversal

**Confirmation.** A raid alone is a hypothesis. `confirm_sweep_reversals` links it to a
structure break in the reversal direction within `sweep_confirmation_bars` (default 20).
Without that break, price simply continued.

---

## 6. Dealing range, premium/discount, OTE

**Dealing range.** The most recent confirmed swing high/low pair, within `lookback` bars
(default 100).

* Above 50% = **premium** → look for shorts
* Below 50% = **discount** → look for longs
* **OTE** = the 0.62–0.79 retracement — deep enough to be efficient, shallow enough to still
  be a retracement

**Judgement call.** `premium_discount_bias` is deliberately *contrarian to position*: price
in premium argues for shorts. This is a **filter on directional ideas, not a signal**.

**Session references.** CBDR (14:00–20:00 ET), Asian range (19:00–00:00 ET), and Flout
(15:00–16:00 ET) project standard deviations to estimate where the day's expansion
terminates. The **true day open** (New York midnight) is the intraday premium/discount
reference independent of any swing range.

---

## 7. The session clock

All windows are New York local time, DST-aware, and half-open `[start, end)`.

| Window | ET | Role |
|---|---|---|
| Asia killzone | 20:00–00:00 | Range-building; its extremes become the day's first pools |
| London killzone | 02:00–05:00 | Judas swing against the Asian range, then reversal |
| **NY AM killzone** | 07:00–10:00 | Highest-probability window for the daily range extreme |
| London close | 10:00–12:00 | Reversion as European desks flatten |
| NY lunch | 12:00–13:00 | Low participation; breakouts here are suspect |
| NY PM killzone | 13:30–16:00 | Afternoon expansion into the close |
| Silver Bullet | 03:00–04:00, 10:00–11:00, 14:00–15:00 | One-hour FVG-entry windows |
| Macros | :50–:10 each hour | Algorithmic reprice periods |

**Judas swing.** The opening manipulation: price runs one side of the prior range to collect
stops, then reverses. `judas_swing()` returns the direction of the *false* move.

---

## 8. SMT divergence

Correlated instruments should make extremes together. When one makes a new high and its
correlate does not, the move is unsupported across the complex — a raid, not expansion.

**Judgement call — and a real trap.** Comparison is done on **aligned timestamps only**,
never bar-index to bar-index. Two instruments with different session calendars compared
positionally are silently offset, which manufactures divergences that do not exist.
`align()` raises if the two share no timestamps.

Inverse pairs (EURUSD/DXY) are handled explicitly: for them, divergence means the two moved
the *same* way.

---

## 9. Confluence scoring

`confluence_score(state, price, direction)` returns 0–1 as a weighted tally:

| Component | Weight |
|---|---|
| Structural bias agrees | 0.25 |
| Premium/discount position agrees | 0.20 |
| Price in OTE band | 0.10 |
| Live order block in direction | 0.15 |
| Live FVG in direction | 0.15 |
| Recent sweep implying direction | 0.15 |

**This is a ranking heuristic, not an edge.** It is deliberately simple and transparent — a
weighted tally of independent confirmations rather than a fitted model. Its role is to rank
candidate setups *within* a strategy. The weights are priors and should be replaced by
measured hit rates once real data has been run.

---

## Tuning

`ICTConfig` holds every threshold in one auditable object. `STRICT_CONFIG` is a
higher-conviction preset (swing strength 3, displacement 0.5 ATR, minimum gap 0.15 ATR,
order blocks required to have left an imbalance).

Changing thresholds changes what the engine sees. Change them deliberately, one at a time,
and re-measure — tuning several at once on a single sample is how curve-fitting starts.
