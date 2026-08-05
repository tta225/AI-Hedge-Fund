# ICT methodology, as formalised in AXIOM

ICT concepts are taught discretionarily — "the last down candle before the move", "a clean
sweep of the highs". Trading them systematically means committing to precise definitions.
This document states the definition AXIOM uses for each concept, where judgement was
required, and how to change it.

Every threshold below is a **prior, not an estimate**. They are starting points chosen to be
defensible, not values fitted to data. Replace them with measured hit rates once you have
run real market data through the backtester.

> **Reconciliation status.** These definitions have been checked against the
> [ICT Knowledge Library](https://github.com/SrsBlack/ict-knowledge-library) (226 concept
> files, 2016–2026). Four corrections came out of that pass and are marked **[corrected]**
> below. Where your own stratagem differs from the canonical source, those differences
> belong in `ICTConfig` and in the strategy classes, and this document should be updated
> alongside them.
>
> Since implemented from the library: CRT, IPDA data ranges, turtle soup, and
> rejection blocks (see §10-13 below).
>
> Still not implemented: quarterly theory, propulsion/vacuum blocks, BPR
> (balanced price range), and the named models in `31-models/`.

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
* **[added]** The canonical *candle-level* displacement test is separate and now available
  as `is_displacement_candle`: body ≥ 1.5× the trailing average body, body ≥ 70% of the
  candle range, opposing wick ≤ 20% of range. Range alone cannot distinguish displacement
  from indecision — a bar with long wicks both ways has a large range and no conviction.
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
broke structure. Bullish order block = the final down-close candle before the up leg. The
zone is that candle's **body** (open→close); its midpoint is the **mean threshold**, the
entry depth ICT specifies.

**Quality gates, in order of importance:**

1. The leg **must break structure**. A displacement that resolves nothing is not an order block.
2. The leg **should leave an FVG**. Blocks without one are kept but flagged
   (`has_imbalance=False`); `require_order_block_imbalance=True` drops them entirely.
3. The block is invalidated when price **closes** through its far edge, becoming a
   **breaker** that trades with inverted polarity.

**Judgement calls:**

* Search runs **backwards from the break**, up to `order_block_lookback` bars (default 12),
  so the block anchors to the leg that actually did the work regardless of its length.
* **[corrected]** Zone is the candle **body** (open→close), not the full range. The
  canonical source is explicit that the body is the default and the range is a broader,
  less precise variant — available via `use_candle_range=True`. The body midpoint is the
  **mean threshold** (`OrderBlock.mean_threshold`), which is the entry depth ICT actually
  specifies.
* Stops use the **full candle** extreme (`protective_edge`), not the body edge. A stop at
  the body edge sits inside the origin candle's own wick and is taken out by noise that
  never invalidated the block.
* **[corrected]** Pivot anchoring (canonical criterion 4) is now flagged as
  `anchored_at_pivot` rather than ignored. Flagged, not required: pivot-anchored blocks
  are the highest quality, but they are not the only valid ones.
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
| **NY AM killzone** | 08:00–11:00 | Highest-probability window for the daily range extreme **[corrected]** |
| London close | 10:00–12:00 | Reversion as European desks flatten |
| NY lunch | 12:00–13:00 | Low participation; breakouts here are suspect |
| NY PM killzone | 13:30–16:00 | Afternoon expansion into the close |
| Silver Bullet | 03:00–04:00, 10:00–11:00, 14:00–15:00 | One-hour FVG-entry windows |
| Macros | 00:50, 02:50, 09:50, 13:50, 14:50 (±10m) | Five named windows **[corrected]** — previously one per session hour |

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


---

## 10. Rejection blocks

**Definition.** A candle whose **wick** documents a level being tested and refused.

**The distinction from an order block is the whole point:** an order block references
the **body** (where absorption happened); a rejection block references the **wick**
(where price was refused). Different objects from different parts of the same candle.

**Criteria (all required):** wick ≥ 60% of candle range · wick tip reaches a **known
liquidity level** · close at the far end of the range · next candle displaces in the
rejection direction.

**Judgement call.** The level requirement is not optional and is the criterion most
implementations drop. Without it, the detector returns every long-wicked candle on the
chart. `find_rejection_blocks` takes liquidity pools and returns nothing without them.

**Confirmation** is `origin + 1` — criterion 4 needs the next candle.

---

## 11. Turtle soup

**Definition.** A failed breakout: a known level is violated, reclaimed within 1–3
bars, and price then displaces in the opposite direction.

**Relationship to liquidity sweeps.** Deliberately a separate object:

| | Sweep | Turtle soup |
|---|---|---|
| Reclaim | same bar | within 1–3 bars |
| Displacement | not required | **required** |

Turtle soup is the stricter, complete pattern — it insists the market actually did
something after reclaiming. Every resolving sweep is a candidate; not every sweep
resolves.

**Confirmation** is credited at the displacement bar, the first point the whole pattern
is knowable.

---

## 12. CRT — Candle Range Theory

**⚠️ Not ICT-original.** Popularised in 2024 by Romeo and TTrades. ICT's public
position: *"based on my ideas but not my concept."* Implemented because the community
conflates the two, and stating the attribution is better than losing it.

**Mechanic.** Take a **completed** higher-timeframe candle; a later bar sweeps one bound
and closes back inside; target the opposite bound.

**The lookahead trap.** Reading an HTF candle's high/low *while inside that candle* puts
the traded move into its own reference range. This module only uses completed HTF
candles, and `CRTSetup.reference_close_index` records when the reference became
knowable. Pinned by test.

**Judgement calls.** CRT is materially less standardised than ICT — selection rules vary
between teachers. The time-of-day filter (02:00/03:00/05:00/09:00/13:00 NY) is
**opt-in**, because its canonical values are genuinely disputed. `reward_risk()` takes a
`stop_buffer`: without one, a single-tick sweep yields an R:R in the tens that no real
stop could achieve.

---

## 13. IPDA data ranges

**Definition.** Highest high and lowest low over trailing **20 / 40 / 60 trading days** —
the longest-horizon liquidity references on the chart.

**Two details the canonical source is specific about, both easy to get wrong:**

* **trading days, not calendar days.** A 20-calendar-day window covers roughly 14
  trading days; over 60 days the gap is nearly three weeks of data.
* **wick extremes, not closes.** The level is where price *traded* — that is where the
  stops are.

**Judgement call.** `ipda_bias` is contrarian to position, like the dealing-range
filter: above 70% of the 20-day range the draw is more plausibly the low. A filter, not
a signal.

IPDA itself is an **interpretive model**, not a documented protocol. The lookback levels
are the operationally useful part and are what is implemented.
