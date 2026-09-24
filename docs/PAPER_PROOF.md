# Proving the plumbing: the first order this desk ever sent

*Run: 2026-09-01, Alpaca paper account. One BTC-USD round trip, 0.0002 BTC.*

Every component of the execution path had tests. The path itself had never been
run. The venue's `submit()` was exercised against a stub that returned what the
code expected, which is a test of the code's beliefs rather than of the API.

**Three bugs surfaced in the first twenty minutes, and all three would have
stopped a live desk dead.** None was findable without sending a real order.

---

## What was verified

| link | result |
|---|---|
| account reachable | equity $99,943.71, trading not blocked |
| symbol translation | `BTC-USD` → `BTC/USD` |
| order accepted | Alpaca id `ce2a0ea2…`, client_order_id preserved |
| **fill** | **0.0002 BTC @ $78,716.50** |
| fill polled into store | 1 activity → 1 fill booked |
| idempotency | re-poll: "1 new, 1 already booked" — no double count |
| reconciliation | discrepancies detected and correctly classified |
| sell / close | position returned to flat |

The loop works. The bugs were in the details that only a real venue has.

---

## Bug 1 — crypto orders were rejected outright

```
HTTP 422: {"code":42210000,"message":"invalid crypto time_in_force"}
```

`Order` defaults to `TimeInForce.DAY`. Crypto trades continuously, so Alpaca has
no notion of a day order on it and rejects one. The error reads like a malformed
request rather than the one-word mapping problem it is.

**Every crypto order this desk could ever have sent would have failed**, and the
failure would have looked like a venue outage.

Fixed by translating DAY → GTC for crypto only, which is what a day order means
on a market that never closes. FOK is left alone; Alpaca accepts it and it
expresses something GTC does not.

## Bug 2 — the desk would have halted forever on any crypto position

Alpaca spells the same holding two different ways:

| endpoint | spelling |
|---|---|
| `/v2/orders` | `BTC/USD` |
| fill activities | `BTC/USD` |
| **`/v2/positions`** | **`BTCUSD`** |

`local_symbol()` handled only the slashed form. So the reconciler compared
`BTCUSD` from the broker against `BTC-USD` in the store, found neither in the
other's book, and reported **both a phantom and an unknown position** — for a
single, entirely correct holding.

The desk halts on a failed reconciliation. It would have halted on every tick,
forever, citing a discrepancy that did not exist. A crypto desk could never have
traded, and the halt reason would have sent an operator looking in the wrong
place.

Fixed using Alpaca's own `asset_class` field rather than guessing at the string,
because the separator-free spelling is genuinely ambiguous — an equity ticker
ending in `USD` is rare but not impossible, and splitting it would invent a
currency pair.

## Bug 3 — fees are charged in kind, and reported nowhere

Ordered 0.0002 BTC. The fill activity says `qty: 0.0002`. The position says
`0.0001995`.

The 0.25% difference is Alpaca's crypto fee, taken **in the asset itself**, and
it appears in no field of either the fill activity or the order record:

```json
{"qty": "0.0002", "price": "78716.497775622", "cum_qty": "0.0002",
 "leaves_qty": "0", "order_status": "filled", "swap_rate": "1"}
```

A desk deriving position from fills therefore disagrees with the broker by the
fee on **every** crypto trade. The absolute tolerance is 1e-8; the gap was 5e-7,
fifty times larger.

This is not rounding, so it was not fixed by widening a rounding allowance.
`reconcile_positions` gained a `relative_tolerance` that scales with the
position, **defaulting to zero** so nothing is silently loosened. A desk trading
a fee-in-kind venue sets it to at least its fee tier; an equity desk leaves it
off, because there the fee is cash and the share count is exact.

### The part that is still open

Closing the position leaves the mirror image: the desk's fills net to `+5e-07`
while the broker reports `0`. A relative allowance is a fraction of the
broker's quantity, and a fraction of zero is zero — **so the dust halts the desk
permanently, however the tolerance is set.**

Clearing it needs either a notional-denominated dust floor, which
`reconcile_positions` cannot compute because it has no prices, or an explicit
operator adoption of the broker's book. It is documented and pinned by a test
rather than papered over, because the alternative is a tolerance wide enough to
hide a genuine phantom position — and that is the one thing reconciliation
exists to catch.

**For a crypto desk this is a blocker.** For an equity desk it does not arise.

---

## The equity round trip — 2026-09-24, 13:36 UTC

Run at the open with `scripts/paper_round_trip.py`. **14/14 steps passed**, and
the two paths the crypto order could not reach are now proven.

**Brackets are held by the venue.** One share of XLF entered at $54.40 with a
bracket, and the legs were read back from `/v2/orders?nested=true` rather than
trusted from the submit response:

```
[PASS] venue holds protective legs — 2 leg(s): ['limit', 'stop']
[PASS] a stop is among them — limit@59.84, stop@48.96
```

This is the one that mattered. The reason the desk sends stops to the venue
instead of holding them in memory is that a stop in this process does not exist
during a crash — and a bracket that silently fails to attach turns a bounded
loss into an unbounded one while looking like a successful entry. It attaches.

**Equity mechanics are clean.** No time-in-force problem (DAY is valid), no
symbol-spelling problem (plain tickers), and no fee-in-kind problem — the
position matched the fill exactly, so `XLF reconciles — clean` with the default
absolute tolerance and no relative allowance at all.

The idempotency guarantee held again: re-polling recognised every fill as a
duplicate and booked nothing twice.

## Bug 4 — the fill poller crashed on a naive clock

```
TypeError: Cannot compare tz-naive and tz-aware timestamps
  axiom/desk/fills.py:187 in poll
```

The poller mixed three sources of time and they did not agree:

| source | timezone |
|---|---|
| stored watermark | **aware** — rebuilt from integer microseconds with an explicit UTC |
| cold-start watermark | **inherits the caller's** — `now - cold_start` |
| venue fill timestamps | **aware** — Alpaca sends an offset |

So `max(watermark, fill.timestamp)` raised, but only when a cold start, a naive
`now`, and at least one fill coincided. The comparison sits inside the loop over
fills, so every poll of a quiet window passed — including the desk run on
2026-09-01, which polled zero fills and reported success. The desk's own clock
produces naive UTC, so this would have fired on the first fill the desk ever
polled for itself.

Fixed by normalising at the boundary rather than at each comparison: the
alternative is remembering to convert at every site, and the site that gets
forgotten is the one that crashes the polling loop in production. Three
regression tests, each verified to fail against the old code.

## What is still unproven

**The account holds a position the desk did not create.** 10 SPY, bought
2026-08-10, before any of this. Reconciliation correctly reports it as an
unknown position and refuses to trade. That is the system working, and it needs
an operator decision — flatten it, or record it as an opening balance — rather
than a code change.

---

## The general lesson

Four bugs, four different layers: a request parameter, a symbol mapping, an
accounting assumption, and a timezone convention. Every one passed its unit
tests. Every one was found by a real order — three by the first crypto order,
the fourth by the first equity fill three weeks later.

The fourth is the most instructive, because it had been reached before and not
triggered: the 2026-09-01 desk run polled fills successfully, but polled *zero*
of them, and the crash lives inside the loop body. A path that runs is not the
same as a path that is exercised.

A stub returns what the code expects. That makes it a good regression test and a
poor discovery tool, and it is worth being precise about which of the two you
have. The regression tests added alongside these fixes are grouped under
`TestLiveApiRegressions` for exactly that reason — so it stays obvious that they
were written *after* the API disagreed, not before.
