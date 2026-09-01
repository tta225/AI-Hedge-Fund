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

## What is still unproven

**The equity path.** US markets were closed at the time of the run (22:35 ET),
so the round trip was done on crypto. Equities avoid all three bugs above — DAY
is valid, positions come back as plain tickers, fees are cash — but "should be
fine" is exactly the reasoning that produced these three, and the equity path
deserves the same treatment during market hours.

**Brackets.** No protective legs were attached to the test order, so the
bracket/OTO construction remains stub-tested only.

**The account holds a position the desk did not create.** 10 SPY, bought
2026-08-10, before any of this. Reconciliation correctly reports it as an
unknown position and refuses to trade. That is the system working, and it needs
an operator decision — flatten it, or record it as an opening balance — rather
than a code change.

---

## The general lesson

Three bugs, three different layers: a request parameter, a symbol mapping, and
an accounting assumption. Every one passed its unit tests. Every one was found
by the first real order.

A stub returns what the code expects. That makes it a good regression test and a
poor discovery tool, and it is worth being precise about which of the two you
have. The regression tests added alongside these fixes are grouped under
`TestLiveApiRegressions` for exactly that reason — so it stays obvious that they
were written *after* the API disagreed, not before.
