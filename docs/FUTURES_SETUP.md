# Futures: data and trading

Everything on this page is about **futures** — ES, NQ, CL, GC. It is separate
from equities because almost nothing carries over: different brokers, different
data vendors, different symbols, and a different cost structure.

Read the summary table, then only the section you need.

| What you want | Use | Cost | Works on a Mac |
|---|---|---|---|
| Trade futures from this platform | **Tradovate** | free demo; funded account to trade | **yes** |
| Trade through NinjaTrader | NinjaTrader ATI bridge | you already pay for it | **no — Windows only** |
| Historical futures data for backtests | **Databento** or **Massive** | pay per GB / monthly plan | yes |
| Live futures quotes | Tradovate, Databento or Massive | included / paid | yes |

---

## First, the thing that surprises people

**Your Hugging Face bucket has no futures in it.** It is 411 files of
1-minute equity and ETF bars — 22,057 tickers, 1992 to 2026 — and no futures
contracts at all.

This is worth stating plainly because the tickers look like they disagree.
Asking the bucket for `ES` returns data. It is not the E-mini S&P: it is
**Eversource Energy**, a utility trading around $71. `CL` returns
**Colgate-Palmolive** at about $89, not crude oil at $75. Both are real
companies with real prices, so nothing errors and nothing looks wrong — a
backtest on "ES" from the bucket runs to completion and produces a number.

That number is about a utility stock.

So: futures data has to come from somewhere else. That is what the rest of this
page is for.

---

## Option A — Tradovate (recommended, and the only one that works on a Mac)

Tradovate and NinjaTrader are the same company (NinjaTrader Group). Tradovate is
the browser-based side, and unlike NinjaTrader it exposes a **REST API** —
which is why this platform can talk to it from any operating system.

### What you get

* Live CME futures quotes, included with the account
* Order placement with acknowledgements
* A free demo account with simulated fills, indefinitely

### Step 1 — get API access

1. Log in at **https://trader.tradovate.com**
2. Go to **Application Settings → API Access**
3. Request API access if you have not already. Approval is not instant.
4. When approved you get a **key ID** (a number) and a **secret**

### Step 2 — save the credentials

```
python3 -m axiom.cli setup --futures
```

It asks for four things and hides the two secret ones as you type:

| Prompt | What to paste |
|---|---|
| Tradovate username | the email you log in with |
| Tradovate password | your login password |
| Tradovate API key ID | the number from API Access |
| Tradovate API secret | the string issued alongside it |

### Step 3 — check it

```
python3 -m axiom.cli broker --venue tradovate
```

A working demo account prints:

```
tradovate  (is_live=False)
  ✓ demo: account 'DEMO12345' (id 42), 0 open position(s)
```

### Step 4 — allow the network through

If you get `could not reach ... 403`, the hosts are blocked rather than your
credentials being wrong. Add these to your environment's allowed domains, the
same way you added the Hugging Face ones:

```
demo.tradovateapi.com
live.tradovateapi.com
md.tradovateapi.com
```

Then **start a new session** — the setting is only read at session start.

### Going live

`--live` swaps to the real-money host. Two separate gates stand in the way, on
purpose:

```
python3 -m axiom.cli broker --venue tradovate --live
```

1. The venue refuses to be constructed against the live host without an explicit
   opt-in, so a config copied from somewhere else cannot arm real trading by
   itself.
2. The order router refuses to route to any live venue unless it is in live mode
   **and** carries a confirmation phrase.

Neither gate is clever. They are just two deliberate acts instead of one.

Every order this platform sends to Tradovate is flagged `isAutomated`. That is a
CME Group requirement for automated orders, not an option — so it is set
unconditionally rather than being a setting someone can turn off.

---

## Option B — NinjaTrader (Windows only)

**If you are on a Mac, skip this section.** NinjaTrader's Automated Trading
Interface is a Windows platform feature. There is no Mac version, no wrapper,
and no workaround — Tradovate reaches the same exchanges and the same account,
with a real API.

The adapter is built and tested, so it is there when you want it on a Windows
machine. Here is what it does and what to know.

### How it works

NinjaTrader has no REST API. The supported way in is a **file bridge**: this
platform writes semicolon-delimited instruction files into an `incoming` folder,
NinjaTrader reads them the instant they appear, and reports back by writing
files into an `outgoing` folder.

### Setup

1. In NinjaTrader: **Tools → Options → Automated trading interface**
2. Tick **Enable**
3. Note the folder — usually `Documents\NinjaTrader 8`, containing `incoming`
   and `outgoing`
4. Save your account name:

```
python3 -m axiom.cli setup --futures
```

5. Check it:

```
python3 -m axiom.cli broker --venue ninjatrader --live
```

### Two things to know before using it

**It is always treated as live.** A NinjaTrader Sim account and a funded account
differ only by the name in a config file — there is no structural difference this
platform can detect. So `is_live` is `True` unconditionally, and the router
demands live mode plus a confirmation phrase whatever account you name. That is
stricter than necessary for Sim101 and correct for the case that matters.

**Entry and stop are not simultaneous.** The OIF format has no bracket field:
NinjaTrader attaches stops through ATM strategy templates configured inside the
platform, which this bridge cannot create. So the entry goes first and the
protective orders follow milliseconds later. That window is small but real, and
it is exactly the window a gap happens in. If you trade gap-prone instruments
through this venue, attach protection with an ATM template in NinjaTrader rather
than relying on the follow-up orders.

The Tradovate venue does not have this problem — stop and target travel with the
entry in a single OSO request.

### If it says it cannot find anything

The names of NinjaTrader's outgoing state files were taken from secondary
sources, because NinjaTrader's documentation site is unreachable from the
environment this was built in. A wrong filename pattern **fails silently** — it
looks like a market with no fills rather than a broken adapter — so there is a
command that shows what NinjaTrader actually wrote:

```
python3 -c "from axiom.execution.ninjatrader import NinjaTraderVenue; print(NinjaTraderVenue(allow_live=True).discover())"
```

If the filenames that prints do not match the patterns at the top of
`src/axiom/execution/ninjatrader.py`, that is the bug — send me the list and it
is a two-minute fix.

---

## Option C — Databento (historical futures data)

Databento sells CME historical and live data directly. Unlike most vendors there
is no monthly minimum: you pay for the bytes you pull.

### What to buy

| Dataset | What it is | Use it for |
|---|---|---|
| `GLBX.MDP3` | CME Globex, all futures and options | ES, NQ, CL, GC — everything |

Ask for schema **`ohlcv-1m`** (1-minute bars). It is far cheaper than tick data
and matches what the rest of this platform already reads.

### Cost, roughly

1-minute bars are small. One year of ES 1-minute bars is on the order of tens of
megabytes, so a few dollars rather than a few hundred. Tick data for the same
period is thousands of times larger — do not start there.

### Getting a key

1. Sign up at **https://databento.com**
2. Go to **Portal → API Keys**
3. Create a key (it starts with `db-`)

Set it:

```
export DATABENTO_API_KEY=db-your-key-here
```

Allow the host:

```
hist.databento.com
live.databento.com
```

> **The Databento adapter is not built yet.** When you have the key, say so and
> it goes in next — the provider interface it plugs into already exists, which is
> what the Hugging Face bucket and Alpaca both use.

---

## Option D — Massive (formerly Polygon.io)

**Massive is Polygon.io.** The company rebranded in October 2025; same API, same
keys, same endpoints. The host moved to `api.massive.com` and `api.polygon.io`
still runs alongside it. If you already have a Polygon key, it works here — the
adapter reads `POLYGON_API_KEY` as well as `MASSIVE_API_KEY`.

### Getting a key

1. Sign up at **https://massive.com**
2. **Dashboard → API Keys**
3. Copy the key

```
python3 -m axiom.cli setup --futures
```

Allow the host:

```
api.massive.com
api.polygon.io
```

Check it:

```
python3 -m axiom.cli data-check
```

### Massive vs Databento

Both give you futures. They differ in one way that matters for backtesting.

| | Databento | Massive |
|---|---|---|
| Billing | per GB pulled, no minimum | monthly plan |
| Equities | yes | yes |
| Futures | yes | yes (separate plan) |
| Options | yes | yes |
| **Continuous contracts** | **yes, with roll rules** | **no** |

**That last row is the important one.** A futures contract expires, so a
multi-year chart is several contracts joined together. Databento does that
joining for you and this platform reports every join. Massive addresses one
contract at a time (`ESH6` is the March 2026 E-mini), so the adapter fetches a
**single front-month contract** and tells you which one it used.

It deliberately does **not** glue contracts together. The price jump where one
contract ends and the next begins is not a gain or a loss — nothing happened to
your position — and a data source that silently glued them would turn that jump
into a fake return in every backtest that crossed it.

**So:** for a backtest inside one contract's life (up to about three months),
they are equivalent. For anything longer, use Databento.

### What you can type

The platform understands both forms:

```
python3 -m axiom.cli analyse --symbol ES --timeframe 15m
```

`ES` is the root — the adapter looks up the current front-month contract.

```
python3 -m axiom.cli analyse --symbol ESH6 --timeframe 15m
```

`ESH6` names one specific contract: **ES** + **H** (March) + **6** (2026).

Month codes, because they are not obvious:

| F | G | H | J | K | M | N | Q | U | V | X | Z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |

### One thing to watch

The free tier is **5 requests per minute**. One backtest window can be several
requests, so a long lookback will hit that limit and stop with an HTTP 429. The
error says so explicitly rather than looking like a network fault.

---

## Futures symbols, briefly

Futures contracts expire, so a symbol names a specific one:

```
ES 03-26     E-mini S&P 500, March 2026
ESH6         the same contract, in the older code style
```

| Root | Contract | Tick | Tick value |
|---|---|---|---|
| ES | E-mini S&P 500 | 0.25 | $12.50 |
| NQ | E-mini Nasdaq 100 | 0.25 | $5.00 |
| CL | Crude oil | 0.01 | $10.00 |
| GC | Gold | 0.10 | $10.00 |

**Continuous contracts** stitch the expiring months into one long series so a
backtest can span years. The stitching is not free: the roll leaves a price gap
that is not a real return, and a backtest that treats it as one is measuring the
roll. This platform reports what it did at the roll rather than hiding it.

---

## Which to start with

If you are on a Mac and want to see futures working today: **Tradovate demo**.
It is free, it takes about ten minutes, and it gives you live quotes and
simulated fills without a funded account.

Databento matters once you want to *backtest* futures rather than watch them.
