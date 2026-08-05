# Connecting real market data

Everything in AXIOM runs offline on generated bars. None of it is *evidence*
until it runs on real market history. This is how you connect the three sources
that are wired up.

---

## ⚠️ Never paste credentials into a chat

Not into a Claude conversation, not into a commit message, not into a code
comment. Anything in a transcript can end up in logs, and anything in a file can
end up in a repository. Both are hard to undo — the only real fix for a leaked
key is to revoke it.

Credentials belong in **environment variables**, set through your Claude Code
environment settings or your shell. Nothing below ever asks you to type a secret
where it would be stored.

**Environment variables are injected when the container starts.** A secret saved
mid-session will not appear until you **start a new session**. If a key seems to
be ignored, that is almost always why — verify with:

```bash
python -m axiom.cli data-check
```

---

## 1. Alpaca — US equities and crypto

The fastest route to real bars. Free tier is enough.

### Get keys

1. Sign in at [app.alpaca.markets](https://app.alpaca.markets)
2. The **paper trading** dashboard is sufficient — market data does not require
   a funded account
3. Generate an API key; you get a **Key ID** and a **Secret Key** (the secret is
   shown once)

### Set them

```bash
APCA_API_KEY_ID=<your key id>
APCA_API_SECRET_KEY=<your secret key>
```

`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` also work.

### Verify, then use

```bash
python -m axiom.cli data-check
python -m axiom.cli analyse --symbol SPY --timeframe 15m --days 60
python -m axiom.cli backtest --strategy silver-bullet --symbol SPY --days 365
```

### The feed caveat that matters for ICT

The free feed is `iex` — **one exchange**, not the consolidated tape. Its volume
is a fraction of what actually traded, and it can miss the prints that set a
high or low.

That is a bigger problem here than for most strategies. Liquidity pools, equal
highs, and sweeps are all claims about *where price traded*. A level derived
from IEX-only bars is approximate. Fine for research; do not treat an
IEX-derived sweep as ground truth.

`sip` is the full consolidated feed and needs a paid subscription:

```python
AlpacaProvider(feed="sip")
```

---

## 2. Hugging Face — your own dataset

For `Tta225/OHLCV-1m-bucket`, which is **private** (confirmed: a public dataset
resolves anonymously, this one returns 401).

### Get a token

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **New token** → type **Read** (write access is not needed and should not be
   granted)
3. If the dataset lives under an organisation, make sure the token's role has
   read access to it

### Set it

```bash
HF_TOKEN=<your read token>
```

### Install the extra

```bash
pip install -e ".[huggingface]"
```

### Inspect before loading — always

```python
from axiom.data import HuggingFaceProvider

provider = HuggingFaceProvider("Tta225/OHLCV-1m-bucket")
print(provider.inspect())
```

This prints row count, date span, and column names. Run it first: it turns a
schema mismatch into one clear line instead of a confusing failure deep inside a
backtest.

### Load a series

```python
series = provider.series("ES", "1m")
print(series.describe())
```

**If the column names do not match**, map them — no need to reshape the dataset:

```python
provider = HuggingFaceProvider(
    "Tta225/OHLCV-1m-bucket",
    column_map={"Open": "open", "Datetime": "timestamp"},
    symbol_column="ticker",      # if one dataset holds several instruments
    timestamp_column="Datetime", # if auto-detection picks the wrong column
)
```

Timestamps are auto-detected: ISO strings parse directly, and integer epochs are
resolved to seconds, milliseconds, or nanoseconds from their magnitude.

### Two decisions you have to make

**Is this observed market data, or generated?** The adapter defaults to
`DataKind.REAL`. If any of it is synthetic, augmented, or reconstructed, say so:

```python
from axiom.core.provenance import DataKind
provider = HuggingFaceProvider("Tta225/OHLCV-1m-bucket", kind=DataKind.SYNTHETIC)
```

The provenance system is only as honest as what it is told at the boundary.
Mislabel it once and every backtest downstream reports fiction as evidence.

**1-minute bars are the wrong timeframe for most of this engine.** The HMM
refits on rolling windows and a year of 1-minute data is ~370,000 bars — that
will not finish in reasonable time. Resample:

```python
regime_series = series.resample("15m")   # ICT structure, regimes
entry_series  = series                   # keep 1m for execution detail
```

Resampling upward is safe and provenance-preserving. Resampling *down* is
refused by design — it would invent bars that never traded.

---

## 3. ICT Knowledge Library — the methodology reference

```bash
git clone https://github.com/SrsBlack/ict-knowledge-library.git
```

Not a runtime dependency. It is the canonical source AXIOM's ICT definitions
were reconciled against, and the reason four errors were caught — including the
NY AM killzone being an hour early. Consult it before changing any threshold in
`ICTConfig`, and update `docs/ICT_METHODOLOGY.md` when you do.

---

## Provider precedence

`default_registry()` tries, in order:

| Priority | Provider | When it is used |
|---|---|---|
| 1 | CSV | when `data_root` is given — fastest, fully under your control |
| 2 | **Alpaca** | when keyed; silently skipped otherwise |
| 3 | yfinance | last resort; unofficial endpoint, do not depend on it |

**Synthetic is deliberately absent.** Falling back to generated bars when a real
feed fails is exactly how a fabricated backtest happens. You have to ask for it
explicitly — `--synthetic` on the CLI, or `research_registry()` in code.

---

## Once data is connected

The point of all of this is the measurement that has not happened yet:

1. **Base rates.** How often does a swept pool actually reverse? How often is an
   unmitigated FVG revisited? What is the order-block hit rate by quality tier?
2. **Replace the priors.** The confluence weights in `ICTConfig` are defensible
   guesses, not estimates. Real data replaces them with measured numbers.
3. **Then, and only then, run the strategy search** — and read the deflated
   Sharpe, not the raw one.
