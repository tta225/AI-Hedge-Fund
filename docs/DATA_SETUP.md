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

---

# The simplest way: `axiom setup`

Run one command and paste each key when prompted. Nothing you type is displayed,
and nothing lands in your shell history.

```bash
cd /home/user/AI-Hedge-Fund
pip install -e .
python -m axiom.cli setup
```

**Use the absolute path.** On this machine `~` is `/root`, while the project
lives under `/home/user/` — so `cd ~/AI-Hedge-Fund` fails with "no such file or
directory". If you are unsure where the project is:

```bash
find / -name pyproject.toml -path '*AI-Hedge-Fund*' 2>/dev/null
```

### On your own computer instead

The code lives on a branch, not `main`, so the checkout line is not optional:

```bash
git clone https://github.com/tta225/AI-Hedge-Fund.git
cd AI-Hedge-Fund
git checkout claude/ai-hedge-fund-platform-vxebwq
pip install -e .
python -m axiom.cli setup
```

It asks for three things in turn:

```
Alpaca Key ID (starts with PK, about 20 characters)
  paste here:

Alpaca Secret Key (about 40 characters)
  paste here:            <- hidden as you type

Hugging Face token (starts with hf_ — leave blank if your dataset is public)
  paste here:            <- press Enter to skip
```

Then it writes `.env`, locks it to owner-only permissions, and immediately
verifies the keys. Re-run it any time; press Enter at a prompt to keep an
existing value.

### Checking later

```bash
python -m axiom.cli data-check
```

### Doing it by hand instead

If you would rather write the file yourself, create `.env` in the project root:

```
APCA_API_KEY_ID=PK7XXXXXXXXXXXXXXXXX
APCA_API_SECRET_KEY=abcdefghijklmnopqrstuvwxyz0123456789ABCD
```

`NAME=value`, one per line, no quotes, no spaces around the `=`.

You should see:

```
Credentials
  .env file : /path/to/AI-Hedge-Fund/.env
    ✓ Alpaca key id  present (20 chars)
    ✓ Alpaca secret  present (40 chars)
```

It reports **presence and length only** — never the value, not even a prefix.

### Why this is safe

* `.env` is in `.gitignore`, so `git add -A` will not pick it up.
* The loader **refuses to run** if it finds `.env` tracked by git, and tells you
  how to untrack it. A tracked credential file is one commit away from being
  published.
* If a platform environment variable is also set, **it wins** — a stale file can
  never silently shadow a real secret.

---

## The alternative: platform environment variables

If you prefer to set them in your Claude Code environment settings instead, that
works too and takes precedence over `.env`.

There is no settings page for this and no direct URL. The path is:

1. Go to **claude.ai/code**
2. Click the **cloud icon** in the row just above the message box — it shows the
   environment name. For this project that is **`AI Hedge Fund`**, *not*
   `Default`; saving to the wrong environment is the most common reason a key
   never shows up
3. Hover that environment → click the **gear icon**
4. Find the **Environment variables** box
5. Enter them in `.env` format, one per line:
   ```
   APCA_API_KEY_ID=PK7XXXXXXXXXXXXXXXXX
   APCA_API_SECRET_KEY=abcdefghijklmnopqrstuvwxyz0123456789ABCD
   ```
6. Save
7. **Start a new session**

**Use paper keys only here** (Key ID starting `PK`). Anthropic's own docs are
blunt about this: *"Anyone who uses the environment can read the values, and
cloud environments have no dedicated secrets store, so don't add API keys or
other credentials."* A paper key in a personal environment is a bounded risk —
it is readable by your account and cannot touch real money. A live `AK` key is
not, and does not belong here.

### The three ways this fails, and how to tell them apart

| `data-check` says | What actually happened |
|---|---|
| `not set` | The variable never arrived — saved to a different environment, or saved after this session started |
| `not set` **plus a rename hint** | The key *is* there under a name nothing reads. Rename it; do not re-copy the key |
| `rejected the credentials: HTTP 401` | It arrived and Alpaca refused it — now it is genuinely the key |

The middle row is the one that wastes days, so `data-check` looks for it
explicitly and prints the name it found alongside the name to use. It reads
names only, never values.

**The catch that costs people the most time:** environment variables are
injected when the container *starts*. A secret saved mid-session does not appear
until you **begin a new session** — which looks exactly like a rejected key.

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

Put them in `.env` as shown above. `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are
accepted as alternative names, since Alpaca's own docs use both.

The **Key ID** starts with `PK` (paper) or `AK` (live) and is ~20 characters.
The **Secret Key** is ~40 characters and is shown **once** at creation — if you
did not copy it, generate a new pair rather than hunting for it.

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

### Easiest option: make the dataset public

This removes the token requirement entirely — no credential to manage at all.

1. Go to `huggingface.co/datasets/Tta225/OHLCV-1m-bucket`
2. **Settings** tab
3. Scroll to the bottom → **Change dataset visibility** → **Public**

Then just:

```python
provider = HuggingFaceProvider("Tta225/OHLCV-1m-bucket")   # no token needed
```

**Is that safe?** It depends entirely on what is in the dataset:

| Contents | Verdict |
|---|---|
| Raw OHLCV bars | **Public is fine.** Price history is not proprietary — the exchanges sell it and dozens of identical datasets are already public. |
| Your own labels, annotations, or engineered features | **Keep it private.** That is your research, and it is exactly the part with value. |
| Anything licensed from a vendor | **Keep it private.** Redistribution usually breaches the licence. |

If it is plain 1-minute bars, making it public is the least complicated option
and costs you nothing. If you added anything of your own, use a token.

### Get a token (if keeping it private)

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **New token** → type **Read** (write access is not needed and should not be
   granted)
3. If the dataset lives under an organisation, make sure the token's role has
   read access to it

### Set it

Add one line to the same `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
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
