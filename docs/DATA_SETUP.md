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
python -m axiom.cli data-check --dataset Tta225/OHLCV-1m-bucket   # also verify a HF dataset
```

`data-check` probes each source rather than just looking for keys. It reports
`credentialed` and `reachable` as separate lines, because they are separate
things — see [When the keys are fine and the network is not](#when-the-keys-are-fine-and-the-network-is-not).

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
| `Could not reach …: Tunnel connection failed: 403` | The key is irrelevant — the network refused to route to the host. See below |

The middle row is the one that wastes days, so `data-check` looks for it
explicitly and prints the name it found alongside the name to use. It reads
names only, never values.

**The catch that costs people the most time:** environment variables are
injected when the container *starts*. A secret saved mid-session does not appear
until you **begin a new session** — which looks exactly like a rejected key.

---

## When the keys are fine and the network is not

In a sandboxed environment, outbound HTTPS goes through a policy-enforcing
proxy, and a host that is not on the allowlist fails the CONNECT with **403
Forbidden** before any request is sent. Perfectly valid credentials produce:

```
Alpaca
  ✗ Could not reach Alpaca: Tunnel connection failed: 403 Forbidden
```

This is not a credential problem and no amount of re-pasting the key fixes it.
The market data hosts that need to be allowed are:

| Host | Used by |
|---|---|
| `data.alpaca.markets`, `paper-api.alpaca.markets` | Alpaca |
| `api.exchange.coinbase.com` | Coinbase |
| `huggingface.co` | Hugging Face datasets |
| `query1.finance.yahoo.com` | yfinance |

Add them to the environment's network policy — for Claude Code on the web, in
the environment settings alongside the environment variables. Then start a new
session.

**Why `data-check` reports `credentialed` and `reachable` separately:** a
provider's `is_available()` tests credentials and imports, never the network,
and `CoinbaseProvider.is_available()` is unconditionally `True` because it needs
no credentials at all. Collapsing the two into one "usable" line reported
Coinbase as usable on a network that refused to route to it. They are different
questions and the output now asks both.

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

> **`Tta225/OHLCV-1m-bucket` is a Storage Bucket, not a dataset.** This page has
> been wrong about it twice, and both errors are worth recording because they
> are the same mistake in different clothes.
>
> *First*, it claimed the repo was "confirmed private" because an anonymous
> request returned 401. That does not follow: the Hub returns 401 anonymously
> for a private repo **and** for one that is absent, precisely so absence cannot
> be probed.
>
> *Second*, after checking authenticated and getting 404 from
> `/api/datasets/...`, this page concluded the repo did not exist. Also wrong.
> `/api/buckets/Tta225/OHLCV-1m-bucket` returns **200**: it is public, holds
> **411 monthly Parquet files** spanning **1992-01 to 2026-03**, and totals
> **87.7 GB**. The 404 was correct and the inference from it was not — the repo
> was never a dataset, and the datasets API cannot see any other repo type.
>
> The lesson both times: a negative result from one API is evidence about that
> API, not about the world.
>
> **Buckets are a different repo type from datasets.** They are S3-like,
> non-versioned, mutable object storage, and `datasets.load_dataset()` cannot
> read them at all. Use [`HFBucketProvider`](#4-hugging-face-storage-buckets)
> below, or `axiom data-check --bucket <id>`.

The rest of this section is about **datasets**. For buckets, skip to
[Hugging Face Storage Buckets](#4-hugging-face-storage-buckets).

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

## 4. Hugging Face Storage Buckets

A **Storage Bucket** is a different repo type from a dataset: S3-like,
non-versioned, mutable object storage. The distinction matters more than it
sounds:

* `datasets.load_dataset()` **cannot read a bucket**, at all.
* `https://huggingface.co/api/datasets/<id>` returns **404** for a bucket.
* The bucket lives at `https://huggingface.co/**buckets**/<id>`, and its API is
  `https://huggingface.co/api/buckets/<id>`.

Reading that 404 as "the repo does not exist" is a mistake this document made,
and it is worth stating plainly: **a negative result from one API is evidence
about that API, not about the world.**

### Check a bucket

```bash
python -m axiom.cli data-check --bucket Tta225/OHLCV-1m-bucket
```

```
HF bucket
  ✓ bucket 'Tta225/OHLCV-1m-bucket' resolves (public): 413 files, 87.7 GB
    objects : 411 parquet files (87.7 GB)
    span    : 1992-01 → 2026-03 (from filenames, not from the data)
```

### Load bars from one

```python
from axiom.data import HFBucketProvider

provider = HFBucketProvider("Tta225/OHLCV-1m-bucket", prefix="data/")
print(provider.inspect())          # listing only — no download
series = provider.fetch_bars(request)
```

**Only the months you ask for are downloaded.** The adapter reads the object
listing (metadata, free) and parses the calendar month out of each filename, so
a six-week window pulls 2 files (~0.7 GB) rather than all 411 (~87.7 GB). Files
are cached under `data/cache/hf-bucket/`, so a repeated backtest costs nothing.

| Window | Files | Downloaded |
|---|---|---|
| 6 weeks | 2 | 0.67 GB |
| 1 year | 12 | 4.0 GB |
| everything | 411 | 87.7 GB |

If your filenames are not `..._YYYY-MM.parquet`, pass your own
`filename_pattern` — a regex with named groups `year` and `month`. Files that do
not match are always included rather than skipped, because silently dropping
history is worse than a wasted download.

### Buckets need more hosts than datasets do

Bucket **metadata** comes from `huggingface.co`, but the **bytes** are served by
Xet and the S3 gateway. On a restricted network the listing succeeds and the
download fails — which looks confusing until you know the hosts differ:

```
cas-server.xethub.hf.co
transfer.xethub.hf.co
s3.hf.co
cdn-lfs.hf.co
```

Allow those alongside `huggingface.co`. `HFBucketProvider` names them in its
error message when a download fails, so the failure diagnoses itself.

> **Not verified end-to-end.** The listing, month-selection and caching paths
> are tested. The download path could not be exercised where this was written —
> those Xet hosts were blocked — so the real files' **column schema is
> unconfirmed**. The adapter is schema-flexible (see `column_map` and
> `timestamp_column`) and `inspect()` deliberately does not claim to know the
> columns. Fetch a narrow window first and check what comes back.

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
