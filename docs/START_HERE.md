# Start here

This page assumes you have never used a terminal. Every command is written out
in full. **Copy the whole line, paste it, press Enter, wait for it to finish
before doing the next one.**

Lines starting with `#` are notes for you — you can paste them too, the computer
ignores them.

---

## Step 0 — open a terminal

* **Mac:** press `Cmd` + `Space`, type `Terminal`, press Enter.
* **Windows:** press the Start button, type `PowerShell`, press Enter.

A window with text in it opens. That is the terminal. You type commands there.

---

## Step 1 — get the code onto your computer

Paste this **one line at a time**:

```
git clone https://github.com/tta225/AI-Hedge-Fund.git
```

```
cd AI-Hedge-Fund
```

> **If `git clone` says the folder already exists**, you already downloaded it.
> Just run `cd AI-Hedge-Fund` and then `git pull`.

---

## Step 2 — install it

```
pip install -e ".[web]"
```

Wait for it. It prints a lot of text and ends with something like
`Successfully installed axiom-0.1.0`.

> **If it says `pip: command not found`**, try `pip3` instead of `pip` in that
> command, and in every command below.

---

## Step 3 — open the app

```
python -m axiom.cli web
```

You will see:

```
AXIOM console → http://127.0.0.1:8000
Press Ctrl+C to stop.
```

**Now open your web browser and go to:** http://127.0.0.1:8000

That is the app. Leave the terminal window open — closing it turns the app off.

To stop the app: click the terminal window and press `Ctrl` and `C` together.

---

## What you are looking at

Three tabs across the top:

| Tab | What it does |
|---|---|
| **Dashboard** | The chart, with the ICT levels drawn on it |
| **Backtest** | "If I had traded this strategy, what would have happened" |
| **Data Sources** | Whether the app can reach real market prices right now |

### The coloured bar at the very top

This is the single most important thing on the screen.

* **Green — `REAL MARKET DATA`** → the numbers came from prices that actually
  traded.
* **Orange — `⚠ GENERATED DATA`** → the numbers came from a made-up price series
  the computer invented. **These are not results.** They only prove the software
  runs without crashing. Never make a decision from an orange screen.

---

## Step 4 — check whether real data works

Click the **Data Sources** tab. Each source shows **OK** (green) or **FAIL** (red).

You can also check from the terminal. Stop the app first (`Ctrl` + `C`), then:

```
python -m axiom.cli data-check
```

### Right now, on this project, all market data sources say FAIL

The message looks like this:

```
Alpaca
  ✗ Could not reach Alpaca: Tunnel connection failed: 403 Forbidden
```

**Your API keys are fine.** This is not a key problem. The message means the
*network* refused to let the app talk to Alpaca's computers at all — it was
blocked before your key was ever used.

Think of it like this: the key is the right key, but the road to the front door
is closed.

#### How to fix it

The app is running inside a Claude Code environment that has a list of websites
it is allowed to visit. Anything not on that list is refused before your key is
ever used.

**Market data** (Alpaca, Coinbase, Yahoo):

```
data.alpaca.markets
paper-api.alpaca.markets
api.exchange.coinbase.com
query1.finance.yahoo.com
```

**Your Hugging Face bucket.** `huggingface.co` is usually already allowed, but
that host only serves the *file list*. The actual **file contents** come from
completely different addresses, which is why you can see your 411 files and
still not download one:

```
cas-server.xethub.hf.co
transfer.xethub.hf.co
xethub.hf.co
s3.hf.co
cdn-lfs.hf.co
cdn-lfs-us-1.hf.co
```

The first one, `cas-server.xethub.hf.co`, is the one that actually blocks the
download — the rest are the other routes Hugging Face may use. Add all six.

If the setting accepts wildcards, this single line replaces all six:

```
*.hf.co
```

To add them:

1. Go to **claude.ai/code**
2. Click the **cloud icon** just above the message box. It shows an environment
   name — for this project it must be **`AI Hedge Fund`**, not `Default`.
3. Hover over that environment, click the **gear icon**
4. Find the **network** or **allowed domains** setting
5. Add the addresses above, one per line
6. Save
7. **Start a brand new session.** This part is not optional — the setting is only
   read when a session starts, so an existing one will keep failing.

### Checking it worked

In the new session:

```
python -m axiom.cli data-check --bucket Tta225/OHLCV-1m-bucket
```

That confirms the file *list*. To confirm the **bytes**, pull one small window:

```
python -c "from axiom.data import HFBucketProvider; from axiom.core.types import get_instrument; from axiom.data.base import BarRequest; p=HFBucketProvider('Tta225/OHLCV-1m-bucket', prefix='data/'); print(p.fetch_bars(BarRequest.lookback(get_instrument('SPY'),'1m',5)).describe())"
```

If that prints a line with a bar count, the bytes are flowing. If it fails and
mentions `xethub`, the hosts above are still blocked.

If you cannot find that setting, it may not be available on your plan. In that
case run the app on your own computer instead (Steps 1–3 above), where there is
no such restriction and your keys will work immediately.

---

## About your Hugging Face dataset

You said the dataset is in your buckets. Here is exactly what was checked and
what came back:

* Your token is **valid**. It signs in as `Tta225` and has read permission.
* Asking for `Tta225/OHLCV-1m-bucket` **while signed in as you** returns
  **404 Not Found** — that is the Hub saying "no such dataset", not "you are not
  allowed".
* Listing everything on your account returns **0 datasets, 0 models, 0 spaces**.

So the app is not failing to see a dataset that is there. As far as the Hub is
concerned, nothing has been uploaded to that account yet.

**A "bucket" on the Hugging Face website is not the same as a dataset repository.**
Files sitting in a storage bucket, or in a folder you have not published, are not
visible through the datasets API — which is what the app uses.

### What to do

Go to your dataset page in your browser and **copy the address bar**. It will
look like:

```
https://huggingface.co/datasets/SOMETHING/SOMETHING-ELSE
```

Send me that link. Then we can test the exact name:

```
python -m axiom.cli data-check --dataset PASTE-THE-PART-AFTER-datasets/-HERE
```

For example, if the address is
`https://huggingface.co/datasets/Tta225/my-bars`, the command is:

```
python -m axiom.cli data-check --dataset Tta225/my-bars
```

You will get one of two clear answers:

* `✓ authenticated as Tta225; 'Tta225/my-bars' resolves (private)` → it works.
* `✗ ... did not resolve` → that name does not exist on the Hub.

### If you have not actually uploaded it yet

1. Go to https://huggingface.co/new-dataset
2. Give it a name, choose **Private**, click **Create dataset**
3. Click **Files** → **Add file** → **Upload files**, and upload your CSV or
   Parquet files
4. Then run the `data-check --dataset` command above with the new name

---

## Using the app without real data

Everything still runs. Tick the **Generated bars** box and press **Analyse**.

This is genuinely useful — it shows you what the app does and proves the maths
works. It just is not evidence about any real market, and the orange banner will
say so the whole time. That is deliberate.

---

## Command reference

Every one of these is run from inside the `AI-Hedge-Fund` folder.

```
# Open the browser app
python -m axiom.cli web
```

```
# Check whether real data is reachable
python -m axiom.cli data-check
```

```
# Check a specific Hugging Face dataset
python -m axiom.cli data-check --dataset your-name/your-dataset
```

```
# Save your API keys (asks you to paste them, hides them as you type)
python -m axiom.cli setup
```

```
# The text version of the dashboard
python -m axiom.cli terminal --synthetic
```

```
# Run a backtest in the terminal
python -m axiom.cli backtest --strategy silver-bullet --symbol ES --days 120 --synthetic
```

---

## When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `command not found: python` | Python is not installed | Install it from python.org, then start again at Step 2 |
| `pip: command not found` | Same thing | Use `pip3` instead of `pip` |
| `No such file or directory` | You are in the wrong folder | Run `cd AI-Hedge-Fund` first |
| `Address already in use` | The app is already running | Use `python -m axiom.cli web --port 8001` and visit http://127.0.0.1:8001 |
| `Tunnel connection failed: 403` | The network blocked it | See "How to fix it" above — this is not a key problem |
| The browser page is blank | The app is not running | Check the terminal window is still open and shows no error |

---

## One rule worth keeping

If the banner at the top of the screen is **orange**, you are looking at made-up
data. It does not matter how good the numbers are. Nothing on an orange screen is
a reason to risk money.
