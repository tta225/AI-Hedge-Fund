"""Cache a survivorship-free equity universe: the names that died, too.

Every equity cache in this project so far was built from a hand-written list of
tickers that are liquid *today*. That list cannot contain Silicon Valley Bank,
First Republic, Twitter, Activision, or the several hundred other US listings
that stopped trading during the sample — so no backtest run on it was ever given
the chance to own them, and every measured return is the return of a portfolio
that knew in advance which companies would survive.

This builds the other kind of universe. Alpaca's asset endpoint lists roughly
19,000 **inactive** US equities alongside the active ones, and the data API
still serves their bars, which stop on the day each name stopped trading. That
is enough to reconstruct membership as a function of time rather than as a
function of today.

The procedure:

1. Pull every US equity asset, active and inactive, on a major exchange.
2. Fetch daily bars for all of them in batches.
3. Derive each name's listing interval from the bars it actually printed —
   **not** from the venue's status field, because a name whose record says
   "active" but which last printed in 2021 is delisted for every purpose a
   backtest has.
4. Write a wide close/volume matrix plus a listings table.

The result is a matrix with holes in it, which is the honest shape: a name that
had not yet listed and one that has already died are both NaN, and
:class:`~axiom.data.universe.PointInTimeUniverse` knows which is which.

Names carrying Alpaca's ``_DELISTED`` suffix are skipped — the data API rejects
them outright, and the underlying ticker is usually present separately.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.data.universe import MAJOR_EXCHANGES, PointInTimeUniverse

_ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
#: Symbols per bar request. Alpaca accepts long symbol lists; this is kept
#: modest so one failure costs one batch rather than the whole run.
BATCH = 90
#: Seconds between requests, to stay well inside the free tier's rate limit.
PAUSE = 0.3


def _headers() -> dict[str, str]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY. The paper dashboard's "
            "keys are enough — this script only reads market data."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(url: str, headers: dict[str, str], retries: int = 6) -> Any:
    """GET with backoff, retrying transport failures as well as HTTP ones.

    Retrying only ``HTTPError`` was a real defect: a connection reset raises
    ``URLError``, which is not an ``HTTPError``, so a twenty-minute download
    of fifteen thousand symbols died on a single transient blip and discarded
    everything it had fetched. Over a job that long a reset is not an
    exceptional event, it is an expected one.
    """
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # A 4xx that is not a rate limit is a bad request; retrying it just
            # asks the same wrong question more slowly.
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            last = exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            # Reset by peer, DNS blip, truncated body. All transient.
            last = exc
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up after {retries} attempts: {last}")


def fetch_assets(headers: dict[str, str]) -> list[dict[str, Any]]:
    """Every US equity on a major exchange, living and dead."""
    assets: list[dict[str, Any]] = []
    for status in ("active", "inactive"):
        query = urllib.parse.urlencode({"status": status, "asset_class": "us_equity"})
        batch = _get(f"{_ASSETS_URL}?{query}", headers)
        assets.extend(batch)
        print(f"  {status}: {len(batch):,} assets")

    keep = [
        asset
        for asset in assets
        if asset.get("exchange") in MAJOR_EXCHANGES
        # Alpaca renames some delisted tickers with a suffix; the data API
        # rejects those names outright, and the plain ticker is usually
        # present as its own record.
        and "_" not in asset["symbol"]
        and "." not in asset["symbol"]
    ]
    print(f"  on major exchanges, usable symbol: {len(keep):,}")
    return keep


def fetch_bars(
    symbols: list[str],
    headers: dict[str, str],
    start: str,
    end: str,
    feed: str,
    checkpoint_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Daily bars for many symbols, paged, batched, and resumable.

    Each batch is written to its own file as it completes, and a batch whose
    file already exists is skipped. Fetching fifteen thousand symbols takes
    twenty minutes, and a job that long will meet a connection reset; without
    checkpoints the choice on failure is to lose everything or to write retry
    logic that eventually becomes this.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(symbols), BATCH):
        chunk = symbols[offset : offset + BATCH]
        shard = checkpoint_dir / f"batch-{offset:06d}.json" if checkpoint_dir else None

        if shard is not None and shard.exists():
            try:
                for symbol, bars in json.loads(shard.read_text()).items():
                    out.setdefault(symbol, []).extend(bars)
                continue
            except json.JSONDecodeError:
                # A shard written during a crash. Refetch rather than trust it.
                shard.unlink()

        batch: dict[str, list[dict[str, Any]]] = {}
        token: str | None = None
        while True:
            params = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": start,
                "end": end,
                "feed": feed,
                "limit": 10000,
                "adjustment": "all",
            }
            if token:
                params["page_token"] = token
            try:
                payload = _get(f"{_BARS_URL}?{urllib.parse.urlencode(params)}", headers)
            except (urllib.error.HTTPError, RuntimeError) as exc:
                # One bad symbol poisons its whole batch, and the API does not
                # say which. Skipping the batch loses at most BATCH symbols;
                # failing the run loses all of them.
                print(f"    batch at {offset} failed ({exc}); skipping")
                batch = {}
                break
            for symbol, bars in (payload.get("bars") or {}).items():
                batch.setdefault(symbol, []).extend(bars)
            token = payload.get("next_page_token")
            if not token:
                break
            time.sleep(PAUSE)

        if shard is not None:
            # Written atomically: a half-written shard read back on resume
            # would look like a symbol with truncated history, which is worse
            # than no shard at all.
            temporary = shard.with_suffix(".partial")
            temporary.write_text(json.dumps(batch))
            temporary.replace(shard)

        for symbol, bars in batch.items():
            out.setdefault(symbol, []).extend(bars)

        done = min(offset + BATCH, len(symbols))
        print(f"    {done:,}/{len(symbols):,} symbols, {len(out):,} with bars")
        time.sleep(PAUSE)
    return out


def to_frames(bars: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide close and volume matrices, NaN where a name was not trading."""
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for symbol, rows in bars.items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        index = pd.DatetimeIndex(pd.to_datetime(frame["t"], utc=True)).tz_localize(None).normalize()
        # A duplicated timestamp is a vendor artefact; keep the last print.
        frame = frame.assign(_i=index).drop_duplicates("_i", keep="last").set_index("_i")
        closes[symbol] = frame["c"].astype(float)
        volumes[symbol] = frame["v"].astype(float)
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/pit")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--feed", default="iex", choices=("iex", "sip"))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap the number of symbols fetched. 0 means every one.",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Refetch every batch instead of reusing checkpoint shards.",
    )
    parser.add_argument(
        "--supplement", default="data/known_delistings.txt",
        help="File of extra tickers to fetch beyond the asset endpoint. See the "
             "file's own header for why it is necessary and what it does not cover.",
    )
    args = parser.parse_args()

    headers = _headers()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Assets:")
    assets = fetch_assets(headers)
    active = {a["symbol"] for a in assets if a.get("status") == "active"}
    exchanges = {a["symbol"]: a.get("exchange", "") for a in assets}
    symbols = {a["symbol"] for a in assets}

    # Alpaca purges the asset record of some delisted names while still serving
    # their bars — Silicon Valley Bank, First Republic, Twitter, VMware and
    # Activision among them. They are undiscoverable from /v2/assets, and they
    # are exactly the large-cap failures whose absence flatters a backtest
    # most, so they are supplied by name.
    supplement = Path(args.supplement)
    if supplement.exists():
        extra = {
            token
            for line in supplement.read_text().splitlines()
            if not line.startswith("#")
            for token in line.split()
        }
        new = extra - symbols
        symbols |= extra
        print(f"  supplement: {len(extra)} tickers, {len(new)} not in the asset list")

    symbols = sorted(symbols)
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"\nBars for {len(symbols):,} symbols ({args.start} -> {args.end}, {args.feed}):")
    checkpoints = out / "_batches" if args.resume else None
    bars = fetch_bars(symbols, headers, args.start, args.end, args.feed, checkpoints)

    closes, volumes = to_frames(bars)
    print(f"\nMatrix: {closes.shape[0]:,} bars x {closes.shape[1]:,} symbols")

    universe = PointInTimeUniverse.from_frames(
        closes, volumes, active=active, exchanges=exchanges
    )
    dead = universe.delisted
    print(f"Listings: {len(universe):,}, of which {len(dead):,} stopped trading in-sample")

    # Gzipped CSV rather than parquet: the matrix is sparse and compresses
    # well, and it avoids making pyarrow a dependency of the research path.
    closes.to_csv(out / "closes.csv.gz")
    volumes.to_csv(out / "volumes.csv.gz")
    universe.save(str(out / "listings.csv"))
    print(f"\nWrote {out}/closes.csv.gz, volumes.csv.gz, listings.csv")

    # The headline the whole exercise exists to produce.
    dates = pd.date_range(closes.index[0], closes.index[-1], freq="YE")
    report = universe.survivorship_report(
        [d for d in dates if d > closes.index[0] + pd.Timedelta(days=400)],
        closes=closes, volumes=volumes, top_n=100,
    )
    if not report.empty:
        print("\nSurvivorship — of the 100 most liquid names on each date, how many")
        print("stopped trading within a year (a today-only universe deletes these):")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
