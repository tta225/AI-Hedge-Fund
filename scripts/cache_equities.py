"""Cache a daily equity universe for cross-sectional research.

**This universe is survivorship-biased by construction.** It is a list of
symbols that are liquid *today*, applied to a window starting six years ago, so
every name in it survived that window. Anything measured on it is therefore an
upper bound, and a strategy that cannot clear the bar even with that advantage
certainly cannot clear it without.

Removing the bias properly needs point-in-time index membership, which Alpaca
does not serve. Naming the bias is the honest alternative to pretending it is
not there.
"""

from __future__ import annotations

import argparse

from axiom.core.types import get_instrument
from axiom.data.alpaca import AlpacaProvider
from axiom.data.base import BarRequest, ProviderError
from axiom.data.providers import CSVProvider

#: Large, liquid US names spanning several sectors, all listed well before the
#: window opens. Breadth across sectors matters more than count: a universe of
#: one sector cannot express a cross-sectional ranking, it can only express a
#: bet on that sector.
#: A second universe, disjoint from the first, for confirming a hypothesis
#: selected on the first. Reusing the discovery universe to confirm a
#: discovery made on it measures nothing.
CONFIRMATION_UNIVERSE = [
    # Technology and communications
    "ACN", "AMAT", "LRCX", "KLAC", "MU", "ADI", "SNPS", "CDNS", "PANW", "FTNT",
    "ANET", "MSI", "ROP", "APH", "TEL",
    # Consumer
    "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "MAR", "BKNG", "GM", "F",
    "EBAY", "ORLY", "AZO", "KMB", "CL",
    # Financials and real estate
    "PNC", "USB", "TFC", "COF", "MET", "PRU", "AIG", "TRV", "PLD", "AMT",
    # Healthcare
    "AMGN", "GILD", "VRTX", "REGN", "ISRG", "SYK", "BSX", "MDT", "CI", "CVS",
    # Industrials, materials, energy, utilities
    "LMT", "RTX", "NOC", "GD", "DE", "EMR", "ETN", "ITW", "MMM", "LIN",
    "APD", "SHW", "PSX", "VLO", "MPC", "SO", "D", "AEP", "EXC", "SRE",
]

UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO",
    "TXN", "QCOM", "IBM", "NOW", "INTU",
    # Communications and consumer
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "AMZN", "TSLA", "HD", "MCD", "NKE",
    "SBUX", "LOW", "TGT", "COST", "WMT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "SPGI",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    # Industrials, energy, staples, utilities
    "CAT", "BA", "HON", "UPS", "GE", "XOM", "CVX", "COP", "SLB", "PG",
    "KO", "PEP", "PM", "NEE", "DUK",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--out", default="data/equities")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Fetch the disjoint confirmation universe instead.",
    )
    args = parser.parse_args()

    universe = CONFIRMATION_UNIVERSE if args.confirmation else UNIVERSE
    overlap = set(UNIVERSE) & set(CONFIRMATION_UNIVERSE)
    if overlap:
        raise SystemExit(f"universes must be disjoint; shared: {sorted(overlap)}")

    provider = AlpacaProvider()
    if not provider.is_available():
        raise SystemExit("Alpaca credentials not found; see docs/DATA_SETUP.md")

    ok = failed = 0
    for symbol in universe:
        try:
            series = provider.fetch_bars(
                BarRequest.lookback(get_instrument(symbol), args.timeframe, args.days)
            )
        except (ProviderError, KeyError) as exc:
            print(f"  {symbol:<6} FAILED: {str(exc)[:90]}")
            failed += 1
            continue
        path = CSVProvider.write(series, f"{args.out}/{symbol}_{args.timeframe}.csv")
        print(f"  {symbol:<6} {len(series):>5,} bars  {series.index[0].date()} "
              f"-> {series.index[-1].date()}  {path}")
        ok += 1

    print(f"\n{ok} cached, {failed} failed.")


if __name__ == "__main__":
    main()
