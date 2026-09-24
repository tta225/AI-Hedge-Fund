# Survivorship bias, finally measured — and it was small

*Run: 2026-08-31. Source: Alpaca IEX daily bars, 11,787 US equities including
1,018 that stopped trading in sample. Reproduce with
`python -m scripts.cache_pit_universe` then `python -m scripts.survivorship_study`.*

Six studies in this repository carried a version of the same sentence: *"the
universe is survivorship-biased, so every figure is an upper bound."* Honest,
and useless — it gave no idea whether the correction was worth 0.05 Sharpe or
0.5, and it functioned as a standing excuse for six null results.

The universe is now point-in-time and the caveat is now a number.

**The number is +0.013 Sharpe, and it is indistinguishable from zero.**

---

## 1. The universe

11,787 US equities on major exchanges, 2020-07 to 2026-08, each name's trading
life derived from the bars it actually printed rather than from a status flag.
1,018 of them stopped trading inside the sample.

Two things had to be got right for that number to mean anything.

**The panel unions trading days rather than intersecting them.** The existing
`Panel.from_series` inner-joins, which is correct for names that all survived
and catastrophic here: a company that delisted in 2022 shares no bars with one
that listed in 2023, so the intersection of a realistic universe collapses to
whichever handful spanned everything. That is survivorship bias reintroduced by
a join, and it would have been invisible.

**Delisting forces a liquidation.** Previously a dead name had no price at the
next bar, its return went NaN, `nan_to_num` turned that into 0.0, and the book
held the position at exactly zero return for the rest of the backtest — free of
cost, free of risk, diluting the denominator of every subsequent figure. Silicon
Valley Bank would have sat in the portfolio forever, marked flat.

### The data source lies by omission

Alpaca's asset endpoint lists ~19,000 inactive equities, and **does not contain
SIVB, FRC, TWTR, VMW or ATVI at all** — those records are purged, while the bars
API still serves them. Precisely the large-cap failures a liquidity-ranked
universe would have held.

The first run of this study, built only from the asset endpoint, returned a bias
of **+0.000** — because only 1 of 200 selected names had delisted. That was not
a finding, it was an absence of data, and it is not reported as one.
`data/known_delistings.txt` supplies 107 verified tickers by name, 66 of which
the asset endpoint does not know exist.

---

## 2. Attrition: how much there was to correct

Of the 1,000 most liquid names on each date, how many stopped trading within a
year:

| as of | members | later delisted | rate |
|---|---|---|---|
| 2021-12-31 | 1000 | 19 | 1.9% |
| 2022-12-31 | 1000 | 12 | 1.2% |
| 2023-12-31 | 1000 | 7 | 0.7% |
| 2024-12-31 | 1000 | 7 | 0.7% |

Among the **top 100** it is essentially zero. Mega-caps get acquired, not wound
up, and an acquisition at a premium is not the event survivorship bias is about.
That is why this study varies universe breadth: a single cut at 100 names would
have shown nothing and concluded, wrongly, that the effect does not exist.

---

## 3. The controlled comparison

Same strategy, same dates, same costs, same buffer. One arm holds the names that
later died; the other deletes them from history, reconstructing exactly what the
old caches were. Selected as of 2022-08-08.

Bias = survivor-only Sharpe − point-in-time Sharpe. **Positive means deleting
the failures flattered the result.**

| family | top 100 (2% dead) | top 500 (4% dead) | top 1000 (3.5% dead) |
|---|---|---|---|
| mom12_1 | −0.011 | −0.029 | −0.063 |
| residual_mom | −0.000 | −0.006 | −0.056 |
| low_vol | +0.054 | −0.026 | −0.034 |
| illiquidity | +0.045 | +0.020 | +0.072 |
| lt_reversal | +0.138 | +0.064 | +0.027 |
| **mean** | **+0.045** | **+0.005** | **−0.011** |

**Mean across all 15 pairs: +0.013 Sharpe.** The range is −0.063 to +0.138, the
sign is inconsistent across both families and breadths, and there is no
monotonic relationship with attrition. This is noise around zero, not a small
systematic effect.

---

## 4. Why it is smaller than everyone assumes

The intuition that survivorship bias is large comes from **long-only** equity
research, where it genuinely is: a long-only book of today's survivors has had
the zeros deleted from its return distribution, and nothing offsets that.

Every book in this project is **dollar-neutral long/short**. A delisted name is
about as likely to have been a short as a long, so removing it from history
removes a roughly symmetric contribution. The bias does not vanish, but it stops
being systematic, and at 2–4% attrition over four years what remains is smaller
than the estimation error on the Sharpe itself.

Two conditions that would change this:

*Long-only or long-biased books.* The asymmetry returns immediately.

*A period with real failure clustering.* 2020–2026 had a SPAC unwind and two
bank failures. 2008–2009 would look different, and this sample cannot speak to it.

---

## 5. What this does and does not license

**It does not license going back to survivorship-biased universes.** The
correction is cheap now that it is built, the machinery is the same machinery a
long-only or a stressed-period study would need, and "the bias was small in the
one regime I measured" is not a reason to stop controlling for it.

**It does retire a specific excuse.** Six campaigns produced nothing, and every
one of them noted survivorship as a reason the true numbers were probably worse.
That is now quantified at roughly a hundredth of a Sharpe. Survivorship is not
why nothing worked. The signals were not there.

---

## Caveats that are not footnotes

**The exit price is the last close.** A name acquired at a premium and one that
fell to zero both simply stop. For an acquisition that is roughly right; for a
bankruptcy that halts before the equity is wiped out it is optimistic, so the
measured bias is a **lower** bound. It is a small correction to an already small
number, but it points the same way.

**The supplement is not exhaustive.** 107 verified tickers, assembled from
knowledge of large US delistings, skewed toward names big enough to be
remembered. Whatever is still missing was smaller and less liquid than what was
recovered.

**One selection date, one window.** The universe is chosen once at 2022-08-08
rather than re-selected quarterly — deliberately, so membership changes cannot
differ between the two arms for reasons other than survivorship. A rolling
universe is more realistic and would confound this particular measurement.

**IEX, not SIP.** Volume is a fraction of the consolidated tape, so the
liquidity ranking that selects the universe is approximate.
