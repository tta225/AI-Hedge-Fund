"""Hugging Face Storage Bucket adapter. Offline — the Hub is stubbed.

The month-selection tests are the ones that earn their keep. The bucket this was
written for holds 87.7 GB across 411 monthly files; a 60-day backtest that
downloads all of them is not slow, it is unusable. Selecting files by the
calendar month in their *name* — before any bytes move — is what makes the
adapter practical, and an off-by-one there silently drops a month of history
from a backtest that still completes and still reports metrics.

`TestUnparseableFilenames` pins the deliberate choice to *include* files whose
names do not match the pattern. Excluding them would silently discard data the
user asked for, which is the worse failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from axiom.core.provenance import DataKind
from axiom.core.timeframe import Timeframe
from axiom.core.types import AssetClass, Instrument
from axiom.data.base import BarRequest, ProviderError
from axiom.data.hf_bucket import BucketObject, HFBucketProvider

SPY = Instrument("SPY", AssetClass.EQUITY, "ARCA")


def _request(start: str, end: str) -> BarRequest:
    return BarRequest(
        instrument=SPY,
        timeframe=Timeframe.parse("1m"),
        start=datetime.fromisoformat(start).replace(tzinfo=UTC),
        end=datetime.fromisoformat(end).replace(tzinfo=UTC),
    )


class _Item:
    """Mimics a huggingface_hub BucketFile well enough for the listing path."""

    def __init__(self, path: str, size: int = 1000) -> None:
        self.path = path
        self.size = size


class _StubProvider(HFBucketProvider):
    """Serves a fixed listing and writes fixture Parquet instead of downloading."""

    def __init__(self, paths: list[str], frames: dict[str, pd.DataFrame] | None = None,
                 **kwargs: object) -> None:
        super().__init__("acme/bars", **kwargs)  # type: ignore[arg-type]
        self._paths = paths
        self._frames = frames or {}
        self.downloaded: list[str] = []

    def is_available(self) -> bool:
        return True

    def _size_of(self, path: str) -> int:
        """Real byte length of the fixture, so the cache check behaves as it
        would against a real bucket (which reports true object sizes)."""
        frame = self._frames.get(path)
        if frame is None:
            return 0
        import io

        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        return buffer.getbuffer().nbytes

    def _api(self) -> object:
        provider = self

        class _Api:
            @staticmethod
            def list_bucket_tree(_bucket: str, recursive: bool = True) -> list[_Item]:
                return [
                    _Item(p, provider._size_of(p) or 1000) for p in provider._paths
                ]

            @staticmethod
            def download_bucket_files(
                _bucket: str, files: list[tuple[str, str]], **_kw: object
            ) -> None:
                for remote, local in files:
                    provider.downloaded.append(remote)
                    frame = provider._frames.get(remote)
                    if frame is None:
                        raise FileNotFoundError(remote)
                    Path(local).parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(local)

        return _Api()


def _bars(month: str, rows: int = 40) -> pd.DataFrame:
    index = pd.date_range(f"{month}-01", periods=rows, freq="1min", tz="UTC")
    base = 100.0 + float(month[-2:])
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": 1000.0,
        }
    )


class TestListing:
    def test_parses_the_month_from_each_filename(self) -> None:
        provider = _StubProvider(
            ["data/ohlcv_1992-01.parquet", "data/ohlcv_2026-03.parquet"]
        )
        objects = provider.listing()
        assert [o.period_start.strftime("%Y-%m") for o in objects] == [
            "1992-01", "2026-03"
        ]

    def test_ignores_non_parquet_objects(self) -> None:
        provider = _StubProvider(
            ["README.md", ".gitattributes", "data/ohlcv_2024-01.parquet"]
        )
        assert [o.path for o in provider.listing()] == ["data/ohlcv_2024-01.parquet"]

    def test_prefix_filters_the_listing(self) -> None:
        provider = _StubProvider(
            ["data/a_2024-01.parquet", "other/b_2024-01.parquet"], prefix="data/"
        )
        assert [o.path for o in provider.listing()] == ["data/a_2024-01.parquet"]

    def test_listing_is_cached_until_refreshed(self) -> None:
        provider = _StubProvider(["data/ohlcv_2024-01.parquet"])
        first = provider.listing()
        assert provider.listing() is first
        assert provider.listing(refresh=True) is not first


class TestMonthSelection:
    """Selecting by filename is what makes an 87 GB bucket usable."""

    PATHS: ClassVar[list[str]] = [
        f"data/ohlcv_2024-{m:02d}.parquet" for m in range(1, 13)
    ]

    def test_narrow_window_selects_only_the_months_it_spans(self) -> None:
        provider = _StubProvider(self.PATHS)
        files = provider.files_for(
            datetime(2024, 3, 10, tzinfo=UTC), datetime(2024, 4, 20, tzinfo=UTC)
        )
        assert [f.path for f in files] == [
            "data/ohlcv_2024-03.parquet",
            "data/ohlcv_2024-04.parquet",
        ]

    def test_a_window_inside_one_month_selects_one_file(self) -> None:
        provider = _StubProvider(self.PATHS)
        files = provider.files_for(
            datetime(2024, 7, 5, tzinfo=UTC), datetime(2024, 7, 9, tzinfo=UTC)
        )
        assert [f.path for f in files] == ["data/ohlcv_2024-07.parquet"]

    def test_a_window_ending_exactly_on_a_boundary_excludes_the_next_month(self) -> None:
        """The off-by-one that would silently add or drop a month."""
        provider = _StubProvider(self.PATHS)
        files = provider.files_for(
            datetime(2024, 5, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)
        )
        assert [f.path for f in files] == ["data/ohlcv_2024-05.parquet"]

    def test_a_window_outside_the_bucket_selects_nothing(self) -> None:
        provider = _StubProvider(self.PATHS)
        assert not provider.files_for(
            datetime(2030, 1, 1, tzinfo=UTC), datetime(2030, 2, 1, tzinfo=UTC)
        )

    def test_full_span_selects_everything(self) -> None:
        provider = _StubProvider(self.PATHS)
        files = provider.files_for(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
        )
        assert len(files) == 12


class TestUnparseableFilenames:
    def test_an_unparsed_name_is_included_rather_than_dropped(self) -> None:
        """Including it risks a wasted download; excluding it loses data silently."""
        obj = BucketObject(path="data/everything.parquet", size=10, period_start=None)
        assert obj.covers(datetime(1999, 1, 1, tzinfo=UTC), datetime(1999, 2, 1, tzinfo=UTC))

    def test_inspect_reports_how_many_were_unparsed(self) -> None:
        provider = _StubProvider(
            ["data/ohlcv_2024-01.parquet", "data/everything.parquet"]
        )
        text = provider.inspect()
        assert "1 filenames did not match" in text
        assert "not inspected" in text, "inspect must not claim to know the schema"


class TestFetching:
    PATHS: ClassVar[list[str]] = [
        f"data/ohlcv_2024-{m:02d}.parquet" for m in range(1, 4)
    ]

    def _provider(self, tmp_path: Path, **kwargs: object) -> _StubProvider:
        frames = {p: _bars(f"2024-{p[-10:-8]}") for p in self.PATHS}
        return _StubProvider(
            self.PATHS, frames, cache_dir=tmp_path / "cache", **kwargs
        )

    def test_fetches_and_normalises_bars(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path)
        series = provider.fetch_bars(_request("2024-01-01", "2024-01-01T00:30"))
        assert len(series) == 31
        assert series.provenance.kind is DataKind.REAL
        assert "acme/bars" in series.provenance.detail

    def test_only_the_needed_months_are_downloaded(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path)
        provider.fetch_bars(_request("2024-02-01", "2024-02-01T00:20"))
        assert provider.downloaded == ["data/ohlcv_2024-02.parquet"]

    def test_a_cached_file_is_not_downloaded_twice(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path)
        provider.fetch_bars(_request("2024-01-01", "2024-01-01T00:20"))
        provider.fetch_bars(_request("2024-01-01", "2024-01-01T00:25"))
        assert provider.downloaded == ["data/ohlcv_2024-01.parquet"]

    def test_spanning_months_concatenates_files(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path)
        provider.fetch_bars(_request("2024-01-01", "2024-02-01T00:10"))
        assert len(provider.downloaded) == 2

    def test_a_window_the_bucket_does_not_cover_names_the_span(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path)
        with pytest.raises(ProviderError, match="Bucket covers 2024-01 to 2024-03"):
            provider.fetch_bars(_request("2030-01-01", "2030-02-01"))

    def test_max_files_refuses_an_enormous_download(self, tmp_path: Path) -> None:
        provider = self._provider(tmp_path, max_files=1)
        with pytest.raises(ProviderError, match="above the 1-file cap"):
            provider.fetch_bars(_request("2024-01-01", "2024-03-31"))

    def test_symbol_column_filters_and_reports_what_is_there(self, tmp_path: Path) -> None:
        frame = _bars("2024-01")
        frame["ticker"] = "QQQ"
        provider = _StubProvider(
            ["data/ohlcv_2024-01.parquet"], {"data/ohlcv_2024-01.parquet": frame},
            cache_dir=tmp_path / "cache", symbol_column="ticker",
        )
        with pytest.raises(ProviderError, match=r"no rows for SPY.*QQQ"):
            provider.fetch_bars(_request("2024-01-01", "2024-01-01T00:20"))

    def test_a_download_failure_names_the_hosts_involved(self, tmp_path: Path) -> None:
        """The failure mode seen in practice: listing works, bytes are blocked."""
        provider = _StubProvider(
            ["data/ohlcv_2024-01.parquet"], {}, cache_dir=tmp_path / "cache"
        )
        with pytest.raises(ProviderError, match="xethub"):
            provider.fetch_bars(_request("2024-01-01", "2024-01-31"))


class TestProvenance:
    def test_synthetic_kind_is_honoured(self, tmp_path: Path) -> None:
        """A bucket of generated bars must not be stamped as evidence."""
        paths = ["data/ohlcv_2024-01.parquet"]
        provider = _StubProvider(
            paths, {paths[0]: _bars("2024-01")},
            cache_dir=tmp_path / "cache", kind=DataKind.SYNTHETIC,
        )
        series = provider.fetch_bars(_request("2024-01-01", "2024-01-01T00:20"))
        assert not series.provenance.is_evidential
