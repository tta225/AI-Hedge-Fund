"""Hugging Face **Storage Bucket** adapter.

Buckets are not datasets
------------------------
A Storage Bucket is a distinct repo type on the Hub: S3-like, non-versioned,
mutable object storage. `datasets.load_dataset()` **cannot read one**, and
`/api/datasets/<id>` returns 404 for one — which is exactly how
`docs/DATA_SETUP.md` twice reached the wrong conclusion that a real bucket did
not exist. A 404 from the datasets API is evidence about the datasets API.

So this adapter talks to the bucket API directly through `huggingface_hub`.

Why the filename index matters
------------------------------
`Tta225/OHLCV-1m-bucket` holds 411 monthly Parquet files totalling **87.7 GB**.
Downloading all of it to serve a 60-day backtest would be absurd, and doing it
once per backtest would be worse.

So the adapter reads the **object listing** (cheap — it is metadata) and parses
the calendar month out of each filename. A request for March-to-April 2024 then
downloads two files, not four hundred. `filename_pattern` is a regex with named
groups `year` and `month`; files that do not match are ignored rather than
guessed at.

Files are cached under `cache_dir` and re-used, so the second run of a backtest
costs nothing.

What is verified and what is not
--------------------------------
The listing, month-selection and caching paths are exercised against a local
fixture in `tests/test_hf_bucket.py`.

**The download path is not verified end-to-end in the environment this was
written in.** Bucket bytes are served by Xet (`cas-server.xethub.hf.co`,
`transfer.xethub.hf.co`) and the S3 gateway (`s3.hf.co`), all of which are
blocked by that environment's egress policy — only `huggingface.co` itself is
reachable, which serves metadata. The column schema of the real files is
likewise **unconfirmed**, because the files could not be opened. The adapter is
therefore deliberately schema-flexible and reports what it found rather than
assuming; see `inspect`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from axiom.core.credentials import get_credential
from axiom.core.provenance import DataKind, Provenance
from axiom.data.base import BarRequest, BaseProvider, ProviderError, ProviderUnavailableError
from axiom.data.huggingface import normalise_ohlcv_frame

__all__ = ["BucketObject", "HFBucketProvider"]

#: Matches `ohlcv_1992-01.parquet`, `data/ohlcv_2026-03.parquet`, and similar.
DEFAULT_FILENAME_PATTERN = r"(?P<year>\d{4})-(?P<month>\d{2})\.parquet$"


@dataclass(frozen=True, slots=True)
class BucketObject:
    """One object in the bucket, with the month parsed from its name."""

    path: str
    size: int
    #: First instant the file's name claims to cover, or None when unparsed.
    period_start: pd.Timestamp | None = None

    @property
    def period_end(self) -> pd.Timestamp | None:
        if self.period_start is None:
            return None
        return self.period_start + pd.offsets.MonthBegin(1)

    def covers(self, start: datetime, end: datetime) -> bool:
        """True when this file's month overlaps `[start, end]`.

        A file whose month cannot be parsed returns True — an unparseable name
        is a reason to include the file and let the data speak, not a reason to
        silently drop history the user asked for.
        """
        if self.period_start is None or self.period_end is None:
            return True
        return bool(
            self.period_start < pd.Timestamp(end)
            and self.period_end > pd.Timestamp(start)
        )


class HFBucketProvider(BaseProvider):
    """Reads OHLCV bars from a Hugging Face Storage Bucket of Parquet files.

    Args:
        bucket: bucket id, e.g. ``"Tta225/OHLCV-1m-bucket"``.
        prefix: only objects under this key prefix are considered.
        symbol_column: column to filter on when one file holds many instruments.
        filename_pattern: regex with named groups ``year`` and ``month``, used
            to skip files outside the requested window without opening them.
        kind: provenance kind. Leave ``REAL`` only if these are genuinely
            observed market bars — mislabel it once and every backtest
            downstream reports fiction as evidence.
        max_files: refuses rather than quietly starting an enormous download.
    """

    name = "hf-bucket"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        token: str | None = None,
        symbol_column: str | None = None,
        timestamp_column: str | None = None,
        column_map: dict[str, str] | None = None,
        filename_pattern: str = DEFAULT_FILENAME_PATTERN,
        kind: DataKind = DataKind.REAL,
        cache_dir: Path | str = "data/cache/hf-bucket",
        max_files: int = 60,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.token = token or get_credential("HF_TOKEN", "HUGGINGFACE_TOKEN")
        self.symbol_column = symbol_column
        self.timestamp_column = timestamp_column
        self.column_map = column_map or {}
        self.filename_pattern = re.compile(filename_pattern)
        self.kind = kind
        self.cache_dir = Path(cache_dir)
        self.max_files = max_files
        self._listing: list[BucketObject] | None = None

    def is_available(self) -> bool:
        try:
            from huggingface_hub import HfApi  # noqa: F401
        except ImportError:
            return False
        return True

    def _api(self) -> object:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - guarded by is_available
            raise ProviderUnavailableError(
                "huggingface_hub is not installed (pip install 'axiom[huggingface]')"
            ) from exc
        return HfApi(token=self.token)

    # --- listing ------------------------------------------------------------

    def _parse_period(self, path: str) -> pd.Timestamp | None:
        match = self.filename_pattern.search(path)
        if match is None:
            return None
        try:
            return pd.Timestamp(
                year=int(match.group("year")), month=int(match.group("month")),
                day=1, tz=UTC,
            )
        except (ValueError, IndexError):
            return None

    def listing(self, *, refresh: bool = False) -> list[BucketObject]:
        """Every Parquet object under `prefix`, with its month parsed.

        Metadata only — no bytes are transferred. Cached for the provider's
        lifetime because a listing of several hundred objects is not free and
        does not change during a backtest.
        """
        if self._listing is not None and not refresh:
            return self._listing

        api = self._api()
        try:
            items = list(api.list_bucket_tree(self.bucket, recursive=True))  # type: ignore[attr-defined]
        except Exception as exc:
            raise ProviderError(
                f"{self.name}: could not list bucket {self.bucket!r}: {exc}"
            ) from exc

        objects: list[BucketObject] = []
        for item in items:
            path = str(getattr(item, "path", item))
            if not path.endswith(".parquet"):
                continue
            if self.prefix and not path.startswith(self.prefix):
                continue
            objects.append(
                BucketObject(
                    path=path,
                    size=int(getattr(item, "size", 0) or 0),
                    period_start=self._parse_period(path),
                )
            )

        objects.sort(key=lambda o: (o.period_start is None, o.period_start, o.path))
        self._listing = objects
        return objects

    def files_for(self, start: datetime, end: datetime) -> list[BucketObject]:
        """The subset of objects whose months overlap the requested window."""
        return [o for o in self.listing() if o.covers(start, end)]

    # --- fetching -----------------------------------------------------------

    def _download(self, objects: list[BucketObject]) -> list[Path]:
        """Download any objects not already cached, and return local paths."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        wanted: list[tuple[str, str]] = []
        local_paths: list[Path] = []

        for obj in objects:
            local = self.cache_dir / obj.path.replace("/", "__")
            local_paths.append(local)
            # Size is the cheapest available integrity check. It will not catch
            # a file rewritten to the same length — buckets are mutable, so
            # that is possible — but it catches truncation, which is the failure
            # an interrupted download actually produces.
            if local.exists() and (obj.size == 0 or local.stat().st_size == obj.size):
                continue
            wanted.append((obj.path, str(local)))

        if wanted:
            api = self._api()
            try:
                api.download_bucket_files(  # type: ignore[attr-defined]
                    self.bucket, wanted, raise_on_missing_files=True
                )
            except Exception as exc:
                raise ProviderError(
                    f"{self.name}: could not download from {self.bucket!r}: {exc}\n"
                    f"  A Xet download is two hops: cas-server.xethub.hf.co "
                    f"answers with a signed URL pointing at a regional CDN "
                    f"(us.aws.cdn.hf.co, eu.aws.cdn.hf.co). BOTH must be "
                    f"reachable — allowlisting only the first produces a second "
                    f"failure naming the second, which looks like the fix did "
                    f"not work. '*.hf.co' covers every hop. Listing uses "
                    f"huggingface.co only, so it keeps succeeding throughout. "
                    f"See docs/DATA_SETUP.md."
                ) from exc

        return local_paths

    def _fetch_raw(self, request: BarRequest) -> pd.DataFrame:
        objects = self.files_for(request.start, request.end)
        if not objects:
            available = self.listing()
            span = ""
            dated = [o for o in available if o.period_start is not None]
            if dated:
                span = (
                    f" Bucket covers {dated[0].period_start:%Y-%m} to "
                    f"{dated[-1].period_start:%Y-%m}."
                )
            raise ProviderError(
                f"{self.name}: no objects in {self.bucket!r} cover "
                f"{request.start:%Y-%m-%d} to {request.end:%Y-%m-%d}.{span}"
            )

        if len(objects) > self.max_files:
            total_gb = sum(o.size for o in objects) / 1e9
            raise ProviderError(
                f"{self.name}: that window needs {len(objects)} files "
                f"(~{total_gb:.1f} GB), above the {self.max_files}-file cap. "
                f"Narrow the window, or raise max_files if you mean it."
            )

        frames: list[pd.DataFrame] = []
        for path in self._download(objects):
            try:
                frames.append(pd.read_parquet(path))
            except Exception as exc:
                raise ProviderError(
                    f"{self.name}: could not read {path.name}: {exc}"
                ) from exc

        frame = normalise_ohlcv_frame(
            pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0],
            column_map=self.column_map,
            timestamp_column=self.timestamp_column,
            provider_name=self.name,
        )

        if self.symbol_column and self.symbol_column.lower() in frame.columns:
            column = frame[self.symbol_column.lower()].astype(str).str.upper()
            frame = frame[column == request.instrument.symbol.upper()]
            if frame.empty:
                available = sorted(set(column.unique()))[:20]
                raise ProviderError(
                    f"{self.name}: no rows for {request.instrument.symbol}; "
                    f"these files contain {available}"
                )

        window = frame.loc[str(request.start) : str(request.end)]
        if window.empty:
            raise ProviderError(
                f"{self.name}: files cover {frame.index[0]} to {frame.index[-1]}, "
                f"which does not overlap the requested "
                f"{request.start} to {request.end}"
            )
        return window

    def _provenance(self, request: BarRequest) -> Provenance:
        return Provenance(
            source=self.name,
            kind=self.kind,
            detail=f"bucket={self.bucket} prefix={self.prefix or '/'}",
        )

    # --- diagnostics --------------------------------------------------------

    def check_bucket(self) -> str:
        """Report whether the bucket resolves, and what is in it.

        Deliberately reports the *bucket* API, because the datasets API returns
        404 for a bucket and reading that as "does not exist" is the exact
        mistake this adapter was written to correct.
        """
        import json
        import urllib.error
        import urllib.request

        if not self.is_available():
            return "✗ huggingface_hub not installed — pip install -e '.[huggingface]'"

        headers = {"User-Agent": "axiom"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            probe = urllib.request.Request(
                f"https://huggingface.co/api/buckets/{self.bucket}", headers=headers
            )
            with urllib.request.urlopen(probe, timeout=30) as response:
                info = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 404):
                return (
                    f"✗ bucket '{self.bucket}' did not resolve (HTTP {exc.code}). "
                    f"If this id is a *dataset*, use --dataset instead."
                )
            return f"✗ HF returned HTTP {exc.code} for bucket '{self.bucket}'"
        except urllib.error.URLError as exc:
            return f"✗ Could not reach Hugging Face: {exc.reason}"

        visibility = "private" if info.get("private") else "public"
        size_gb = float(info.get("size", 0)) / 1e9
        return (
            f"✓ bucket '{self.bucket}' resolves ({visibility}): "
            f"{info.get('totalFiles', '?')} files, {size_gb:.1f} GB"
        )

    def inspect(self) -> str:
        """Summarise the listing without downloading anything.

        Reports what the *names* say. It does not open a file, so it makes no
        claim about the column schema — that requires bytes.
        """
        objects = self.listing()
        dated = [o for o in objects if o.period_start is not None]
        total_gb = sum(o.size for o in objects) / 1e9
        lines = [
            f"bucket  : {self.bucket}",
            f"objects : {len(objects)} parquet files ({total_gb:.1f} GB)",
        ]
        if dated:
            lines.append(
                f"span    : {dated[0].period_start:%Y-%m} → {dated[-1].period_start:%Y-%m} "
                f"(from filenames, not from the data)"
            )
        undated = len(objects) - len(dated)
        if undated:
            lines.append(
                f"unparsed: {undated} filenames did not match the month pattern; "
                f"they are always included in a fetch"
            )
        lines.append(
            "schema  : not inspected — that needs a download. "
            "Fetch a narrow window first to confirm the columns."
        )
        return "\n".join(lines)
