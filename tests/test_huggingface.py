"""Hugging Face credential and dataset probes. No network — HTTP is stubbed.

The test that earns its keep here is
`test_a_404_is_not_reported_as_private`. The Hub answers an *anonymous* request
with 401 both for a dataset that is private and for one that does not exist, so
"it returned 401, therefore it is private" is an unsound inference — and it is
the inference `docs/DATA_SETUP.md` originally recorded about a dataset that
turned out not to exist. Only an authenticated request separates the two, and
the failure message must not claim more than the response supports.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from axiom.data.huggingface import HuggingFaceProvider


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _stub(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    monkeypatch.setattr("urllib.request.urlopen", handler)


def _requires_datasets() -> None:
    if not HuggingFaceProvider("a/b").is_available():
        pytest.skip("datasets extra not installed")


class TestTokenCheck:
    def test_names_the_account_the_token_belongs_to(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token for the wrong account authenticates and sees nothing."""
        _requires_datasets()
        _stub(monkeypatch, lambda *a, **k: _Response({"name": "someone-else"}))
        status = HuggingFaceProvider("a/b", token="hf_x").check_token()
        assert status.startswith("✓")
        assert "someone-else" in status

    def test_a_rejected_token_reports_the_status_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _requires_datasets()

        def refuse(*_a: object, **_k: object) -> None:
            raise urllib.error.HTTPError("url", 401, "unauthorized", {}, None)  # type: ignore[arg-type]

        _stub(monkeypatch, refuse)
        assert "401" in HuggingFaceProvider("a/b", token="hf_x").check_token()

    def test_a_missing_token_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Public datasets need no token, so absence is a caveat, not a failure."""
        _requires_datasets()
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        status = HuggingFaceProvider("a/b").check_token()
        assert not status.startswith("✗")


class TestDatasetCheck:
    def test_a_resolving_dataset_reports_its_visibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _requires_datasets()
        pages = [_Response({"name": "me"}), _Response({"id": "me/bars", "private": True})]
        _stub(monkeypatch, lambda *a, **k: pages.pop(0))
        status = HuggingFaceProvider("me/bars", token="hf_x").check_dataset()
        assert status.startswith("✓")
        assert "private" in status

    def test_a_404_is_not_reported_as_private(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: an authenticated miss means gone, not hidden."""
        _requires_datasets()
        calls: list[object] = [_Response({"name": "me"})]

        def handler(*_a: object, **_k: object) -> object:
            if calls:
                return calls.pop(0)
            raise urllib.error.HTTPError("url", 404, "not found", {}, None)  # type: ignore[arg-type]

        _stub(monkeypatch, handler)
        status = HuggingFaceProvider("me/missing", token="hf_x").check_dataset()

        assert status.startswith("✗")
        assert "me/missing" in status
        assert "private" not in status

    def test_a_valid_token_is_not_mistaken_for_a_working_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`check_token` passing must not make `check_dataset` pass."""
        _requires_datasets()
        calls: list[object] = [_Response({"name": "me"})]

        def handler(*_a: object, **_k: object) -> object:
            if calls:
                return calls.pop(0)
            raise urllib.error.HTTPError("url", 404, "not found", {}, None)  # type: ignore[arg-type]

        _stub(monkeypatch, handler)
        provider = HuggingFaceProvider("me/missing", token="hf_x")

        assert provider.check_token().startswith("✓")
        assert provider.check_dataset().startswith("✗")
