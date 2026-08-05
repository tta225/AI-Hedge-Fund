"""Credential loading and the safety rails around it.

The security tests here matter more than the parsing ones. A credential file
that git can see will eventually be pushed, and unlike most mistakes that one
cannot be fixed by a later commit — the secret is in the history and has to be
rotated.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from axiom.core.credentials import (
    CredentialSecurityError,
    assert_env_not_tracked,
    describe_credentials,
    find_env_file,
    get_credential,
    misnamed_credentials,
    parse_env_file,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():  # type: ignore[no-untyped-def]
    reset_cache()
    yield
    reset_cache()


class TestParsing:
    def test_reads_simple_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("APCA_API_KEY_ID=abc123\nAPCA_API_SECRET_KEY=def456\n")
        assert parse_env_file(path) == {
            "APCA_API_KEY_ID": "abc123",
            "APCA_API_SECRET_KEY": "def456",
        }

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("# a comment\n\nKEY=value\n\n#KEY2=ignored\n")
        assert parse_env_file(path) == {"KEY": "value"}

    def test_strips_quotes_and_export(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text('export KEY="quoted"\nOTHER=\'single\'\n')
        assert parse_env_file(path) == {"KEY": "quoted", "OTHER": "single"}

    def test_preserves_hash_inside_a_secret(self, tmp_path: Path) -> None:
        """Secrets contain '#'. Stripping it would silently corrupt the key."""
        path = tmp_path / ".env"
        path.write_text("KEY=abc#def\n")
        assert parse_env_file(path)["KEY"] == "abc#def"

    def test_strips_a_clearly_separated_trailing_comment(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("KEY=value # this is a note\n")
        assert parse_env_file(path)["KEY"] == "value"

    def test_tolerates_a_malformed_line(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("not_a_pair\nKEY=value\n")
        assert parse_env_file(path) == {"KEY": "value"}


class TestDiscovery:
    def test_finds_env_in_the_current_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("KEY=value\n")
        assert find_env_file(tmp_path) == tmp_path / ".env"

    def test_searches_upward(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("KEY=value\n")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_env_file(nested) == tmp_path / ".env"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_env_file(tmp_path) is None


class TestSecurity:
    def test_refuses_a_git_tracked_env_file(self, tmp_path: Path) -> None:
        """The rail that stops a credential reaching the remote."""
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        path = tmp_path / ".env"
        path.write_text("KEY=value\n")
        subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True)

        with pytest.raises(CredentialSecurityError, match="tracked by git"):
            assert_env_not_tracked(path)

    def test_error_says_to_rotate_after_a_push(self, tmp_path: Path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        path = tmp_path / ".env"
        path.write_text("KEY=value\n")
        subprocess.run(["git", "add", "-f", ".env"], cwd=tmp_path, check=True)

        with pytest.raises(CredentialSecurityError) as excinfo:
            assert_env_not_tracked(path)
        # Deleting the file does not remove it from history.
        assert "rotate" in str(excinfo.value)

    def test_untracked_env_is_accepted(self, tmp_path: Path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / ".gitignore").write_text(".env\n")
        path = tmp_path / ".env"
        path.write_text("KEY=value\n")
        assert_env_not_tracked(path)  # must not raise

    def test_outside_a_repo_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("KEY=value\n")
        assert_env_not_tracked(path)

    def test_project_gitignores_env(self) -> None:
        """The repository itself must never track a credential file."""
        ignored = Path(__file__).resolve().parents[1] / ".gitignore"
        assert ".env" in ignored.read_text().splitlines()

    def test_description_never_reveals_a_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "SUPERSECRETVALUE1234567890"
        monkeypatch.setenv("APCA_API_KEY_ID", secret)
        reset_cache()
        report = describe_credentials()
        assert secret not in report
        assert "present" in report
        # Length is safe to report; any prefix of the value is not.
        assert str(len(secret)) in report


class TestPrecedence:
    def test_environment_wins_over_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale file must never shadow a platform-provided secret."""
        (tmp_path / ".env").write_text("APCA_API_KEY_ID=from_file\n")
        monkeypatch.setenv("APCA_API_KEY_ID", "from_environment")
        monkeypatch.chdir(tmp_path)
        reset_cache()
        assert get_credential("APCA_API_KEY_ID") == "from_environment"

    def test_falls_back_through_alias_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vendors document several names; users copy whichever they saw."""
        monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
        monkeypatch.setenv("ALPACA_API_KEY", "aliased")
        reset_cache()
        assert get_credential("APCA_API_KEY_ID", "ALPACA_API_KEY") == "aliased"

    def test_missing_credential_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        reset_cache()
        assert get_credential("NOT_SET_ANYWHERE") is None

    def test_required_credential_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        reset_cache()
        with pytest.raises(CredentialSecurityError, match="DATA_SETUP"):
            get_credential("NOT_SET_ANYWHERE", required=True)

    def test_blank_value_counts_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLANK_KEY", "   ")
        reset_cache()
        assert get_credential("BLANK_KEY") is None


class TestMisnamedCredentials:
    """A key under the wrong name is indistinguishable from a key never saved.

    Both report "not set", which sends people back to re-copy a key that was
    always correct. These tests pin the distinction.
    """

    def test_flags_a_misnamed_alpaca_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_KEY_ID", "PK000000000000000000")
        reset_cache()
        assert ("ALPACA_KEY_ID", "APCA_API_KEY_ID") in misnamed_credentials()

    def test_routes_secrets_to_the_secret_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPACA_API_SECRET", "x" * 40)
        reset_cache()
        assert ("ALPACA_API_SECRET", "APCA_API_SECRET_KEY") in misnamed_credentials()

    def test_accepted_names_are_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APCA_API_KEY_ID", "PK000000000000000000")
        monkeypatch.setenv("ALPACA_API_KEY", "PK000000000000000000")
        reset_cache()
        flagged = {name for name, _ in misnamed_credentials()}
        assert "APCA_API_KEY_ID" not in flagged
        assert "ALPACA_API_KEY" not in flagged

    def test_configuration_is_not_mistaken_for_a_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`ANTHROPIC_BASE_URL` is set in this very environment. It is not a key."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
        reset_cache()
        assert "ANTHROPIC_BASE_URL" not in {n for n, _ in misnamed_credentials()}

    def test_unrelated_variables_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_APP_SECRET", "value")
        reset_cache()
        assert "MY_APP_SECRET" not in {n for n, _ in misnamed_credentials()}

    def test_blank_value_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`.env.example` ships empty placeholders; those are not misnamed."""
        monkeypatch.setenv("ALPACA_KEY_ID", "")
        reset_cache()
        assert "ALPACA_KEY_ID" not in {n for n, _ in misnamed_credentials()}

    def test_never_returns_a_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret = "SUPERSECRETVALUE1234567890"
        monkeypatch.setenv("ALPACA_KEY_ID", secret)
        reset_cache()
        assert secret not in str(misnamed_credentials())


class TestProviderIntegration:
    def test_alpaca_picks_up_env_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.data import AlpacaProvider

        monkeypatch.setenv("APCA_API_KEY_ID", "key")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
        reset_cache()
        assert AlpacaProvider().is_available()

    def test_alpaca_unavailable_without_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.data import AlpacaProvider

        for name in (
            "APCA_API_KEY_ID", "ALPACA_API_KEY",
            "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.chdir(os.path.expanduser("~"))
        reset_cache()
        assert not AlpacaProvider().is_available()

    def test_explicit_keys_override_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.data import AlpacaProvider

        monkeypatch.setenv("APCA_API_KEY_ID", "from_env")
        reset_cache()
        provider = AlpacaProvider(api_key="explicit", secret_key="explicit")
        assert provider.api_key == "explicit"
