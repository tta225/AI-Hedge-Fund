"""Credential loading.

Credentials are read from two places, in this order:

1. the process environment;
2. a ``.env`` file in the project root.

The ``.env`` path exists because it is the least ceremonious option that is
still safe. Platform-level environment variables are injected when a container
starts, so a secret saved mid-session does not appear until a new session
begins — which looks exactly like a rejected key and wastes a lot of time. A
``.env`` file works the moment it is saved.

Safety
------
``.env`` is in ``.gitignore`` and :func:`assert_env_not_tracked` verifies that
at load time. A credential file that git can see is a credential that will
eventually be pushed.

Nothing here ever logs a secret. :func:`describe_credentials` reports only
whether a value is present and its length.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

#: Filename searched for, from the working directory upward.
ENV_FILENAME = ".env"
#: How far up the tree to look before giving up.
MAX_PARENTS = 4


#: The variable names read for each credential, in priority order. Several
#: spellings are accepted per credential because vendors are inconsistent about
#: their own naming — Alpaca's docs use both ``APCA_API_KEY_ID`` and
#: ``ALPACA_API_KEY``, and users reasonably copy whichever one they saw.
CREDENTIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "Alpaca key id": ("APCA_API_KEY_ID", "ALPACA_API_KEY"),
    "Alpaca secret": ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"),
    "Hugging Face": ("HF_TOKEN", "HUGGINGFACE_TOKEN"),
    "Anthropic": ("ANTHROPIC_API_KEY",),
}

#: Vendors whose credentials this project knows how to use.
_PROVIDER_TOKENS = (
    "ALPACA", "APCA", "HUGGINGFACE", "HUGGING_FACE",
    "ANTHROPIC", "POLYGON", "DATABENTO", "FRED",
)
#: Substrings that distinguish a secret from ordinary configuration, so that
#: e.g. ``ANTHROPIC_BASE_URL`` is not reported as a misnamed credential.
_SECRET_TOKENS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


class CredentialSecurityError(RuntimeError):
    """Raised when a credential file is in a state that could leak it."""


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate ``.env``, searching upward from ``start``."""
    current = (start or Path.cwd()).resolve()
    for _ in range(MAX_PARENTS + 1):
        candidate = current / ENV_FILENAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def assert_env_not_tracked(path: Path) -> None:
    """Refuse to load a ``.env`` that git is tracking.

    A tracked credential file is one ``git add -A`` away from being published,
    and unlike most mistakes this one cannot be undone by a later commit — the
    secret is in the history and must be rotated. Failing loudly here is worth
    the inconvenience.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
            cwd=path.parent,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return  # not a git repo, or git unavailable — nothing to check
    if result.returncode == 0:
        raise CredentialSecurityError(
            f"{path} is tracked by git. Credentials in a tracked file will be "
            f"committed and pushed. Run:\n"
            f"    git rm --cached {path.name}\n"
            f"and confirm '{ENV_FILENAME}' is in .gitignore before continuing. "
            f"If it was already pushed, rotate the keys — removing the file "
            f"does not remove it from history."
        )


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=value`` file. Tolerates comments, blanks, and quotes."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        # Trailing comments are only stripped when clearly separated, so a '#'
        # inside a secret is preserved.
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values


@lru_cache(maxsize=1)
def load_env_file(start: str | None = None) -> dict[str, str]:
    """Load ``.env`` once per process. Existing environment wins.

    The process environment takes precedence so a platform-level secret is
    never silently shadowed by a stale file.
    """
    path = find_env_file(Path(start) if start else None)
    if path is None:
        return {}
    assert_env_not_tracked(path)

    values = parse_env_file(path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def get_credential(*names: str, required: bool = False) -> str | None:
    """First non-empty value among ``names``, from env or ``.env``.

    Several names are accepted per credential because vendors are inconsistent
    about their own variable names — Alpaca documents both ``APCA_API_KEY_ID``
    and ``ALPACA_API_KEY``, and users reasonably copy whichever they saw.
    """
    load_env_file()
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if required:
        raise CredentialSecurityError(
            f"none of {list(names)} is set. Add it to .env or the environment; "
            "see docs/DATA_SETUP.md."
        )
    return None


def describe_credentials() -> str:
    """Report which credentials are visible, without revealing any value."""
    load_env_file()
    path = find_env_file()
    lines = [
        f".env file : {path if path else 'not found'}",
    ]

    for label, names in CREDENTIAL_ALIASES.items():
        value = get_credential(*names)
        if value:
            # Length only — never the value, not even a prefix.
            lines.append(f"  ✓ {label:<14} present ({len(value)} chars)")
        else:
            lines.append(f"  ✗ {label:<14} not set — tried {', '.join(names)}")
    return "\n".join(lines)


def _canonical_name_for(upper: str) -> str:
    """The name this project would read, for a variable that looks like ``upper``."""
    wants_secret = "SECRET" in upper
    if "ALPACA" in upper or "APCA" in upper:
        return "APCA_API_SECRET_KEY" if wants_secret else "APCA_API_KEY_ID"
    if "HUGGING" in upper or upper.startswith("HF_"):
        return "HF_TOKEN"
    if "ANTHROPIC" in upper:
        return "ANTHROPIC_API_KEY"
    for vendor in ("POLYGON", "DATABENTO", "FRED"):
        if vendor in upper:
            return f"{vendor}_API_KEY"
    return ""


def misnamed_credentials() -> list[tuple[str, str]]:
    """Set variables that look like a supported credential under a name nothing reads.

    This is the failure that is hardest to see from the outside. A key saved as
    ``ALPACA_KEY`` is present, correct, and completely invisible — the report
    says "not set", which reads as *the save did not work* rather than *the save
    worked and the name is wrong*.

    Returns ``(actual_name, name_to_use)`` pairs. **Names only** — the values are
    never read, so this cannot leak a secret even into a log.
    """
    load_env_file()
    known = {name for names in CREDENTIAL_ALIASES.values() for name in names}
    found: list[tuple[str, str]] = []
    for name in os.environ:
        upper = name.upper()
        if upper in known or not os.environ[name].strip():
            continue
        looks_like_a_vendor = any(
            token in upper for token in _PROVIDER_TOKENS
        ) or upper.startswith("HF_")
        if not looks_like_a_vendor:
            continue
        if not any(token in upper for token in _SECRET_TOKENS):
            continue  # configuration, not a credential
        canonical = _canonical_name_for(upper)
        if canonical and canonical.upper() != upper:
            found.append((name, canonical))
    return sorted(found)


def reset_cache() -> None:
    """Clear the memoised ``.env`` load. Used by tests."""
    load_env_file.cache_clear()
