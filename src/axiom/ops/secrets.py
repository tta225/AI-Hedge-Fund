"""Where credentials come from, and what may be said about them.

Until now the desk read ``APCA_API_KEY_ID`` out of the process environment and
trusted whatever it found. That works on a laptop and fails an audit for three
separate reasons, each of which this module closes.

**There is no separation between paper and live.** One environment variable
name serves both, so the difference between a simulation and real money is
which ``.env`` file happened to be on disk. That is a configuration accident
standing where a control should be. :class:`CredentialSet` is scoped to an
:class:`Environment`, and resolution for ``live`` deliberately refuses to fall
back to an unscoped variable: promoting to live must be a thing someone did,
not a thing that happened because a variable was already set.

**There is no provenance.** A key read from the environment, a key read from a
file, and a key printed by a vault agent are operationally different objects
with different rotation stories, and the desk could not tell you which one it
was holding. Every :class:`Credential` carries its source.

**Secrets leak through the exception handler.** The moment a credential is an
ordinary string it can end up in a traceback, a log line, or a JSON dump of a
config object. :class:`Credential` refuses to render itself — ``str``, ``repr``
and formatting all produce the fingerprint, never the value, and getting at the
real thing requires calling :meth:`Credential.reveal` at the point of use.

Nothing here talks to a specific vault. A vault integration would drag a
network dependency and its own credentials into the component that must work
before anything else does. Instead :class:`CommandSource` shells out to a
command the operator configures — ``vault read``, ``aws secretsmanager
get-secret-value``, ``op read``, ``pass`` — which is the same integration
without the coupling.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: How many hex characters of the digest identify a credential in logs. Eight
#: is enough to tell two keys apart in an incident and far too few to attack.
FINGERPRINT_CHARS = 8
#: Seconds a credential-fetching command may run before it is presumed hung.
#: A vault that has not answered in fifteen seconds is not going to.
COMMAND_TIMEOUT = 15.0


class Environment(str, Enum):
    """Which world a credential belongs to.

    ``PAPER`` and ``LIVE`` are not two settings of one thing. They are separate
    namespaces, and the whole point is that a mistake in one cannot reach the
    other.
    """

    PAPER = "paper"
    LIVE = "live"

    @property
    def is_live(self) -> bool:
        return self is Environment.LIVE


class SecretsError(RuntimeError):
    """A credential could not be resolved, or was resolved unsafely."""


@dataclass(frozen=True, slots=True)
class Credential:
    """One secret, and everything about it that is safe to say out loud.

    The value is stored under a private field name and every rendering path is
    overridden. That is not paranoia about a hostile caller — it is defence
    against the ordinary case where somebody logs a config object.
    """

    name: str
    _value: str
    source: str
    environment: Environment

    def __post_init__(self) -> None:
        if not self._value:
            raise SecretsError(f"credential {self.name!r} resolved to an empty value")

    @property
    def fingerprint(self) -> str:
        """A stable, non-reversible identifier for this exact value.

        Salted with the credential's name so the same secret used for two
        purposes does not correlate across logs.
        """
        digest = hashlib.sha256(f"{self.name}:{self._value}".encode()).hexdigest()
        return digest[:FINGERPRINT_CHARS]

    def reveal(self) -> str:
        """The actual secret. Call this at the point of use and nowhere else."""
        return self._value

    def matches(self, other: str) -> bool:
        """Constant-time comparison, so a check cannot be timed."""
        import hmac

        return hmac.compare_digest(self._value, other)

    def __str__(self) -> str:
        return f"<{self.name} {self.environment.value}:{self.fingerprint}>"

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return str(self)


class SecretSource(ABC):
    """Somewhere a credential can come from."""

    name: str = "source"

    @abstractmethod
    def fetch(self, key: str) -> str | None:
        """The raw value, or ``None`` if this source does not have it."""


@dataclass(frozen=True, slots=True)
class EnvSource(SecretSource):
    """The process environment.

    Convenient, and the weakest of the three: environment variables are visible
    to anything that can read ``/proc``, are inherited by every subprocess, and
    survive in shell history. Fine for paper, and flagged as such.
    """

    name: str = "env"

    def fetch(self, key: str) -> str | None:
        value = os.environ.get(key)
        return value.strip() if value else None


@dataclass(frozen=True, slots=True)
class FileSource(SecretSource):
    """A directory of single-secret files, as mounted by Kubernetes or systemd.

    One secret per file is the shape every orchestrator already speaks, and it
    has a property the environment does not: the file's permissions are an
    enforceable control, and this checks them.
    """

    directory: Path
    #: Refuse to read a secret that is readable by group or other. A secret the
    #: whole machine can read is not a secret, and silently using it would hide
    #: exactly the misconfiguration worth catching.
    require_owner_only: bool = True
    name: str = "file"

    def fetch(self, key: str) -> str | None:
        path = self.directory / key
        if not path.is_file():
            return None
        if self.require_owner_only:
            mode = path.stat().st_mode & 0o077
            if mode:
                raise SecretsError(
                    f"{path} is readable beyond its owner (mode ...{mode:03o}). "
                    f"Run `chmod 600 {path}` or construct FileSource with "
                    "require_owner_only=False if this is deliberate."
                )
        return path.read_text().strip() or None


@dataclass(frozen=True, slots=True)
class CommandSource(SecretSource):
    """A command that prints the secret on stdout.

    This is the vault integration, without a vault dependency. ``template`` is
    a shell-style command with ``{key}`` substituted, e.g.::

        CommandSource("vault read -field=value secret/axiom/{key}")
        CommandSource("op read op://desk/{key}/credential")

    The command is split with :func:`shlex.split` and run **without** a shell,
    so a key containing a semicolon cannot become a second command.
    """

    template: str
    timeout: float = COMMAND_TIMEOUT
    name: str = "command"

    def fetch(self, key: str) -> str | None:
        if any(c in key for c in " \t\n'\"$`\\"):
            raise SecretsError(
                f"credential key {key!r} contains characters that are not safe to "
                "interpolate into a command"
            )
        argv = shlex.split(self.template.format(key=key))
        if not argv:
            raise SecretsError("command template is empty")
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except FileNotFoundError as exc:
            raise SecretsError(f"credential command {argv[0]!r} not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise SecretsError(
                f"credential command {argv[0]!r} did not answer in {self.timeout}s"
            ) from exc
        if result.returncode != 0:
            # stderr may echo the key but should not contain the value; it is
            # truncated regardless, because "should not" is not a guarantee.
            detail = result.stderr.strip()[:200]
            raise SecretsError(
                f"credential command failed for {key!r} (exit {result.returncode}): {detail}"
            )
        return result.stdout.strip() or None


@dataclass
class CredentialSet:
    """Resolves credentials for one environment, in a fixed source order.

    Sources are tried in order and the first hit wins, so a file-mounted secret
    overrides the environment when the file source is listed first. Resolution
    is cached: a vault command that is invoked on every order is both slow and
    a great way to get rate-limited mid-session.
    """

    environment: Environment
    sources: tuple[SecretSource, ...] = field(default_factory=lambda: (EnvSource(),))
    #: Prefix applied to every key, so paper and live cannot collide.
    #: ``APCA_API_KEY_ID`` becomes ``AXIOM_LIVE_APCA_API_KEY_ID``.
    prefix: str = "AXIOM"
    _cache: dict[str, Credential] = field(default_factory=dict, repr=False)

    def scoped(self, key: str) -> str:
        return f"{self.prefix}_{self.environment.value.upper()}_{key}"

    def get(self, key: str, *, allow_unscoped: bool = True) -> Credential:
        """Resolve ``key``, or raise with the names that were tried.

        The scoped name is always tried first. The unscoped fallback exists so
        an existing paper setup keeps working, and is **refused for live**: a
        live key must be placed deliberately under its live name, because the
        alternative is that leaving a paper variable set arms real money.
        """
        if key in self._cache:
            return self._cache[key]

        names = [self.scoped(key)]
        if allow_unscoped and not self.environment.is_live:
            names.append(key)

        for name in names:
            for source in self.sources:
                value = source.fetch(name)
                if value:
                    credential = Credential(
                        name=key,
                        _value=value,
                        source=f"{source.name}:{name}",
                        environment=self.environment,
                    )
                    self._cache[key] = credential
                    return credential

        tried = ", ".join(names)
        suffix = (
            ""
            if allow_unscoped and not self.environment.is_live
            else (
                f" The unscoped name {key!r} is deliberately not consulted for "
                "the live environment; place the secret under its scoped name."
            )
        )
        raise SecretsError(
            f"no source provided {key!r} for the {self.environment.value} environment "
            f"(tried: {tried} across {[s.name for s in self.sources]})." + suffix
        )

    def get_optional(self, key: str) -> Credential | None:
        try:
            return self.get(key)
        except SecretsError:
            return None

    def audit(self) -> list[dict[str, str]]:
        """What has been resolved, in a form safe to write to a log or report.

        Only credentials already fetched appear — this does not go looking, so
        calling it never triggers a vault round trip.
        """
        return [
            {
                "name": credential.name,
                "environment": credential.environment.value,
                "source": credential.source,
                "fingerprint": credential.fingerprint,
            }
            for credential in sorted(self._cache.values(), key=lambda c: c.name)
        ]

    def clear(self) -> None:
        """Drop the cache, so the next read picks up a rotated secret."""
        self._cache.clear()


def default_credentials(environment: Environment | str = Environment.PAPER) -> CredentialSet:
    """The conventional source order: mounted files, then a vault command, then env.

    Files first because a Kubernetes or systemd mount is the strongest of the
    three and should not be shadowed by a stale shell variable. The environment
    last because it is the fallback that makes a laptop work.

    ``AXIOM_SECRETS_DIR`` and ``AXIOM_SECRETS_COMMAND`` opt into the stronger
    sources; with neither set this behaves exactly as the desk always has.
    """
    env = Environment(environment)
    sources: list[SecretSource] = []

    directory = os.environ.get("AXIOM_SECRETS_DIR")
    if directory:
        sources.append(FileSource(Path(directory)))

    command = os.environ.get("AXIOM_SECRETS_COMMAND")
    if command:
        sources.append(CommandSource(command))

    sources.append(EnvSource())
    return CredentialSet(environment=env, sources=tuple(sources))
