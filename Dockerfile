# A desk that runs the same way on a laptop and on a server.
#
# Two stages. The builder installs into a virtualenv with the compilers numpy
# and pandas need; the runtime copies the finished virtualenv and carries none
# of them. The result is smaller, and — the part that matters — a compromised
# process has no toolchain to build with.
#
# Pinned to a digest-free but explicit minor version rather than `latest`: an
# image that silently changes Python versions between a test run and a deploy
# is a reproducibility hole wearing a convenience.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build

# Dependency metadata first, so a source-only change does not re-resolve and
# re-download every wheel. The README is copied because the project metadata
# references it and hatchling refuses to build without it.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .


FROM python:3.12-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runs as a non-root user with no login shell. The desk needs to write exactly
# one directory — the store — and nothing else on the filesystem.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 axiom

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN mkdir -p /app/data && chown -R axiom:axiom /app

USER axiom

# The store and any cached bars belong on a mounted volume. Left in the image
# they would be lost on every redeploy, which for the store means losing the
# record of what the desk owns.
VOLUME ["/app/data"]

# Health is readable from the store alone, by design — see axiom.ops.health.
# That means the container can be probed without asking the desk process
# anything, which is the only useful behaviour when the process is wedged.
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD ["axiom", "desk-health", "--db", "/app/data/desk.db"]

ENTRYPOINT ["axiom"]
CMD ["--help"]
