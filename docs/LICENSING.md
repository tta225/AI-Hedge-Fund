# Licensing

## Why this document exists

AXIOM is intended to be **proprietary**. One of the reference materials for this project
recommends OpenBB as a free Bloomberg Terminal alternative. OpenBB is excellent — and it is
licensed **AGPL-3.0**, which is the single most consequential license for a proprietary
product to link against.

This is not a reason to avoid OpenBB. It is a reason to be deliberate about *where* it sits.

## What AGPL-3.0 requires

The GNU Affero GPL is a strong copyleft license. Its distinguishing feature is
**§13, the network clause**:

> If you modify the Program, your modified version must prominently offer all users
> interacting with it remotely through a computer network … an opportunity to receive the
> Corresponding Source of your version.

The practical consequences for a trading platform:

1. **Distribution triggers copyleft.** Ship software that links AGPL code, and the combined
   work must be offered under AGPL — including your source.
2. **Network use also triggers it.** This is the part that surprises people. A hosted
   dashboard, an internal web terminal offered to clients, or a SaaS API that reaches AGPL
   code over a network can trigger the source-offering obligation *without any distribution
   at all*.
3. **"Linking" is broad.** Importing an AGPL Python package into your process and calling it
   is generally understood to create a derivative work.

**Purely internal use is fine.** Running OpenBB on your own machine for your own research,
distributing nothing and serving nothing over a network, does not trigger the obligations.
That is a genuinely useful and entirely legitimate mode.

> This is engineering guidance for structuring a codebase, not legal advice. Before
> commercialising, distributing, or network-serving anything that touches AGPL code, get an
> opinion from a lawyer who has read your actual architecture.

## How AXIOM is structured

OpenBB is **quarantined**:

1. **Not a core dependency.** It lives in an optional extra (`pip install 'axiom[openbb]'`).
   A default install never fetches it.
2. **Isolated behind our own interface.** `OpenBBProvider` implements the same
   `MarketDataProvider` protocol as every other adapter. Nothing in AXIOM depends on OpenBB
   types, and nothing imports it unless that class is explicitly constructed.
3. **Import is lazy.** `import openbb` happens inside `_fetch_raw`, never at module scope.
   Importing `axiom.data` does not load AGPL code.
4. **Construction is an explicit, auditable act.** The constructor refuses to run without
   `acknowledge_agpl=True`:

```python
OpenBBProvider()
# ProviderError: OpenBB is licensed AGPL-3.0. Linking it into a distributed or
# network-served product places copyleft obligations on the surrounding code.
# Pass acknowledge_agpl=True to proceed for internal research use, and read
# docs/LICENSING.md first.
```

The result: a proprietary build of AXIOM simply never touches AGPL code, and the decision to
change that is a one-line, greppable, reviewable act rather than a transitive dependency
nobody noticed.

## If you want OpenBB's breadth without the license

Options, roughly in order of effort:

| Approach | Copyleft reach | Effort |
|---|---|---|
| Internal research only, nothing distributed or network-served | None | None |
| Replace with commercially-licensed feeds (Polygon, Databento, Alpaca, IBKR) | None | Low–medium |
| Run OpenBB as a **separate process** behind a network boundary you do not distribute | Contested — do not rely on this without counsel | Medium |
| Negotiate a commercial license with OpenBB | None | Varies |

The second row is the recommended path for anything commercial. The provider abstraction
already exists precisely so that swapping the adapter is a contained change.

## Other dependencies

The core dependencies are permissively licensed:

| Package | License |
|---|---|
| numpy, pandas | BSD-3-Clause |
| pydantic, pydantic-settings | MIT |
| rich, typer | MIT |
| anthropic | MIT |
| yfinance (optional) | Apache-2.0 |

**yfinance caveat:** the license is permissive, but it consumes an *unofficial* Yahoo
endpoint. Yahoo's terms of service govern that usage independently of the package's license.
It is fine for research and prototyping; do not route production decisions through it.

## This repository

AXIOM itself is proprietary. `pyproject.toml` declares `license = { text = "Proprietary" }`.
Add an explicit `LICENSE` file stating your terms before sharing the code with anyone.
