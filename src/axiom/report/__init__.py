"""Rendering a pipeline run as a browsable report.

The terminal draws one frame for an operator sitting in front of it. This
package produces the same measured content as a single self-contained HTML
file that can be opened later, on another machine, by someone who was not
there when it ran.

Two properties are load-bearing:

*   **the payload is the report.** :func:`build_payload` produces a plain
    JSON-serialisable dict from a real :class:`~axiom.agents.pipeline.PipelineResult`;
    :func:`render_html` only formats it. Nothing is computed during rendering,
    so the page cannot show a number the engines did not produce.
*   **provenance survives the trip.** Synthetic runs render a banner that
    cannot be dismissed, and every metric block repeats the data source. A
    file that outlives the session it came from must not be mistakable for a
    track record.
"""

from axiom.report.html import render_html
from axiom.report.payload import build_payload, write_report

__all__ = ["build_payload", "render_html", "write_report"]
