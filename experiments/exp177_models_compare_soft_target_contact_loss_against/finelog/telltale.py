"""Compatibility shim for Iris builds that import ``finelog.telltale``.

The current CoreWeave package index serves an Iris/Levanter build that expects
``finelog.telltale.FinelogMetricSink``, but the newest available
``marin-finelog`` wheel on that index predates the module. Telltale forwarding is
best-effort operational telemetry, so this sink intentionally drops forwarded
metric rows rather than blocking training startup.
"""

from collections.abc import Sequence
from typing import Any


class FinelogMetricSink:
    """No-op sink implementing ``rigging.telltale.MetricSink``."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def write(self, rows: Sequence[Any]) -> None:
        del rows

    def close(self) -> None:
        pass
