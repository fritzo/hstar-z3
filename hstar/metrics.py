"""
# Metrics and profiling utilities.

These are observability tools whose growth is logarithmic in time.
"""

import atexit
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping

logger = logging.getLogger(__name__)

COUNTERS: Mapping[str, Counter[str]] = defaultdict(Counter)
"""Global counters for function calls and errors. Not thread safe."""


@atexit.register
def log_counters() -> None:
    """Logs counter statistics."""
    if not any(v for counter in COUNTERS.values() for v in counter.values()):
        return
    table = [("count", "counter", "key")] + [
        (str(count), name, key)
        for name, counter in sorted(COUNTERS.items())
        for key, count in counter.most_common()
    ]
    widths = [max(len(row[i]) for row in table) for i in range(3)]
    template = f"{{:>{widths[0]}}} {{:<{widths[1]}}} {{:<{widths[2]}}}"
    lines = ["Counter info:"]
    lines.extend(template.format(*row) for row in table)
    logger.info("\n".join(lines))
