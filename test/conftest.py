from collections.abc import Iterator

import pytest

from hstar.metrics import log_counters


@pytest.fixture(scope="session", autouse=True)
def _log_metrics() -> Iterator[None]:
    yield
    log_counters()
