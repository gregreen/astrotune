"""Test-session setup.

TensorFlow reserves nearly all of the GPU unless memory growth is enabled *before*
the first operation. This GPU is shared with training runs, so a test session
without growth starves whatever else is running: a full-suite run held 75 GB
while a training job could only get 4.5 GB, and the job's step time went from
1.6 s to 5+ s per step.

:func:`astrotune.model.io.enable_memory_growth` is called here, once, for the
whole session, so a test run allocates only what it needs. The training entry
point calls it itself; only the tests were missing it.
"""

import pytest

try:  # the pure-python test modules must still run without tensorflow
    from astrotune.model.io import enable_memory_growth
except Exception:  # pragma: no cover - tensorflow not installed
    enable_memory_growth = None


def pytest_configure(config):
    if enable_memory_growth is None:
        return
    try:
        enable_memory_growth()
    except Exception:  # pragma: no cover - no GPU, or already initialised
        pass
