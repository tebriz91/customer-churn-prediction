import coverage
import pytest


@pytest.fixture(autouse=True)
def coverage_init():
    """Initialize coverage before any tests run."""
    cov = coverage.Coverage()
    cov.start()
    yield
    cov.stop()
    cov.save()
