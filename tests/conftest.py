import pytest

from firisk.engine import run_analysis


@pytest.fixture(scope="session")
def analysis():
    return run_analysis()
