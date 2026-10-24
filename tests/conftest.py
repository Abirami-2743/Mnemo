from __future__ import annotations

import pytest

from mnemo.catalog import load_catalog
from mnemo.config import load_experiment_config
from mnemo.store import InMemoryStore


@pytest.fixture(scope="session")
def cfg():
    return load_experiment_config()


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture()
def store():
    return InMemoryStore()
