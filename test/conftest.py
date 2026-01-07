import pytest
import numpy as np
import random as rand
from pathlib import Path
import logging


_logger = logging.getLogger(__name__)


@pytest.fixture
def seed_random():
    rand.seed(123)
    np.random.seed(123)


@pytest.fixture
def np_rng():
    return np.random.default_rng(123)


@pytest.fixture
def resource_dir():
    return Path(__file__).parent / "resources"

