import pytest
from model2vec import StaticModel

from semhash import SemHash


@pytest.fixture
def model() -> StaticModel:
    """Load a model for testing."""
    model = StaticModel.from_pretrained("tests/data/test_model")
    return model


@pytest.fixture
def semhash(model: StaticModel) -> SemHash:
    """Load a SemHash object for testing."""
    semhash = SemHash(model=model)
    return semhash
