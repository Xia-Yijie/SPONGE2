from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def outputs_path():
    path = Path(__file__).parent.parent / "outputs"
    if not path.exists():
        path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def statics_path():
    return Path(__file__).parent.parent / "statics"
