import pytest


def pytest_addoption(parser):
    group = parser.getgroup("nonortho")
    group.addoption(
        "--mode",
        action="store",
        default="NVT",
        help="Simulation mode for the nonortho long-run RDF benchmark: NVE, NVT, or NPT.",
    )
    group.addoption(
        "--steps",
        action="store",
        type=int,
        default=50000,
        help="Step limit for the nonortho long-run RDF benchmark.",
    )


@pytest.fixture(scope="session")
def nonortho_mode(request):
    value = str(request.config.getoption("--mode")).strip().upper()
    if value not in {"NVE", "NVT", "NPT"}:
        raise ValueError("--mode must be one of NVE, NVT, or NPT")
    return value


@pytest.fixture(scope="session")
def nonortho_steps(request):
    value = int(request.config.getoption("--steps"))
    if value <= 0:
        raise ValueError("--steps must be positive")
    return value
