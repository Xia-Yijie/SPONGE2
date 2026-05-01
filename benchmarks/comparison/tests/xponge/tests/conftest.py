import importlib
import os

import pytest


def load_xponge_benchmark_module():
    module_name = os.environ.get("XPONGE_BENCH_MODULE", "xponge2")
    return importlib.import_module(module_name)


@pytest.fixture(scope="module")
def xponge_module():
    try:
        return load_xponge_benchmark_module()
    except ImportError as exc:
        pytest.skip(f"xponge benchmark module is not available: {exc}")
