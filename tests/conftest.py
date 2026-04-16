"""pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (model loading, full pipeline runs)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (model loading)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Use --run-slow to run model-loading tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
