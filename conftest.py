"""Pytest configuration: custom markers and shared fixtures."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--overnight", action="store_true", default=False,
        help="Run overnight / long-running tests (MASt3R + Gemini on real video)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "overnight: mark test as overnight-only (long GPU run)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--overnight"):
        # When --overnight is passed, run everything
        return
    skip_overnight = pytest.mark.skip(reason="needs --overnight option to run")
    for item in items:
        if "overnight" in item.keywords:
            item.add_marker(skip_overnight)
