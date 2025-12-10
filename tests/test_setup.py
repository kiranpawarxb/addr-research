"""Basic test to verify pytest setup."""

import pytest


def test_setup_verification():
    """Verify that the test environment is properly configured."""
    assert True, "Basic test should pass"


def test_imports():
    """Verify that core dependencies can be imported."""
    try:
        import pandas
        import requests
        import yaml
        import rapidfuzz
        import tqdm
        import hypothesis
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required dependency: {e}")
