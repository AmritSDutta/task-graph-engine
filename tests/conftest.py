import sys
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def set_test_api_keys(monkeypatch):
    """Set required API keys for all tests.

    This fixture runs automatically for all tests and ensures
    proper isolation using monkeypatch for cleanup.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-for-pytest")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-for-pytest")
    yield
    # Cleanup happens automatically via monkeypatch
