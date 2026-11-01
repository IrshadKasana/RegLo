"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _clean_reglo_env(monkeypatch):
    """Tests always use the repository's own configs and paths, whatever the shell exports, and run
    at the smoke_offline profile's scale (no GPU, no downloads) unless a test names another profile."""
    for var in ("REGLO_ROOT", "REGLO_DATA", "REGLO_ARTIFACTS", "REGLO_CONFIGS", "REGLO_MAX_HOURS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REGLO_PROFILE", "smoke_offline")
