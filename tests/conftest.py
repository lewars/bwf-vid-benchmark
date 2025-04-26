# tests/conftest.py
"""
Pytest configuration file for shared fixtures and hooks.
"""

import pytest
from dataclasses import dataclass, field, FrozenInstanceError, is_dataclass
from typing import Any

from metrics import PeakResourceUsage

# --- Shared Test Data Fixtures ---

@pytest.fixture(scope="session")
def valid_test_case_data() -> dict[str, Any]:
    """Provides a dictionary of valid data for TestCase initialization."""
    return {
        "id": "test001",
        "model_name": "Mochi1",
        "prompt": "A cat riding a skateboard",
        "resolution": (512, 512),
        "duration_secs": 5.0,
        "fps": 15,
        "seed": 42,
        "extra_params": {"guidance_scale": 7.5}
    }

@pytest.fixture(scope="session")
def valid_test_case_data_minimal() -> dict[str, Any]:
    """Provides a dictionary of minimal valid data for TestCase initialization (uses defaults)."""
    return {
        "id": "test002",
        "model_name": "Hunyuan",
        "prompt": "Sunset over mountains",
        "resolution": (1024, 576),
        "duration_secs": 10, # Test with int duration
        "fps": 24,
        # seed and extra_params omitted to test defaults
    }

@pytest.fixture
def valid_peak_resources() -> PeakResourceUsage:
    """Returns a valid PeakResourceUsage instance."""
    # Use the placeholder class for now
    return PeakResourceUsage(peak_vram_mb=1024.5, peak_ram_mb=2048.0)
