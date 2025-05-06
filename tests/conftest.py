# tests/conftest.py
"""
Pytest configuration file for shared fixtures and hooks.
"""

import pytest
import yaml
import os
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
from typing import Any, Dict, Callable, List, Union

from test_case import TestCase
from video_generator import VideoGenerator, GeneratedVideo, ModelAdapter
from metrics import PeakResourceUsage, TestMetrics
from monitor import ResourceMonitor
from orchestrator import BenchmarkOrchestrator


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
        "extra_params": {"guidance_scale": 7.5},
    }


@pytest.fixture(scope="session")
def valid_test_case_data_minimal() -> dict[str, Any]:
    """Provides a dictionary of minimal valid data for TestCase initialization (uses defaults)."""
    return {
        "id": "test002",
        "model_name": "Hunyuan",
        "prompt": "Sunset over mountains",
        "resolution": (1024, 576),
        "duration_secs": 10,  # Test with int duration
        "fps": 24,
        # seed and extra_params omitted to test defaults
    }


@pytest.fixture
def valid_peak_resources() -> PeakResourceUsage:
    """Returns a valid PeakResourceUsage instance."""
    return PeakResourceUsage(peak_vram_mb=1024.5, peak_ram_mb=2048.0)


@pytest.fixture
def completed_test_metrics(
    valid_peak_resources: PeakResourceUsage,
) -> TestMetrics:
    """Returns a valid TestMetrics instance for a completed test."""
    return TestMetrics(
        test_case_id="test_completed_01",
        status="completed",
        generation_time_secs=123.45,
        peak_resources=valid_peak_resources,
        output_video_path=Path("results/run_abc/test_completed_01/video.mp4"),
        error_message=None,
    )


@pytest.fixture
def failed_test_metrics() -> TestMetrics:
    """Returns a valid TestMetrics instance for a failed test."""
    return TestMetrics(
        test_case_id="test_failed_02",
        status="failed",
        generation_time_secs=None,  # Failed before completion
        peak_resources=None,  # Monitoring might have failed
        output_video_path=None,
        error_message="CUDA out of memory",
    )


@pytest.fixture
def minimal_completed_test_metrics() -> TestMetrics:
    """Returns a minimal valid TestMetrics instance for a completed test."""
    return TestMetrics(
        test_case_id="test_minimal_03",
        status="completed",
        # All optional fields are None by default
    )


@pytest.fixture
def yaml_file_creator(
    tmp_path: Path,
) -> Callable[[str, Dict[str, Any]], Path]:
    """
    Provides a fixture that yields a function to create YAML files
    in the temporary test directory (`tmp_path`). Automatically cleans up
    created files after the test using this fixture finishes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Yields:
        A function `_creator(filename: str, content: Dict[str, Any]) -> Path`
        which creates the specified YAML file and returns its path.
    """
    created_files: List[Path] = (
        []
    )  # Keep track of files created by this fixture instance

    def _creator(filename: str, content: Dict[str, Any]) -> Path:
        """Creates a YAML file in the temporary directory."""
        file_path = tmp_path / filename
        try:
            with open(file_path, "w") as f:
                yaml.dump(content, f)
            created_files.append(file_path)  # Track successful creation
        except Exception as e:
            # Fail the test if file creation doesn't work
            pytest.fail(f"Failed to create YAML file {file_path}: {e}")
        return file_path

    yield _creator  # Yield the creator function to the test

    # --- Cleanup Phase ---
    # This code runs after the test function finishes
    for file_path in created_files:
        try:
            if (
                file_path.exists()
            ):  # Check if file still exists before trying to delete
                os.remove(file_path)
                # print(f"Cleaned up: {file_path}") # Optional: for debugging
        except OSError as e:
            # Log error if cleanup fails, but don't fail the test itself
            print(
                f"Warning: Failed to clean up temporary file {file_path}: {e}"
            )


@pytest.fixture
def csv_file_creator(
    tmp_path: Path,
) -> Callable[[str, Union[List[Dict[str, Any]], pd.DataFrame]], Path]:
    """
    Provides a fixture that yields a function to create CSV files
    in the temporary test directory (`tmp_path`). Automatically cleans up
    created files after the test using this fixture finishes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Yields:
        A function `_creator(filename: str, data: Union[List[Dict[str, Any]], pd.DataFrame]) -> Path`
        which creates the specified CSV file and returns its path.
    """
    created_files: List[Path] = []

    def _creator(
        filename: str, data: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> Path:
        """Creates a CSV file in the temporary directory from list of dicts or DataFrame."""
        file_path = tmp_path / filename
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise TypeError(
                    "Data for CSV creation must be a list of dicts or a pandas DataFrame."
                )

            df.to_csv(file_path, index=False, encoding="utf-8")
            created_files.append(file_path)
        except Exception as e:
            pytest.fail(f"Failed to create CSV file {file_path}: {e}")
        return file_path

    yield _creator

    # --- Cleanup Phase ---
    for file_path in created_files:
        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError as e:
            print(
                f"Warning: Failed to clean up temporary file {file_path}: {e}"
            )


# --- Sample Data for Analyzer Tests ---


@pytest.fixture
def sample_summary_data() -> List[Dict[str, Any]]:
    """Provides sample data representing a summary CSV content."""
    return [
        {
            "test_case_id": "test001",
            "status": "completed",
            "generation_time_secs": 120.5,
            "peak_vram_mb": 8192.0,
            "peak_ram_mb": 16384.0,
            "error_message": None,
        },
        {
            "test_case_id": "test002",
            "status": "failed",
            "generation_time_secs": None,
            "peak_vram_mb": 4096.0,
            "peak_ram_mb": 8192.0,
            "error_message": "OOM",
        },
        {
            "test_case_id": "test003",
            "status": "completed",
            "generation_time_secs": 240.0,
            "peak_vram_mb": 12288.0,
            "peak_ram_mb": 24576.0,
            "error_message": None,
        },
    ]


@pytest.fixture
def mock_csv_path(tmp_path: Path) -> Path:
    """Provides a path for a mock CSV file within the temp directory."""
    return tmp_path / "summary_test.csv"


# Mock pynvml if it wasn't available during import in monitor.py
# This allows testing the logic even if the real pynvml isn't installed
if "monitor.pynvml" not in sys.modules:
    # If pynvml failed to import in monitor.py, create a mock for it
    # This is tricky because the import happens at module level.
    # A common approach is to structure the code to allow injection or
    # ensure pynvml is mocked *before* monitor.py is imported by pytest.
    # For simplicity here, we'll rely on patching within tests.
    # If tests fail due to PYNXML_AVAILABLE being False when it shouldn't be,
    # consider using pytest-mock's `module_mocker`.
    pass


@pytest.fixture
def mock_pynvml():
    """Fixture to mock the pynvml library."""
    # Create mocks for pynvml structures if needed
    mock_mem_info = MagicMock()
    mock_mem_info.used = 0  # Default value

    mock_handle = MagicMock(name="NVMLHandle")

    # Mock the pynvml module itself
    pynvml_mock = MagicMock(name="pynvml_module")
    pynvml_mock.NVMLError = Exception  # Mock the exception type
    pynvml_mock.nvmlInit.return_value = None
    pynvml_mock.nvmlShutdown.return_value = None
    pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

    return pynvml_mock, mock_handle, mock_mem_info


@pytest.fixture
def mock_psutil():
    """Fixture to mock the psutil library."""
    mock_process = MagicMock(name="psutil_Process")
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 0  # Default value
    mock_process.memory_info.return_value = mock_mem_info

    psutil_mock = MagicMock(name="psutil_module")
    psutil_mock.Process.return_value = mock_process
    psutil_mock.NoSuchProcess = ProcessLookupError  # Mock exception type

    return psutil_mock, mock_process, mock_mem_info


# --- Mock Adapter Classes ---
# Create mock classes that conform to the ModelAdapter protocol for testing

class _MockAdapterAImpl:
    """Mock adapter class A conforming to ModelAdapter protocol."""

    def __init__(self):
        # Use MagicMock for methods to track calls
        self.load_model = MagicMock(name="load_model_A")
        self.unload_model = MagicMock(name="unload_model_A")
        self.generate = MagicMock(name="generate_A")
        # Mock generate to return a GeneratedVideo object
        self.generate.return_value = GeneratedVideo(
            file_path=Path("mock_a_video.mp4")
        )

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def generate(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        pass


class _MockAdapterBImpl:
    """Mock adapter class B conforming to ModelAdapter protocol."""

    def __init__(self):
        self.load_model = MagicMock(name="load_model_B")
        self.unload_model = MagicMock(name="unload_model_B")
        self.generate = MagicMock(name="generate_B")
        self.generate.return_value = GeneratedVideo(
            file_path=Path("mock_b_video.mp4")
        )

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def generate(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        pass


# --- Fixtures ---


@pytest.fixture
def video_generator() -> VideoGenerator:
    """Fixture to create a VideoGenerator instance for each test."""
    return VideoGenerator()


@pytest.fixture
def test_case_a(valid_test_case_data) -> TestCase:
    """Fixture for a TestCase configured for MockAdapterA."""
    data = valid_test_case_data.copy()
    data["model_name"] = "ModelA"  # Match the registration key
    # Need the actual TestCase class from src/test_case.py

    return TestCase(**data)


@pytest.fixture
def test_case_b(valid_test_case_data_minimal) -> TestCase:
    """Fixture for a TestCase configured for MockAdapterB."""
    data = valid_test_case_data_minimal.copy()
    data["model_name"] = "ModelB"  # Match the registration key

    return TestCase(**data)


@pytest.fixture
def mock_test_cases() -> List[TestCase]:
    """Provides a list of mock TestCase objects."""
    # Use actual TestCase if available and imported, otherwise placeholder
    try:
        # Use data from conftest if TestCase is real
        from test_case import TestCase as RealTestCase

        # Need valid_test_case_data fixtures accessible here
        # Simplification: create basic valid instances directly
        return [
            RealTestCase(
                id="t1",
                model_name="Mochi1",
                prompt="p1",
                resolution=(1, 1),
                duration_secs=1.0,
                fps=1,
            ),
            RealTestCase(
                id="t2",
                model_name="Hunyuan",
                prompt="p2",
                resolution=(2, 2),
                duration_secs=2.0,
                fps=2,
            ),
            RealTestCase(
                id="t3",
                model_name="Mochi1",
                prompt="p3",
                resolution=(3, 3),
                duration_secs=3.0,
                fps=3,
            ),
        ]
    except NameError:  # If RealTestCase wasn't imported
        return [
            TestCase(id="t1", model_name="Mochi1"),
            TestCase(id="t2", model_name="Hunyuan"),
            TestCase(id="t3", model_name="Mochi1"),
        ]


@pytest.fixture
def mock_paths(tmp_path: Path) -> Dict[str, Path]:
    """Provides mock paths for config and results."""
    return {
        "config": tmp_path / "test_cases.yaml",
        "results": tmp_path / "results",
    }


@pytest.fixture
def orchestrator_instance(
    mock_paths: Dict[str, Path],
) -> BenchmarkOrchestrator:
    """Provides a basic BenchmarkOrchestrator instance."""
    return BenchmarkOrchestrator(
        test_cases_yaml_path=mock_paths["config"],
        base_results_dir=mock_paths["results"],
    )

@pytest.fixture
def MockAdapterA() -> type: # Fixture named MockAdapterA, returns a type
    """Pytest fixture that returns the MockAdapterA class."""
    return _MockAdapterAImpl # Return the actual class object

@pytest.fixture
def MockAdapterB() -> type: # Fixture named MockAdapterB, returns a type
    """Pytest fixture that returns the MockAdapterB class."""
    return _MockAdapterBImpl # Return the actual class object
