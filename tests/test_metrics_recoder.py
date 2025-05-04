# tests/test_metrics_recorder.py
"""
Unit tests for the MetricsRecorder class.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from typing import List

# Import classes from src/metrics.py (adjust if needed, assumes PYTHONPATH)
from metrics import MetricsRecorder, TestMetrics, PeakResourceUsage

# --- Test Functions ---

def test_metrics_recorder_initialization(tmp_path: Path):
    """Test successful initialization and run directory creation."""
    base_dir = tmp_path / "results"
    run_id = "test_run_123"
    expected_run_dir = base_dir / f"{run_id}_run"

    assert not expected_run_dir.exists() # Ensure it doesn't exist beforehand

    try:
        recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)
        assert recorder.run_dir_path == expected_run_dir
        assert expected_run_dir.exists() # Check directory was created
        assert expected_run_dir.is_dir()
    except IOError as e:
        pytest.fail(f"MetricsRecorder initialization failed to create directory: {e}")

def test_metrics_recorder_initialization_dir_already_exists(tmp_path: Path):
    """Test initialization when the run directory already exists."""
    base_dir = tmp_path / "results_exist"
    run_id = "test_run_456"
    expected_run_dir = base_dir / f"{run_id}_run"
    expected_run_dir.mkdir(parents=True, exist_ok=True) # Create it beforehand

    try:
        recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)
        assert recorder.run_dir_path == expected_run_dir
        assert expected_run_dir.exists() # Should still exist
    except IOError as e:
        pytest.fail(f"MetricsRecorder initialization failed when directory exists: {e}")

# Mocking Path.mkdir to simulate permission errors might be complex,
# focus on successful creation and file writing for now.

def test_record_detailed_metrics_success(
    tmp_path: Path,
    completed_test_metrics: TestMetrics # Use fixture from conftest
):
    """Test saving detailed metrics for a single test case to JSON."""
    base_dir = tmp_path / "detailed_results"
    run_id = "run_detailed_abc"
    test_subdir_name = completed_test_metrics.test_case_id # Use ID for subdir

    recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)
    expected_test_dir = recorder.run_dir_path / test_subdir_name
    expected_json_path = expected_test_dir / "metrics.json"

    assert not expected_test_dir.exists()

    try:
        saved_path = recorder.record_detailed_metrics(
            metrics=completed_test_metrics,
            test_case_subdir=test_subdir_name
        )
        assert saved_path == expected_json_path
        assert expected_test_dir.exists() and expected_test_dir.is_dir()
        assert expected_json_path.exists() and expected_json_path.is_file()

        # Verify JSON content
        with open(expected_json_path, 'r') as f:
            data = json.load(f)

        assert data["test_case_id"] == completed_test_metrics.test_case_id
        assert data["status"] == completed_test_metrics.status
        assert data["generation_time_secs"] == completed_test_metrics.generation_time_secs
        # Check nested PeakResourceUsage
        assert data["peak_resources"]["peak_vram_mb"] == completed_test_metrics.peak_resources.peak_vram_mb
        assert data["peak_resources"]["peak_ram_mb"] == completed_test_metrics.peak_resources.peak_ram_mb
        # Check Path conversion
        assert data["output_video_path"] == str(completed_test_metrics.output_video_path)
        assert data["error_message"] is None

    except (IOError, TypeError, json.JSONDecodeError) as e:
        pytest.fail(f"record_detailed_metrics failed: {e}")

def test_record_detailed_metrics_invalid_subdir(tmp_path: Path, completed_test_metrics: TestMetrics):
    """Test recording detailed metrics with an invalid subdirectory name."""
    base_dir = tmp_path / "invalid_subdir_test"
    run_id = "run_invalid_subdir"
    recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)

    with pytest.raises(ValueError, match="test_case_subdir must be a non-empty string"):
        recorder.record_detailed_metrics(metrics=completed_test_metrics, test_case_subdir="")
    with pytest.raises(ValueError, match="test_case_subdir must be a non-empty string"):
        recorder.record_detailed_metrics(metrics=completed_test_metrics, test_case_subdir=None) # type: ignore


def test_compile_summary_csv_success(
    tmp_path: Path,
    completed_test_metrics: TestMetrics,
    failed_test_metrics: TestMetrics,
    minimal_completed_test_metrics: TestMetrics
):
    """Test compiling and saving the summary CSV from multiple metrics."""
    base_dir = tmp_path / "summary_results"
    run_id = "run_summary_xyz"
    recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)
    expected_csv_path = recorder.run_dir_path / "summary.csv"

    all_metrics_list = [completed_test_metrics, failed_test_metrics, minimal_completed_test_metrics]

    assert not expected_csv_path.exists()

    try:
        saved_path = recorder.compile_summary_csv(all_metrics=all_metrics_list)
        assert saved_path == expected_csv_path
        assert expected_csv_path.exists() and expected_csv_path.is_file()

        # Verify CSV content using pandas
        df = pd.read_csv(expected_csv_path)

        # Check shape and columns
        assert df.shape == (3, len(recorder._SUMMARY_CSV_COLUMNS)) # 3 rows, expected columns
        assert list(df.columns) == recorder._SUMMARY_CSV_COLUMNS

        # Check data for the first record (completed_test_metrics)
        record1 = df.iloc[0]
        assert record1['test_case_id'] == completed_test_metrics.test_case_id
        assert record1['status'] == completed_test_metrics.status
        assert record1['generation_time_secs'] == completed_test_metrics.generation_time_secs
        assert record1['peak_vram_mb'] == completed_test_metrics.peak_resources.peak_vram_mb
        assert record1['peak_ram_mb'] == completed_test_metrics.peak_resources.peak_ram_mb
        assert pd.isna(record1['error_message']) # Check None becomes NaN
        assert record1['output_video_path'] == str(completed_test_metrics.output_video_path)

        # Check data for the second record (failed_test_metrics)
        record2 = df.iloc[1]
        assert record2['test_case_id'] == failed_test_metrics.test_case_id
        assert record2['status'] == failed_test_metrics.status
        assert pd.isna(record2['generation_time_secs'])
        assert pd.isna(record2['peak_vram_mb'])
        assert pd.isna(record2['peak_ram_mb'])
        assert record2['error_message'] == failed_test_metrics.error_message
        assert pd.isna(record2['output_video_path'])

        # Check data for the third record (minimal_completed_test_metrics)
        record3 = df.iloc[2]
        assert record3['test_case_id'] == minimal_completed_test_metrics.test_case_id
        assert record3['status'] == minimal_completed_test_metrics.status
        assert pd.isna(record3['generation_time_secs']) # Was None
        assert pd.isna(record3['peak_vram_mb'])       # Was None
        assert pd.isna(record3['peak_ram_mb'])        # Was None
        assert pd.isna(record3['error_message'])      # Was None
        assert pd.isna(record3['output_video_path'])  # Was None


    except (IOError, ValueError, pd.errors.EmptyDataError) as e:
        pytest.fail(f"compile_summary_csv failed: {e}")


def test_compile_summary_csv_empty_list(tmp_path: Path):
    """Test compiling summary CSV with an empty input list."""
    base_dir = tmp_path / "summary_empty"
    run_id = "run_summary_empty"
    recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)

    with pytest.raises(ValueError, match="Cannot compile summary CSV: 'all_metrics' list is empty"):
        recorder.compile_summary_csv(all_metrics=[])

def test_compile_summary_csv_invalid_list_type(tmp_path: Path, completed_test_metrics: TestMetrics):
    """Test compiling summary CSV with invalid data types in the list."""
    base_dir = tmp_path / "summary_invalid_type"
    run_id = "run_summary_invalid"
    recorder = MetricsRecorder(base_output_dir=base_dir, run_id=run_id)

    invalid_list = [completed_test_metrics, {"not": "a TestMetrics object"}]

    with pytest.raises(ValueError, match="must be a list of TestMetrics objects"):
        recorder.compile_summary_csv(all_metrics=invalid_list) # type: ignore
