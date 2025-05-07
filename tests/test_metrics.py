# tests/test_metrics.py
"""
Unit tests for the PeakResourceUsage and TestMetrics data classes.
"""

import pytest
from pathlib import Path
from dataclasses import dataclass, field, FrozenInstanceError, is_dataclass
from typing import Any

from metrics import PeakResourceUsage, TestMetrics

# --- Tests for PeakResourceUsage ---


def test_peakresourceusage_is_dataclass():
    """Verify PeakResourceUsage is a dataclass."""
    assert is_dataclass(PeakResourceUsage)


def test_peakresourceusage_successful_initialization():
    """Test successful creation with valid data."""
    try:
        res = PeakResourceUsage(
            peak_vram_mb=4096.0, peak_ram_mb=8192
        )  # Test int RAM
        assert res.peak_vram_mb == 4096.0
        assert (
            res.peak_ram_mb == 8192.0
        )  # Assuming conversion or flexible typing
    except ValueError as e:
        pytest.fail(f"Valid PeakResourceUsage initialization failed: {e}")


def test_peakresourceusage_immutability(valid_peak_resources):
    """Test that PeakResourceUsage attributes cannot be modified."""
    with pytest.raises(FrozenInstanceError):
        valid_peak_resources.peak_vram_mb = 512.0  # type: ignore
    with pytest.raises(FrozenInstanceError):
        valid_peak_resources.peak_ram_mb = 1024.0  # type: ignore


def test_peakresourceusage_validation_negative_values():
    """Test validation failure for negative resource values."""
    with pytest.raises(ValueError, match="peak_vram_mb.*non-negative"):
        PeakResourceUsage(peak_vram_mb=-100.0, peak_ram_mb=1024.0)
    with pytest.raises(ValueError, match="peak_ram_mb.*non-negative"):
        PeakResourceUsage(peak_vram_mb=1024.0, peak_ram_mb=-50.5)


def test_peakresourceusage_validation_invalid_types():
    """Test validation failure for invalid types."""
    with pytest.raises(ValueError, match="peak_vram_mb.*non-negative number"):
        PeakResourceUsage(peak_vram_mb="high", peak_ram_mb=1024.0)  # type: ignore
    with pytest.raises(ValueError, match="peak_ram_mb.*non-negative number"):
        PeakResourceUsage(peak_vram_mb=1024.0, peak_ram_mb=None)  # type: ignore


# --- Tests for TestMetrics ---


def test_testmetrics_is_dataclass():
    """Verify TestMetrics is a dataclass."""
    assert is_dataclass(TestMetrics)


def test_testmetrics_successful_initialization_completed(
    valid_peak_resources,
):
    """Test successful creation for a 'completed' status."""
    try:
        metrics = TestMetrics(
            test_case_id="test001",
            status="completed",
            generation_time_secs=123.45,
            peak_resources=valid_peak_resources,
            output_video_path=Path("/results/run1/test001/video.mp4"),
            error_message=None,
        )
        assert metrics.test_case_id == "test001"
        assert metrics.status == "completed"
        assert metrics.generation_time_secs == 123.45
        assert metrics.peak_resources == valid_peak_resources
        assert metrics.output_video_path == Path(
            "/results/run1/test001/video.mp4"
        )
        assert metrics.error_message is None
    except (ValueError, TypeError) as e:
        pytest.fail(f"Valid 'completed' TestMetrics initialization failed: {e}")


def test_testmetrics_successful_initialization_failed():
    """Test successful creation for a 'failed' status."""
    try:
        metrics = TestMetrics(
            test_case_id="test002",
            status="failed",
            generation_time_secs=None,  # Failed before completion
            peak_resources=None,  # Monitoring might have failed too
            output_video_path=None,
            error_message="CUDA out of memory",
        )
        assert metrics.test_case_id == "test002"
        assert metrics.status == "failed"
        assert metrics.generation_time_secs is None
        assert metrics.peak_resources is None
        assert metrics.output_video_path is None
        assert metrics.error_message == "CUDA out of memory"
    except (ValueError, TypeError) as e:
        pytest.fail(f"Valid 'failed' TestMetrics initialization failed: {e}")


def test_testmetrics_validation_invalid_test_case_id(valid_peak_resources):
    """Test validation failure for invalid test_case_id."""
    with pytest.raises(ValueError, match="test_case_id.*non-empty string"):
        TestMetrics(
            test_case_id="",
            status="completed",
            generation_time_secs=10.0,
            peak_resources=valid_peak_resources,
            output_video_path="path",
            error_message=None,
        )
    with pytest.raises(ValueError, match="test_case_id.*non-empty string"):
        TestMetrics(
            test_case_id=None,
            status="completed",
            generation_time_secs=10.0,  # type: ignore
            peak_resources=valid_peak_resources,
            output_video_path="path",
            error_message=None,
        )


def test_testmetrics_validation_invalid_status(valid_peak_resources):
    """Test validation failure for invalid status."""
    with pytest.raises(ValueError, match="status must be one of"):
        TestMetrics(
            test_case_id="test003",
            status="success",
            generation_time_secs=10.0,
            peak_resources=valid_peak_resources,
            output_video_path="path",
            error_message=None,
        )


def test_testmetrics_validation_invalid_generation_time(
    valid_peak_resources,
):
    """Test validation failure for invalid generation_time_secs."""
    with pytest.raises(ValueError, match="generation_time_secs.*non-negative"):
        TestMetrics(
            test_case_id="test004",
            status="completed",
            generation_time_secs=-5.0,
            peak_resources=valid_peak_resources,
            output_video_path="path",
            error_message=None,
        )
    with pytest.raises(ValueError, match="generation_time_secs.*non-negative"):
        TestMetrics(
            test_case_id="test004",
            status="completed",
            generation_time_secs="fast",  # type: ignore
            peak_resources=valid_peak_resources,
            output_video_path="path",
            error_message=None,
        )


def test_testmetrics_validation_invalid_peak_resources():
    """Test validation failure for invalid peak_resources type."""
    with pytest.raises(
        TypeError, match="peak_resources.*PeakResourceUsage instance"
    ):
        TestMetrics(
            test_case_id="test005",
            status="completed",
            generation_time_secs=10.0,
            peak_resources={"vram": 100},
            output_video_path="path",
            error_message=None,
        )  # type: ignore


def test_testmetrics_validation_invalid_output_path(valid_peak_resources):
    """Test validation failure for invalid output_video_path type."""
    with pytest.raises(
        TypeError,
        match="TestMetrics output_video_path must be a Path instance or None, got .*",
    ):
        TestMetrics(
            test_case_id="test006",
            status="completed",
            generation_time_secs=10.0,
            peak_resources=valid_peak_resources,
            output_video_path=123,
            error_message=None,
        )  # type: ignore


def test_testmetrics_validation_error_message_mismatch():
    """Test validation failure when error_message doesn't match status."""
    # Error message provided but status is 'completed'
    with pytest.raises(
        ValueError,
        match="TestMetrics for .* is 'completed' but has an error_message: .*",
    ):
        TestMetrics(
            test_case_id="test007",
            status="completed",
            generation_time_secs=10.0,
            peak_resources=None,
            output_video_path="path",
            error_message="Should not be here",
        )
    # Status is 'failed' but error_message is None
    with pytest.raises(
        ValueError,
        match="TestMetrics for .* is 'failed' but has no error_message.",
    ):
        TestMetrics(
            test_case_id="test008",
            status="failed",
            generation_time_secs=None,
            peak_resources=None,
            output_video_path=None,
            error_message=None,
        )
    # Status is 'failed' but error_message is empty
    with pytest.raises(
        ValueError,
        match="TestMetrics for .* is 'failed' but has no error_message.",
    ):
        TestMetrics(
            test_case_id="test009",
            status="failed",
            generation_time_secs=None,
            peak_resources=None,
            output_video_path=None,
            error_message="",
        )
