# tests/test_orchestrator.py
"""
Unit tests for the BenchmarkOrchestrator class.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
from typing import Dict, List, Optional

from test_case import TestCase, TestCaseLoadError
from dataclasses import dataclass
from video_generator import VideoGenerator, GeneratedVideo
from metrics import MetricsRecorder, TestMetrics, PeakResourceUsage
from monitor import ResourceMonitor
from orchestrator import BenchmarkOrchestrator

# --- Mocks for Dependencies (applied via @patch) ---

# Patch dependencies within the 'orchestrator' module's namespace
@patch('orchestrator.datetime')
@patch('orchestrator.load_test_cases_from_yaml')
@patch('orchestrator.VideoGenerator')
@patch('orchestrator.MetricsRecorder')
@patch('orchestrator.ResourceMonitor')
@patch('orchestrator.Mochi1Adapter') # Patch concrete adapters too
@patch('orchestrator.HunyuanAdapter')
def test_orchestrator_setup(
    mock_HunyuanAdapter: MagicMock, mock_Mochi1Adapter: MagicMock,
    mock_ResourceMonitor: MagicMock, mock_MetricsRecorder: MagicMock,
    mock_VideoGenerator: MagicMock, mock_load_test_cases: MagicMock,
    mock_datetime: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator, # Use the fixture
    mock_test_cases: List[TestCase],
    mock_paths: Dict[str, Path]
):
    """Test the setup method: component initialization and test case loading."""
    # --- Mock Configuration ---
    mock_datetime.now.return_value.strftime.return_value = "mock_timestamp"
    mock_load_test_cases.return_value = mock_test_cases
    # Mock instances returned by constructors
    mock_vg_instance = MagicMock()
    mock_mr_instance = MagicMock()
    mock_VideoGenerator.return_value = mock_vg_instance
    mock_MetricsRecorder.return_value = mock_mr_instance

    # --- Call setup ---
    orchestrator_instance.setup()

    # --- Assertions ---
    # Timestamp
    assert orchestrator_instance.run_timestamp == "mock_timestamp"
    # Component Initialization
    mock_VideoGenerator.assert_called_once()
    mock_vg_instance.register_adapter.assert_any_call("mochi1", mock_Mochi1Adapter)
    mock_vg_instance.register_adapter.assert_any_call("hunyuan", mock_HunyuanAdapter)
    mock_MetricsRecorder.assert_called_once_with(
        base_results_dir=mock_paths["results"],
        run_id="mock_timestamp"
    )
    assert orchestrator_instance.video_generator is mock_vg_instance
    assert orchestrator_instance.metrics_recorder is mock_mr_instance
    # Test Case Loading
    mock_load_test_cases.assert_called_once_with(mock_paths["config"])
    assert orchestrator_instance.test_cases == mock_test_cases
    # Setup flag
    assert orchestrator_instance._is_setup is True

@patch('orchestrator.load_test_cases_from_yaml')
def test_orchestrator_setup_load_failure(
    mock_load_test_cases: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator # Use fixture
):
    """Test setup failure during test case loading."""
    mock_load_test_cases.side_effect = TestCaseLoadError("YAML parsing failed")

    with pytest.raises(TestCaseLoadError):
        orchestrator_instance.setup() # Call setup which calls _load_test_cases

    assert orchestrator_instance._is_setup is False # Should not be set if setup fails


# Test execute_test_cases requires mocking _run_single_test or its internals
@patch('orchestrator.BenchmarkOrchestrator._run_single_test')
def test_execute_test_cases_all(
    mock_run_single: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator,
    mock_test_cases: List[TestCase]
):
    """Test executing all loaded test cases."""
    # Simulate setup being done
    orchestrator_instance.test_cases = mock_test_cases
    orchestrator_instance._is_setup = True
    # Mock _run_single_test to return dummy metrics
    mock_run_single.side_effect = lambda tc: TestMetrics(test_case_id=tc.id, status="completed")

    orchestrator_instance.execute_test_cases()

    # Assert _run_single_test called for each test case
    assert mock_run_single.call_count == len(mock_test_cases)
    mock_run_single.assert_has_calls([call(tc) for tc in mock_test_cases])
    # Assert metrics collected
    assert len(orchestrator_instance.all_run_metrics) == len(mock_test_cases)
    assert orchestrator_instance.all_run_metrics[0].test_case_id == "t1"
    assert orchestrator_instance.all_run_metrics[1].test_case_id == "t2"


@patch('orchestrator.BenchmarkOrchestrator._run_single_test')
def test_execute_test_cases_specific_ids(
    mock_run_single: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator,
    mock_test_cases: List[TestCase]
):
    """Test executing a subset of test cases by ID."""
    orchestrator_instance.test_cases = mock_test_cases # t1, t2, t3
    orchestrator_instance._is_setup = True
    mock_run_single.side_effect = lambda tc: TestMetrics(test_case_id=tc.id, status="completed")

    ids_to_run = ["t3", "t1"] # Run t3 and t1 only
    orchestrator_instance.execute_test_cases(test_ids_to_run=ids_to_run)

    # Assert _run_single_test called only for specified IDs
    assert mock_run_single.call_count == 2
    mock_run_single.assert_has_calls([call(mock_test_cases[2]), call(mock_test_cases[0])], any_order=True)
    # Assert metrics collected match executed tests
    assert len(orchestrator_instance.all_run_metrics) == 2
    assert {m.test_case_id for m in orchestrator_instance.all_run_metrics} == {"t1", "t3"}


def test_execute_test_cases_requires_setup(orchestrator_instance: BenchmarkOrchestrator):
    """Test that execute_test_cases fails if setup wasn't called."""
    orchestrator_instance._is_setup = False # Ensure setup flag is False
    with pytest.raises(RuntimeError, match="Setup must be performed"):
        orchestrator_instance.execute_test_cases()


# Test _run_single_test (most complex, mocks monitor, vg, recorder interactions)
@patch('orchestrator.time.time')
@patch('orchestrator.ResourceMonitor')
def test_run_single_test_success(
    mock_ResourceMonitor: MagicMock, mock_time: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator, # Has mocked vg, recorder from setup
    mock_test_cases: List[TestCase]
):
    """Test the internal _run_single_test method for a successful case."""
    # --- Mock Setup ---
    test_case = mock_test_cases[0] # Use t1
    # Mock time calls
    mock_time.side_effect = [100.0, 120.5] # Start time, End time
    # Mock ResourceMonitor context manager and stop()
    mock_monitor_instance = MagicMock()
    mock_monitor_instance.stop.return_value = PeakResourceUsage(peak_vram_mb=1024.0, peak_ram_mb=2048.0)
    mock_ResourceMonitor.return_value = mock_monitor_instance
    # Mock VideoGenerator and MetricsRecorder (already mocked if setup was simulated)
    # Let's assume setup was done and components are mocked instances
    mock_vg_instance = MagicMock(spec=VideoGenerator)
    mock_mr_instance = MagicMock(spec=MetricsRecorder)
    orchestrator_instance.video_generator = mock_vg_instance
    orchestrator_instance.metrics_recorder = mock_mr_instance
    # Mock generate_for_test_case return value
    expected_video = GeneratedVideo(file_path=Path("results/run/t1/video.mp4"))
    mock_vg_instance.generate_for_test_case.return_value = expected_video
    # Mock metrics recorder run_dir_path needed for output path construction
    mock_mr_instance.run_dir_path = Path("results/mock_timestamp_run")


    # --- Call Method ---
    result_metrics = orchestrator_instance._run_single_test(test_case)

    # --- Assertions ---
    # ResourceMonitor interactions
    mock_ResourceMonitor.assert_called_once() # Instantiated
    mock_monitor_instance.__enter__.assert_called_once() # Context manager entered
    mock_monitor_instance.__exit__.assert_called_once() # Context manager exited (calls stop)
    # VideoGenerator interaction
    mock_vg_instance.generate_for_test_case.assert_called_once_with(
        test_case=test_case,
        output_dir=mock_mr_instance.run_dir_path / test_case.id
    )
    # MetricsRecorder interaction
    mock_mr_instance.record_detailed_metrics.assert_called_once()
    # Check the metrics passed to record_detailed_metrics
    call_args, _ = mock_mr_instance.record_detailed_metrics.call_args
    recorded_metrics_arg = call_args[0] # The 'metrics' object
    assert isinstance(recorded_metrics_arg, TestMetrics)
    assert recorded_metrics_arg.test_case_id == test_case.id
    assert recorded_metrics_arg.status == "completed"
    assert recorded_metrics_arg.generation_time_secs == pytest.approx(20.5) # 120.5 - 100.0
    assert recorded_metrics_arg.peak_resources == mock_monitor_instance.stop.return_value
    assert recorded_metrics_arg.output_video_path == expected_video.file_path
    assert recorded_metrics_arg.error_message is None
    # Check the subdir passed
    assert call_args[1] == {"test_case_subdir": test_case.id} # Check kwargs if used

    # Check returned metrics object
    assert result_metrics == recorded_metrics_arg


@patch('orchestrator.time.time')
@patch('orchestrator.ResourceMonitor')
def test_run_single_test_generation_fails(
    mock_ResourceMonitor: MagicMock, mock_time: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator,
    mock_test_cases: List[TestCase]
):
    """Test _run_single_test when video_generator.generate_for_test_case fails."""
    # --- Mock Setup ---
    test_case = mock_test_cases[1] # Use t2
    mock_time.side_effect = [200.0] # Only start time is recorded before error
    mock_monitor_instance = MagicMock()
    mock_monitor_instance.stop.return_value = PeakResourceUsage(peak_vram_mb=512.0, peak_ram_mb=1024.0)
    mock_ResourceMonitor.return_value = mock_monitor_instance
    mock_vg_instance = MagicMock(spec=VideoGenerator)
    mock_mr_instance = MagicMock(spec=MetricsRecorder)
    orchestrator_instance.video_generator = mock_vg_instance
    orchestrator_instance.metrics_recorder = mock_mr_instance
    # Make generate fail
    error_message = "CUDA OOM Error"
    mock_vg_instance.generate_for_test_case.side_effect = RuntimeError(error_message)
    mock_mr_instance.run_dir_path = Path("results/mock_timestamp_run")

    # --- Call Method ---
    result_metrics = orchestrator_instance._run_single_test(test_case)

    # --- Assertions ---
    mock_monitor_instance.__enter__.assert_called_once()
    mock_vg_instance.generate_for_test_case.assert_called_once() # Called and failed
    mock_monitor_instance.__exit__.assert_called_once() # Should still be called

    # Check metrics passed to record_detailed_metrics
    mock_mr_instance.record_detailed_metrics.assert_called_once()
    call_args, _ = mock_mr_instance.record_detailed_metrics.call_args
    recorded_metrics_arg = call_args[0]
    assert recorded_metrics_arg.status == "failed"
    assert recorded_metrics_arg.generation_time_secs is None # End time wasn't reached after start
    assert recorded_metrics_arg.peak_resources == mock_monitor_instance.stop.return_value
    assert recorded_metrics_arg.output_video_path is None
    assert error_message in recorded_metrics_arg.error_message # Check error captured
    assert "RuntimeError" in recorded_metrics_arg.error_message

    assert result_metrics == recorded_metrics_arg


def test_compile_summary(orchestrator_instance: BenchmarkOrchestrator):
    """Test the compile_summary method."""
    # Simulate setup and execution having run
    mock_mr_instance = MagicMock(spec=MetricsRecorder)
    orchestrator_instance.metrics_recorder = mock_mr_instance
    orchestrator_instance._is_setup = True
    # Add some dummy metrics collected during execution
    dummy_metrics = [TestMetrics(test_case_id="t1", status="completed"), TestMetrics(test_case_id="t2", status="failed")]
    orchestrator_instance.all_run_metrics = dummy_metrics

    orchestrator_instance.compile_summary()

    # Assert compile_summary_csv was called on the recorder with the collected metrics
    mock_mr_instance.compile_summary_csv.assert_called_once_with(dummy_metrics)


def test_cleanup(orchestrator_instance: BenchmarkOrchestrator):
    """Test the cleanup method."""
    # Simulate setup having run
    mock_vg_instance = MagicMock(spec=VideoGenerator)
    orchestrator_instance.video_generator = mock_vg_instance
    orchestrator_instance._is_setup = True # Assume setup was done

    orchestrator_instance.cleanup()

    # Assert unload_all_adapters was called
    mock_vg_instance.unload_all_adapters.assert_called_once()


# Test the main run() method (integration of other methods)
@patch.object(BenchmarkOrchestrator, 'setup')
@patch.object(BenchmarkOrchestrator, 'execute_test_cases')
@patch.object(BenchmarkOrchestrator, 'compile_summary')
@patch.object(BenchmarkOrchestrator, 'cleanup')
def test_run_method_calls_steps(
    mock_cleanup: MagicMock, mock_compile: MagicMock,
    mock_execute: MagicMock, mock_setup: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator
):
    """Test that the main run() method calls other methods in sequence."""
    orchestrator_instance.run()

    # Assert methods were called in order
    mock_setup.assert_called_once()
    mock_execute.assert_called_once_with() # Called with default args (run all)
    mock_compile.assert_called_once()
    mock_cleanup.assert_called_once()

@patch.object(BenchmarkOrchestrator, 'setup')
@patch.object(BenchmarkOrchestrator, 'cleanup')
def test_run_method_calls_cleanup_on_setup_error(
    mock_cleanup: MagicMock, mock_setup: MagicMock,
    orchestrator_instance: BenchmarkOrchestrator
):
    """Test that run() calls cleanup even if setup fails."""
    mock_setup.side_effect = Exception("Setup Failed!")

    # run() should catch the exception and proceed to finally block
    orchestrator_instance.run()

    mock_setup.assert_called_once()
    mock_cleanup.assert_called_once() # Cleanup should still be called
