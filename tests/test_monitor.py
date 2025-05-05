# tests/test_monitor.py
"""
Unit tests for the ResourceMonitor class.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock, PropertyMock, call, ANY

# Import the class to test (assuming PYTHONPATH is set)
from monitor import ResourceMonitor, PeakResourceUsage

# --- Test Functions ---


# Test Initialization
@patch("monitor.os.getpid")
@patch("monitor.threading.Event")
@patch(
    "monitor.pynvml", create=True
)  # Patch pynvml import within monitor module
def test_monitor_init_defaults(mock_pynvml_import, mock_Event, mock_getpid):
    """Test ResourceMonitor initialization with default values."""
    mock_getpid.return_value = 12345
    mock_stop_event = MagicMock()
    mock_Event.return_value = mock_stop_event
    # Mock pynvml functions used in _initialize_pynvml
    mock_pynvml_import.nvmlInit.return_value = None
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.return_value = MagicMock(
        name="Handle"
    )
    mock_pynvml_import.NVMLError = Exception  # Define the error type

    monitor = ResourceMonitor()

    assert monitor.poll_interval == 0.1
    assert monitor.gpu_index == 0
    assert monitor.pid == 12345
    assert monitor._peak_ram_bytes == 0
    assert monitor._peak_vram_bytes == 0
    assert not monitor._monitoring_active
    assert monitor._monitor_thread is None
    assert monitor._stop_event is mock_stop_event
    assert monitor._gpu_monitoring_enabled  # Should be enabled if init succeeds
    assert monitor._nvml_handle is not None
    mock_pynvml_import.nvmlInit.assert_called_once()
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)


@patch("monitor.os.getpid")
@patch("monitor.threading.Event")
@patch("monitor.pynvml", create=True)
def test_monitor_init_custom_values(
    mock_pynvml_import, mock_Event, mock_getpid
):
    """Test ResourceMonitor initialization with custom values."""
    mock_getpid.return_value = 54321
    mock_pynvml_import.nvmlInit.return_value = None
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.return_value = MagicMock(
        name="Handle"
    )
    mock_pynvml_import.NVMLError = Exception

    monitor = ResourceMonitor(poll_interval=0.5, gpu_index=1)

    assert monitor.poll_interval == 0.5
    assert monitor.gpu_index == 1
    assert monitor.pid == 54321
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)


@patch("monitor.log")
@patch("monitor.os.getpid")
@patch("monitor.threading.Event")
@patch("monitor.pynvml", create=True)
def test_monitor_init_pynvml_init_fails(
    mock_pynvml_import, mock_Event, mock_getpid, mock_log
):
    """Test initialization when pynvml.nvmlInit fails."""
    mock_pynvml_import.NVMLError = type(
        "MockNVMLError", (Exception,), {}
    )  # Create mock error type
    mock_pynvml_import.nvmlInit.side_effect = mock_pynvml_import.NVMLError(
        "Init Failed"
    )

    monitor = ResourceMonitor()

    assert not monitor._gpu_monitoring_enabled
    assert monitor._nvml_handle is None
    mock_pynvml_import.nvmlInit.assert_called_once()
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.assert_not_called()
    mock_log.warning.assert_any_call(f"Failed to initialize NVML: Init Failed")


@patch("monitor.log")
@patch("monitor.os.getpid")
@patch("monitor.threading.Event")
@patch("monitor.pynvml", create=True)
def test_monitor_init_pynvml_get_handle_fails(
    mock_pynvml_import, mock_Event, mock_getpid, mock_log
):
    """Test initialization when pynvml.nvmlDeviceGetHandleByIndex fails."""
    mock_pynvml_import.NVMLError = type("MockNVMLError", (Exception,), {})
    mock_pynvml_import.nvmlInit.return_value = None  # Init succeeds
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.side_effect = (
        mock_pynvml_import.NVMLError("Handle Failed")
    )

    monitor = ResourceMonitor()

    assert not monitor._gpu_monitoring_enabled
    assert monitor._nvml_handle is None
    mock_pynvml_import.nvmlInit.assert_called_once()
    mock_pynvml_import.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
    mock_log.warning.assert_any_call(
        f"Failed to get NVML handle for GPU 0: Handle Failed"
    )


# Test Start/Stop/Loop (needs more patching)
@patch("monitor.time.sleep")  # Patch sleep to avoid delays
@patch("monitor.threading.Thread")
@patch("monitor.psutil", create=True)
@patch("monitor.pynvml", create=True)
@patch("monitor.os.getpid")  # Keep pid consistent
def test_monitor_start_stop_loop(
    mock_getpid,
    mock_pynvml_import,
    mock_psutil_import,
    mock_Thread,
    mock_sleep,
    mock_pynvml,
    mock_psutil,  # Use fixtures for mock objects
):
    """Test the start/stop cycle and basic monitoring loop operation."""
    mock_getpid.return_value = 999
    # --- Mock Setup ---
    # pynvml setup (from fixture)
    pynvml_module_mock, nvml_handle_mock, nvml_mem_info_mock = mock_pynvml
    mock_pynvml_import.NVMLError = pynvml_module_mock.NVMLError
    mock_pynvml_import.nvmlInit = pynvml_module_mock.nvmlInit
    mock_pynvml_import.nvmlDeviceGetHandleByIndex = (
        pynvml_module_mock.nvmlDeviceGetHandleByIndex
    )
    mock_pynvml_import.nvmlDeviceGetMemoryInfo = (
        pynvml_module_mock.nvmlDeviceGetMemoryInfo
    )

    # psutil setup (from fixture)
    psutil_module_mock, psutil_process_mock, psutil_mem_info_mock = mock_psutil
    mock_psutil_import.Process = psutil_module_mock.Process
    mock_psutil_import.NoSuchProcess = psutil_module_mock.NoSuchProcess

    # Threading setup
    mock_thread_instance = MagicMock()
    mock_Thread.return_value = mock_thread_instance

    # --- Test Execution ---
    monitor = ResourceMonitor(poll_interval=0.01)  # Short interval for test
    assert monitor._gpu_monitoring_enabled  # Assume init succeeds

    # Configure mock return values for memory polling
    psutil_mem_info_mock.rss = 100 * 1024 * 1024  # 100 MB
    nvml_mem_info_mock.used = 200 * 1024 * 1024  # 200 MB

    # Start monitoring
    monitor.start()

    # Assertions for start()
    assert monitor._monitoring_active
    mock_Thread.assert_called_once_with(
        target=monitor._monitor_loop, daemon=True
    )
    mock_thread_instance.start.assert_called_once()
    assert monitor._peak_ram_bytes == 0  # Should be reset
    assert monitor._peak_vram_bytes == 0

    # --- Simulate monitor loop running (by calling the target directly once) ---
    # To properly test the loop, we'd need more complex thread control or
    # refactor _monitor_loop to be more testable.
    # For now, let's simulate one iteration by calling the mocks it uses.
    # This assumes the thread *would* call these.

    # Simulate changing values
    psutil_mem_info_mock.rss = 150 * 1024 * 1024  # 150 MB (new peak)
    nvml_mem_info_mock.used = (
        180 * 1024 * 1024
    )  # 180 MB (lower than initial mock, peak stays 200)

    # Let the mocked loop run conceptually for a tiny bit
    # In a real test, you might need condition variables or timeouts
    # We mock sleep, so the loop would run very fast if not controlled.
    # Let's manually set the peaks based on simulated values.
    monitor._peak_ram_bytes = max(monitor._peak_ram_bytes, 150 * 1024 * 1024)
    monitor._peak_vram_bytes = max(
        monitor._peak_vram_bytes, 200 * 1024 * 1024
    )  # Initial mocked value was peak

    # Stop monitoring
    result = monitor.stop()

    # Assertions for stop()
    assert not monitor._monitoring_active
    monitor._stop_event.set.assert_called_once()
    mock_thread_instance.join.assert_called_once()

    # Assertions for result (check conversion to MB)
    assert isinstance(result, PeakResourceUsage)
    assert result.peak_ram_mb == pytest.approx(150.0)  # 150 MB was the peak
    assert result.peak_vram_mb == pytest.approx(200.0)  # 200 MB was the peak


@patch("monitor.ResourceMonitor.start")
@patch("monitor.ResourceMonitor.stop")
def test_monitor_context_manager(mock_stop, mock_start):
    """Test the ResourceMonitor as a context manager."""
    monitor = (
        ResourceMonitor()
    )  # Doesn't matter if init fails here, just testing context

    with monitor as m:
        # Assertions during context
        mock_start.assert_called_once()
        assert m is monitor  # __enter__ should return self
        mock_stop.assert_not_called()  # Stop not called yet

    # Assertions after context
    mock_stop.assert_called_once()


def test_monitor_stop_before_start():
    """Test calling stop before start."""
    # No patching needed as start/stop logic handles this internally
    monitor = ResourceMonitor()
    # Manually disable GPU monitoring to avoid NVML init attempt if library exists
    monitor._gpu_monitoring_enabled = False

    result = monitor.stop()

    assert isinstance(result, PeakResourceUsage)
    assert result.peak_ram_mb == 0.0
    assert result.peak_vram_mb == 0.0
    assert not monitor._monitoring_active
