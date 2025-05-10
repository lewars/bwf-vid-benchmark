import os
import threading
import logging
from typing import Optional

import psutil
import pynvml

from metrics import PeakResourceUsage

log = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitors peak system RAM (RSS) and NVIDIA GPU VRAM usage of the current
    process in a background thread. Provides a context manager interface.
    """

    def __init__(self, poll_interval: float = 0.1, gpu_index: int = 0):
        """
        Initializes the resource monitor.

        Args:
            poll_interval: Time interval (in seconds) between resource checks.
            gpu_index: The index of the NVIDIA GPU to monitor.
        """

        self.poll_interval: float = poll_interval
        self.gpu_index: int = gpu_index
        self.pid: int = os.getpid()
        self._process_handle: Optional[psutil.Process] = psutil.Process(
            self.pid
        )

        self._peak_ram_bytes: int = 0
        self._peak_vram_bytes: int = 0
        self._monitoring_active: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

        self._gpu_monitoring_enabled: bool = False
        self._nvml_handle = None  # For pynvml device handle

        self._initialize_pynvml(self.gpu_index)
        log.debug(
            f"ResourceMonitor initialized for PID {self.pid}. "
            f"GPU monitoring {'enabled' if self._gpu_monitoring_enabled else 'disabled'}."
        )

    def _initialize_pynvml(self, gpu_index: int) -> None:
        """
        Attempts to initialize PyNVIDIA Management Library (pynvml)
        and get a handle for the specified GPU.
        Sets `self._gpu_monitoring_enabled` and `self._nvml_handle`.
        """

        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._gpu_monitoring_enabled = True
            log.info(f"Successfully initialized pynvml for GPU {gpu_index}.")
        except pynvml.NVMLError as e:
            log.error(
                f"Failed to initialize pynvml or get GPU handle for GPU {gpu_index}: {e}. "
                "GPU VRAM monitoring will be disabled."
            )
            self._gpu_monitoring_enabled = False
            self._nvml_handle = None
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as shutdown_err:
                log.error(
                    f"Error during pynvml shutdown after initialization failure: {shutdown_err}"
                )

    def _monitor_loop(self) -> None:
        """
        The main monitoring loop that runs in a background thread.
        Continuously checks RAM and VRAM usage until the stop event is set.
        """
        log.debug(f"Resource monitor thread started for PID {self.pid}.")
        if self._process_handle is None:
            log.error(
                "Process handle is None in monitor loop. Cannot monitor RAM."
            )
            return

        while not self._stop_event.is_set():
            # Monitor System RAM (RSS)
            try:
                ram_info_bytes = self._process_handle.memory_info().rss
                self._peak_ram_bytes = max(self._peak_ram_bytes, ram_info_bytes)
            except psutil.NoSuchProcess:
                log.warning(
                    f"Process {self.pid} not found. Stopping RAM monitoring."
                )
                break
            except psutil.AccessDenied:
                log.warning(
                    f"Access denied to process {self.pid} info. Stopping RAM monitoring."
                )
                break
            except Exception as e:
                log.exception(
                    f"Error during RAM monitoring for PID {self.pid}: {e}"
                )

            # Monitor GPU VRAM
            if self._gpu_monitoring_enabled and self._nvml_handle:
                try:
                    # nvmlDeviceGetMemoryInfo returns a struct with total, free, used
                    vram_info = pynvml.nvmlDeviceGetMemoryInfo(
                        self._nvml_handle
                    )
                    self._peak_vram_bytes = max(
                        self._peak_vram_bytes, vram_info.used
                    )
                except pynvml.NVMLError as e:
                    log.error(
                        f"pynvml error during VRAM monitoring for GPU {self.gpu_index}: {e}. Disabling GPU monitoring for this run."
                    )
                    # Disable further GPU monitoring attempts in this loop to avoid repeated errors
                    self._gpu_monitoring_enabled = False
                except Exception as e:
                    log.exception(
                        f"Unexpected error during VRAM monitoring: {e}"
                    )

            # Wait for the poll interval or until stop event is set
            self._stop_event.wait(self.poll_interval)
        log.debug("Resource monitor thread finished.")

    def start(self) -> None:
        """
        Starts the background monitoring thread.
        Resets peak values before starting.
        """
        if self._monitoring_active:
            log.warning(
                "Monitoring is already active. Call stop() before starting again."
            )
            return

        if not self._process_handle:
            log.error(
                "Cannot start monitoring: psutil process handle not initialized."
            )
            return

        self._peak_ram_bytes = 0
        self._peak_vram_bytes = 0
        self._stop_event.clear()  # Reset the event for a new monitoring session

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitoring_active = True
        self._monitor_thread.start()
        log.info(
            f"Resource monitoring started. Polling every {self.poll_interval}s."
        )

    def stop(self) -> PeakResourceUsage:
        """
        Stops the monitoring thread and returns the peak usage recorded.

        Returns:
            A PeakResourceUsage object containing the peak RAM and VRAM (in MB).
            Returns (0.0, 0.0) if monitoring was not active or failed.
        """
        if not self._monitoring_active or self._monitor_thread is None:
            log.warning(
                "Monitoring was not active or thread not initialized. Returning zero peak usage."
            )
            return PeakResourceUsage(peak_vram_mb=0.0, peak_ram_mb=0.0)

        self._stop_event.set()
        self._monitor_thread.join()
        self._monitoring_active = False

        peak_ram_mb = self._peak_ram_bytes / (1024 * 1024)
        peak_vram_mb = self._peak_vram_bytes / (1024 * 1024)

        log.info(
            f"Resource monitoring stopped. Peak RAM: {peak_ram_mb:.2f} MB, "
            f"Peak VRAM: {peak_vram_mb:.2f} MB (GPU {self.gpu_index if self._nvml_handle else 'N/A'})."
        )

        # NVML cleanup: Shutdown pynvml if it was initialized by this instance
        # This is important to release NVML resources.
        if self._nvml_handle:  # Check pynvml again as it could be None
            try:
                pynvml.nvmlShutdown()
                log.debug("pynvml shutdown successful.")
            except pynvml.NVMLError as e:
                log.error(f"Error during pynvml shutdown: {e}")
            finally:
                self._nvml_handle = None
                self._gpu_monitoring_enabled = False

        return PeakResourceUsage(
            peak_vram_mb=peak_vram_mb, peak_ram_mb=peak_ram_mb
        )

    def __enter__(self):
        """Allows using the monitor as a context manager. Starts monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops monitoring when exiting the context. Returns no value."""
        self.stop()
        # Exceptions are not suppressed by returning True or False
