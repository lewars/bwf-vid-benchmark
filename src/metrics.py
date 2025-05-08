"""
Defines data structures for holding benchmark metrics, including resource usage.
"""

import shutil
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict, is_dataclass
import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeakResourceUsage:
    """
    Immutable data structure holding peak resource usage observed during an operation.

    Attributes:
        peak_vram_mb: Maximum GPU Video RAM used in Megabytes (MB).
        peak_ram_mb: Maximum system RAM (RSS) used by the process in Megabytes (MB).
    """

    peak_vram_mb: float
    peak_ram_mb: float

    def __post_init__(self):
        """Validates that resource usage values are non-negative numbers."""
        if (
            not isinstance(self.peak_vram_mb, (int, float))
            or self.peak_vram_mb < 0
        ):
            raise ValueError(
                f"peak_vram_mb must be a non-negative number, got {self.peak_vram_mb}"
            )
        if (
            not isinstance(self.peak_ram_mb, (int, float))
            or self.peak_ram_mb < 0
        ):
            raise ValueError(
                f"peak_ram_mb must be a non-negative number, got {self.peak_ram_mb}"
            )
        # Ensure they are stored as floats if they were ints
        if isinstance(self.peak_vram_mb, int):
            object.__setattr__(self, "peak_vram_mb", float(self.peak_vram_mb))
        if isinstance(self.peak_ram_mb, int):
            object.__setattr__(self, "peak_ram_mb", float(self.peak_ram_mb))


@dataclass
class TestMetrics:
    """
    Mutable data structure holding all results and performance data
    for a single benchmark test case run.

    Attributes:
        test_case_id: Unique identifier of the TestCase.
        status: Outcome of the test case (e.g., "completed", "failed", "skipped").
        generation_time_secs: Wall-clock time for video generation in seconds. Optional.
        peak_resources: PeakResourceUsage object for VRAM and system RAM. Optional.
        output_video_path: Path to the generated video file. Optional.
        error_message: Error details if the test case failed. Optional.
        video_file_size_bytes: Size of the generated video file in bytes. Optional.
        avg_frame_ssim: Average Structural Similarity Index Measure between consecutive frames. Optional.
    """

    test_case_id: str
    status: str  # "completed", "failed", "skipped"
    generation_time_secs: Optional[float] = None
    peak_resources: Optional[PeakResourceUsage] = None
    output_video_path: Optional[Path] = None
    error_message: Optional[str] = None
    video_file_size_bytes: Optional[int] = None
    avg_frame_ssim: Optional[float] = None

    def __post_init__(self):
        """Validates the integrity and consistency of TestMetrics attributes."""
        if not isinstance(self.test_case_id, str) or not self.test_case_id:
            raise ValueError(
                "TestMetrics test_case_id must be a non-empty string."
            )

        allowed_statuses = ["completed", "failed", "skipped"]
        if self.status not in allowed_statuses:
            raise ValueError(
                f"TestMetrics status must be one of {allowed_statuses}, got {self.status}"
            )

        if self.status == "failed" and not self.error_message:
            raise ValueError(
                f"TestMetrics for {self.test_case_id} is 'failed' but has no error_message."
            )
        if self.status == "completed" and self.error_message:
            raise ValueError(
                f"TestMetrics for {self.test_case_id} is 'completed' but has an error_message: {self.error_message}"
            )

        if self.generation_time_secs is not None and (
            not isinstance(self.generation_time_secs, (int, float))
            or self.generation_time_secs < 0
        ):
            raise ValueError(
                f"TestMetrics generation_time_secs must be a non-negative number or None, got {self.generation_time_secs}"
            )

        if self.peak_resources is not None and not isinstance(
            self.peak_resources, PeakResourceUsage
        ):
            raise TypeError(
                f"TestMetrics peak_resources must be a PeakResourceUsage instance or None, got {type(self.peak_resources)}"
            )

        if self.output_video_path is not None and not isinstance(
            self.output_video_path, Path
        ):
            raise TypeError(
                f"TestMetrics output_video_path must be a Path instance or None, got {type(self.output_video_path)}"
            )

        # Validation for video_file_size_bytes
        if self.video_file_size_bytes is not None:
            if (
                not isinstance(self.video_file_size_bytes, int)
                or self.video_file_size_bytes < 0
            ):
                raise ValueError(
                    f"TestMetrics video_file_size_bytes must be a non-negative integer or None, got {self.video_file_size_bytes}"
                )
            if self.status != "completed":
                log.warning(
                    f"TestMetrics for {self.test_case_id} has video_file_size_bytes but status is '{self.status}'."
                )

        # Validation for avg_frame_ssim
        if self.avg_frame_ssim is not None:
            if not isinstance(self.avg_frame_ssim, float):
                raise ValueError(
                    f"TestMetrics avg_frame_ssim must be a float or None, got {self.avg_frame_ssim}"
                )
            # SSIM is typically between -1 and 1, but can sometimes be slightly outside due to float precision.
            # A broader check might be more practical unless strict adherence to [-1, 1] is critical.
            if not (
                -1.01 <= self.avg_frame_ssim <= 1.01
            ):  # Allow slight tolerance
                log.warning(
                    f"TestMetrics avg_frame_ssim for {self.test_case_id} is {self.avg_frame_ssim}, which is outside the typical [-1, 1] range."
                )
            if self.status != "completed":
                log.warning(
                    f"TestMetrics for {self.test_case_id} has avg_frame_ssim but status is '{self.status}'."
                )


class MetricsRecorder:
    """
    Handles saving benchmark metrics to disk (JSON details, CSV summary).
    """

    _SUMMARY_CSV_COLUMNS = [
        "test_case_id",
        "status",
        "generation_time_secs",
        "peak_vram_mb",
        "peak_ram_mb",
        "output_video_path",
        "video_file_size_bytes",
        "avg_frame_ssim",
        "error_message",
    ]

    def __init__(self, base_output_dir: Path, run_id: str):
        """
        Initializes the recorder and creates the main directory for this run.
        """
        self.base_output_dir = base_output_dir
        self.run_id = run_id
        self.run_dir_path = self.base_output_dir / f"{self.run_id}_run"
        try:
            self.run_dir_path.mkdir(parents=True, exist_ok=True)
            log.info(
                f"MetricsRecorder: Results will be saved in {self.run_dir_path.resolve()}"
            )
        except OSError as e:
            log.error(
                f"MetricsRecorder: Failed to create run directory {self.run_dir_path.resolve()}: {e}"
            )
            raise IOError(
                f"Failed to create run directory {self.run_dir_path.resolve()}"
            ) from e
        log.debug(f"MetricsRecorder initialized for run: {self.run_id}")

    def record_detailed_metrics(
        self, metrics: TestMetrics, test_case_subdir: str
    ) -> Path:
        """
        Saves the detailed metrics for a single test case to a JSON file.
        """
        if not isinstance(test_case_subdir, str) or not test_case_subdir:
            raise ValueError("test_case_subdir must be a non-empty string.")

        test_case_output_dir = self.run_dir_path / test_case_subdir
        try:
            test_case_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(
                f"Failed to create test case directory {test_case_output_dir.resolve()}: {e}"
            )
            raise IOError(
                f"Failed to create test case directory {test_case_output_dir.resolve()}"
            ) from e

        json_file_path = test_case_output_dir / "metrics.json"
        log.debug(
            f"Recording detailed metrics for {metrics.test_case_id} to {json_file_path.resolve()}"
        )

        try:
            # Use the helper to convert dataclass to dict for JSON
            metrics_dict = _to_dict_for_json(metrics)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=4)
            log.info(
                f"Detailed metrics for {metrics.test_case_id} saved to {json_file_path.resolve()}"
            )
            return json_file_path
        except TypeError as e:
            log.error(
                f"TypeError during JSON serialization for {metrics.test_case_id}: {e}",
                exc_info=True,
            )
            raise TypeError(
                f"Could not serialize TestMetrics to JSON: {e}"
            ) from e
        except IOError as e:
            log.error(
                f"IOError saving detailed metrics for {metrics.test_case_id} to {json_file_path.resolve()}: {e}",
                exc_info=True,
            )
            raise IOError(
                f"Could not write metrics JSON to {json_file_path.resolve()}"
            ) from e

    def compile_summary_csv(self, all_metrics: List[TestMetrics]) -> Path:
        """
        Compiles key metrics from all test cases into a summary CSV file.
        """
        if not all_metrics:
            raise ValueError(
                "Cannot compile summary CSV: 'all_metrics' list is empty."
            )
        if not all(isinstance(m, TestMetrics) for m in all_metrics):
            raise ValueError(
                "Cannot compile summary CSV: 'all_metrics' must be a list of TestMetrics objects."
            )

        summary_csv_path = self.run_dir_path / "summary.csv"
        log.debug(
            f"Compiling summary CSV for {len(all_metrics)} test cases to {summary_csv_path.resolve()}"
        )

        records_for_df = []
        for m in all_metrics:
            record = {
                "test_case_id": m.test_case_id,
                "status": m.status,
                "generation_time_secs": m.generation_time_secs,
                "peak_vram_mb": (
                    m.peak_resources.peak_vram_mb if m.peak_resources else None
                ),
                "peak_ram_mb": (
                    m.peak_resources.peak_ram_mb if m.peak_resources else None
                ),
                "output_video_path": (
                    str(m.output_video_path) if m.output_video_path else None
                ),
                "video_file_size_bytes": m.video_file_size_bytes,
                "avg_frame_ssim": m.avg_frame_ssim,
                "error_message": m.error_message,
            }
            records_for_df.append(record)

        try:
            summary_df = pd.DataFrame(records_for_df)
            # Ensure all defined columns are present, even if some have all NaNs
            # and establish the desired order.
            for col in self._SUMMARY_CSV_COLUMNS:
                if col not in summary_df.columns:
                    summary_df[col] = pd.NA  # Use pandas NA for missing values

            summary_df = summary_df[
                self._SUMMARY_CSV_COLUMNS
            ]  # Reorder/select columns

            summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
            log.info(
                f"Benchmark summary CSV saved to: {summary_csv_path.resolve()}"
            )
            return summary_csv_path
        except Exception as e:  # Catch pandas errors or IOErrors
            log.error(
                f"Failed to compile or save summary CSV to {summary_csv_path.resolve()}: {e}",
                exc_info=True,
            )
            raise IOError(
                f"Could not write summary CSV to {summary_csv_path.resolve()}"
            ) from e

    def cleanup(self):
        """
        Cleans up the run directory and all its contents.
        """
        if self.run_dir_path.exists() and self.run_dir_path.is_dir():
            try:
                shutil.rmtree(self.run_dir_path)
                log.info(
                    f"Run directory {self.run_dir_path.resolve()} cleaned up."
                )
            except OSError as e:
                log.error(
                    f"Failed to clean up run directory {self.run_dir_path.resolve()}: {e}",
                    exc_info=True,
                )
                raise IOError(
                    f"Failed to clean up run directory {self.run_dir_path.resolve()}"
                ) from e


def _to_dict_for_json(obj: Any) -> Any:
    """
    Recursively converts complex Python objects (like dataclasses, Path objects,
    lists, tuples, and dicts containing these) into a structure composed of
    basic types suitable for JSON serialization (dicts, lists, strings, numbers, booleans, None).

    The overall goal is to prepare an object, such as a TestMetrics instance,
    for `json.dump()` by ensuring all its components are JSON-compatible.
    """
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_to_dict_for_json(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _to_dict_for_json(v) for k, v in obj.items()}
    return obj
