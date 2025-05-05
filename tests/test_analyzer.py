# tests/test_analyzer.py
"""
Unit tests for the ResultsAnalyzer class.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typing import List, Dict, Any, Callable

from analyzer import ResultsAnalyzer


# Patch pandas.read_csv for initialization tests
@patch("analyzer.pd.read_csv")
def test_analyzer_init_success(
    mock_read_csv: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test successful initialization, loading, and validation."""
    # Configure the mock DataFrame returned by read_csv
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df

    # Create the dummy file so Path.is_file() passes in the real implementation
    mock_csv_path.touch()

    try:
        analyzer = ResultsAnalyzer(csv_path=mock_csv_path)

        # Assertions
        mock_read_csv.assert_called_once_with(mock_csv_path)
        assert analyzer.csv_path == mock_csv_path
        pd.testing.assert_frame_equal(
            analyzer.data, mock_df
        )  # Verify data loaded

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
        pytest.fail(f"ResultsAnalyzer initialization failed unexpectedly: {e}")


@patch("analyzer.pd.read_csv")
def test_analyzer_init_file_not_found(
    mock_read_csv: MagicMock, mock_csv_path: Path
):
    """Test initialization failure when CSV file does not exist."""
    # Don't create the file mock_csv_path.touch()
    # Mock read_csv to raise FileNotFoundError if called (though __init__ should check first)
    mock_read_csv.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        ResultsAnalyzer(csv_path=mock_csv_path)
    mock_read_csv.assert_not_called()  # read_csv shouldn't be called if file check fails


@patch("analyzer.pd.read_csv")
def test_analyzer_init_empty_csv(mock_read_csv: MagicMock, mock_csv_path: Path):
    """Test initialization failure with an empty CSV file."""
    mock_df = pd.DataFrame([])  # Empty DataFrame
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()

    with pytest.raises(ValueError, match="CSV file is empty"):
        ResultsAnalyzer(csv_path=mock_csv_path)
    mock_read_csv.assert_called_once_with(mock_csv_path)


@patch("analyzer.pd.read_csv")
def test_analyzer_init_missing_columns(
    mock_read_csv: MagicMock, mock_csv_path: Path
):
    """Test initialization failure when required columns are missing."""
    # Create DataFrame missing 'peak_vram_mb'
    faulty_data = [
        {
            "test_case_id": "t1",
            "status": "completed",
            "generation_time_secs": 10,
            "peak_ram_mb": 100,
        }
    ]
    mock_df = pd.DataFrame(faulty_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()

    with pytest.raises(ValueError, match="Missing required columns"):
        ResultsAnalyzer(csv_path=mock_csv_path)
    mock_read_csv.assert_called_once_with(mock_csv_path)


# --- Test Plotting Methods ---
# We need to patch plotting libraries (seaborn, matplotlib.pyplot)


@patch("analyzer.plt.close")
@patch("analyzer.plt.savefig")
@patch("analyzer.plt.tight_layout")
@patch("analyzer.plt.xticks")
@patch("analyzer.plt.ylabel")
@patch("analyzer.plt.xlabel")
@patch("analyzer.plt.title")
@patch("analyzer.sns.barplot")
@patch("analyzer.pd.read_csv")  # Also mock read_csv for setup
def test_plot_metric_barplot(
    mock_read_csv: MagicMock,
    mock_barplot: MagicMock,
    mock_title: MagicMock,
    mock_xlabel: MagicMock,
    mock_ylabel: MagicMock,
    mock_xticks: MagicMock,
    mock_tight_layout: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test the generic plot_metric_barplot method."""
    # Setup Analyzer instance with mocked data
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()
    analyzer = ResultsAnalyzer(
        csv_path=mock_csv_path
    )  # Init loads the mock data

    # --- Call the method under test ---
    metric = "generation_time_secs"
    ylabel = "Time (s)"
    title = "Generation Time per Test Case"
    output_file = mock_csv_path.parent / "gen_time_plot.png"

    analyzer.plot_metric_barplot(
        metric_col=metric,
        y_label=ylabel,
        title=title,
        output_path=output_file,
    )

    # --- Assertions ---
    # Check that barplot was called with correct data and axes
    mock_barplot.assert_called_once()
    call_args, call_kwargs = mock_barplot.call_args
    pd.testing.assert_frame_equal(call_kwargs["data"], analyzer.data)
    assert call_kwargs["x"] == "test_case_id"
    assert call_kwargs["y"] == metric

    # Check that plot customization functions were called
    mock_title.assert_called_once_with(title)
    mock_xlabel.assert_called_once_with(
        "Test Case ID"
    )  # Assuming default xlabel
    mock_ylabel.assert_called_once_with(ylabel)
    mock_xticks.assert_called_once()  # Check rotation/alignment if implemented
    mock_tight_layout.assert_called_once()

    # Check that plot was saved and closed
    mock_savefig.assert_called_once_with(output_file)
    mock_close.assert_called_once()


@patch(
    "analyzer.ResultsAnalyzer.plot_metric_barplot"
)  # Patch the method within the class
@patch("analyzer.pd.read_csv")  # Mock read_csv for setup
def test_plot_generation_time(
    mock_read_csv: MagicMock,
    mock_plot_metric: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test the specific plot_generation_time method calls plot_metric_barplot."""
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()
    analyzer = ResultsAnalyzer(csv_path=mock_csv_path)

    output_file = mock_csv_path.parent / "gen_time.png"
    analyzer.plot_generation_time(output_path=output_file)

    # Assert plot_metric_barplot was called with the correct arguments
    mock_plot_metric.assert_called_once_with(
        metric_col="generation_time_secs",
        y_label="Generation Time (s)",
        title="Video Generation Time per Test Case",
        output_path=output_file,
    )


@patch("analyzer.ResultsAnalyzer.plot_metric_barplot")
@patch("analyzer.pd.read_csv")
def test_plot_peak_vram(
    mock_read_csv: MagicMock,
    mock_plot_metric: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test the specific plot_peak_vram method calls plot_metric_barplot."""
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()
    analyzer = ResultsAnalyzer(csv_path=mock_csv_path)

    output_file = mock_csv_path.parent / "vram.png"
    analyzer.plot_peak_vram(output_path=output_file)

    mock_plot_metric.assert_called_once_with(
        metric_col="peak_vram_mb",
        y_label="Peak VRAM (MB)",
        title="Peak VRAM Usage per Test Case",
        output_path=output_file,
    )


@patch("analyzer.ResultsAnalyzer.plot_metric_barplot")
@patch("analyzer.pd.read_csv")
def test_plot_peak_ram(
    mock_read_csv: MagicMock,
    mock_plot_metric: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test the specific plot_peak_ram method calls plot_metric_barplot."""
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()
    analyzer = ResultsAnalyzer(csv_path=mock_csv_path)

    output_file = mock_csv_path.parent / "ram.png"
    analyzer.plot_peak_ram(output_path=output_file)

    mock_plot_metric.assert_called_once_with(
        metric_col="peak_ram_mb",
        y_label="Peak System RAM (MB)",
        title="Peak System RAM Usage per Test Case",
        output_path=output_file,
    )


@patch("analyzer.plt.savefig")
@patch("analyzer.sns.barplot")
@patch("analyzer.pd.read_csv")
def test_plot_metric_barplot_save_error(
    mock_read_csv: MagicMock,
    mock_barplot: MagicMock,
    mock_savefig: MagicMock,
    mock_csv_path: Path,
    sample_summary_data: List[Dict[str, Any]],
):
    """Test error handling when saving the plot fails."""
    mock_df = pd.DataFrame(sample_summary_data)
    mock_read_csv.return_value = mock_df
    mock_csv_path.touch()
    analyzer = ResultsAnalyzer(csv_path=mock_csv_path)

    # Configure savefig to raise an error
    mock_savefig.side_effect = IOError("Permission denied")

    output_file = mock_csv_path.parent / "plot_fail.png"
    with pytest.raises(IOError, match="Failed to save plot"):
        analyzer.plot_metric_barplot(
            metric_col="peak_vram_mb",
            y_label="VRAM",
            title="VRAM Plot Fail",
            output_path=output_file,
        )
    mock_savefig.assert_called_once_with(
        output_file
    )  # Ensure savefig was attempted
