# tests/test_analyze_results.py
"""
Unit tests for the analyze_results.py script.
Focuses on argument parsing and interaction with ResultsAnalyzer.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import the main function or the module to test its behavior
# We will patch the dependencies like argparse and the ResultsAnalyzer class
import analyze_results  # Import the script module


# Patch the dependencies used directly by the script's main function
@patch("analyze_results.argparse.ArgumentParser")
@patch("analyze_results.ResultsAnalyzer")  # Patch the class itself
@patch("analyze_results.Path.mkdir")
@patch("analyze_results.Path.is_file")
@patch("analyze_results.sys.exit")  # To prevent tests from exiting
@patch("analyze_results.log")  # Optional: mock logger to check messages
def test_analyze_script_success(
    mock_log: MagicMock,
    mock_sys_exit: MagicMock,
    mock_is_file: MagicMock,
    mock_mkdir: MagicMock,
    mock_ResultsAnalyzer: MagicMock,  # Mock for the class
    mock_ArgumentParser: MagicMock,
):
    """Test successful execution path of the analyze_results.py script."""
    # --- Setup Mocks ---
    # Mock argparse
    mock_args = MagicMock()
    mock_args.csv_file = Path("path/to/summary.csv")
    mock_args.output_dir = Path("path/to/output")
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_ArgumentParser.return_value = mock_parser

    # Mock Path.is_file to return True
    mock_is_file.return_value = True

    # Mock the ResultsAnalyzer instance returned when the class is called
    mock_analyzer_instance = MagicMock()
    mock_ResultsAnalyzer.return_value = mock_analyzer_instance

    # --- Run the script's main function ---
    analyze_results.main()

    # --- Assertions ---
    # Argparse called correctly
    mock_ArgumentParser.assert_called_once()
    mock_parser.add_argument.assert_any_call(
        "--csv-file", type=Path, required=True, help=pytest.approx(str)
    )
    mock_parser.add_argument(
        "--output-dir", type=Path, required=True, help=pytest.approx(str)
    )
    mock_parser.parse_args.assert_called_once()

    # File/Directory checks and creation
    mock_is_file.assert_called_once_with()  # Called on mock_args.csv_file Path object
    mock_mkdir.assert_called_once_with(
        parents=True, exist_ok=True
    )  # Called on mock_args.output_dir

    # ResultsAnalyzer instantiated correctly
    mock_ResultsAnalyzer.assert_called_once_with(csv_path=mock_args.csv_file)

    # Plotting methods called on the instance
    expected_plot_path_gen = mock_args.output_dir / "generation_time.png"
    expected_plot_path_vram = mock_args.output_dir / "peak_vram.png"
    expected_plot_path_ram = mock_args.output_dir / "peak_ram.png"

    mock_analyzer_instance.plot_generation_time.assert_called_once_with(
        output_path=expected_plot_path_gen
    )
    mock_analyzer_instance.plot_peak_vram.assert_called_once_with(
        output_path=expected_plot_path_vram
    )
    mock_analyzer_instance.plot_peak_ram.assert_called_once_with(
        output_path=expected_plot_path_ram
    )

    # Check for success log message
    mock_log.info.assert_any_call("Analysis complete. Plots saved.")

    # Ensure sys.exit was NOT called
    mock_sys_exit.assert_not_called()


@patch("analyze_results.argparse.ArgumentParser")
@patch("analyze_results.Path.is_file")
@patch("analyze_results.sys.exit")
@patch("analyze_results.log")
def test_analyze_script_csv_not_found(
    mock_log: MagicMock,
    mock_sys_exit: MagicMock,
    mock_is_file: MagicMock,
    mock_ArgumentParser: MagicMock,
):
    """Test script exit when input CSV file is not found."""
    mock_args = MagicMock()
    mock_args.csv_file = Path("non/existent/summary.csv")
    mock_args.output_dir = Path("output")
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_ArgumentParser.return_value = mock_parser

    # Mock Path.is_file to return False
    mock_is_file.return_value = False

    analyze_results.main()

    mock_is_file.assert_called_once()
    mock_log.error.assert_called_with(f"Input CSV file not found: {mock_args.csv_file}")
    mock_sys_exit.assert_called_once_with(1)


@patch("analyze_results.argparse.ArgumentParser")
@patch("analyze_results.Path.is_file")
@patch("analyze_results.Path.mkdir")
@patch("analyze_results.sys.exit")
@patch("analyze_results.log")
def test_analyze_script_mkdir_error(
    mock_log: MagicMock,
    mock_sys_exit: MagicMock,
    mock_mkdir: MagicMock,
    mock_is_file: MagicMock,
    mock_ArgumentParser: MagicMock,
):
    """Test script exit when creating the output directory fails."""
    mock_args = MagicMock()
    mock_args.csv_file = Path("input.csv")
    mock_args.output_dir = Path("cant/create/this")
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_ArgumentParser.return_value = mock_parser

    mock_is_file.return_value = True
    # Mock mkdir to raise OSError
    mock_mkdir.side_effect = OSError("Permission denied")

    analyze_results.main()

    mock_is_file.assert_called_once()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_log.error.assert_called_with(
        f"Could not create output directory {mock_args.output_dir}: Permission denied"
    )
    mock_sys_exit.assert_called_once_with(1)


@patch("analyze_results.argparse.ArgumentParser")
@patch("analyze_results.ResultsAnalyzer")  # Patch the class
@patch("analyze_results.Path.mkdir")
@patch("analyze_results.Path.is_file")
@patch("analyze_results.sys.exit")
@patch("analyze_results.log")
def test_analyze_script_analyzer_init_error(
    mock_log: MagicMock,
    mock_sys_exit: MagicMock,
    mock_is_file: MagicMock,
    mock_mkdir: MagicMock,
    mock_ResultsAnalyzer: MagicMock,  # Mock for the class
    mock_ArgumentParser: MagicMock,
):
    """Test script exit when ResultsAnalyzer initialization fails."""
    mock_args = MagicMock()
    mock_args.csv_file = Path("input.csv")
    mock_args.output_dir = Path("output")
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_ArgumentParser.return_value = mock_parser

    mock_is_file.return_value = True
    # Mock ResultsAnalyzer constructor to raise an error (e.g., ValueError)
    mock_ResultsAnalyzer.side_effect = ValueError("Missing required columns")

    analyze_results.main()

    mock_is_file.assert_called_once()
    mock_mkdir.assert_called_once()
    mock_ResultsAnalyzer.assert_called_once_with(csv_path=mock_args.csv_file)
    mock_log.error.assert_called_with("Analysis failed: Missing required columns")
    mock_sys_exit.assert_called_once_with(1)


@patch("analyze_results.argparse.ArgumentParser")
@patch("analyze_results.ResultsAnalyzer")  # Patch the class
@patch("analyze_results.Path.mkdir")
@patch("analyze_results.Path.is_file")
@patch("analyze_results.sys.exit")
@patch("analyze_results.log")
def test_analyze_script_analyzer_plot_error(
    mock_log: MagicMock,
    mock_sys_exit: MagicMock,
    mock_is_file: MagicMock,
    mock_mkdir: MagicMock,
    mock_ResultsAnalyzer: MagicMock,  # Mock for the class
    mock_ArgumentParser: MagicMock,
):
    """Test script exit when a plotting method on ResultsAnalyzer fails."""
    mock_args = MagicMock()
    mock_args.csv_file = Path("input.csv")
    mock_args.output_dir = Path("output")
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_ArgumentParser.return_value = mock_parser

    mock_is_file.return_value = True
    mock_analyzer_instance = MagicMock()
    # Mock one of the plotting methods to raise an error
    mock_analyzer_instance.plot_generation_time.side_effect = IOError(
        "Cannot save plot"
    )
    mock_ResultsAnalyzer.return_value = mock_analyzer_instance

    analyze_results.main()

    mock_is_file.assert_called_once()
    mock_mkdir.assert_called_once()
    mock_ResultsAnalyzer.assert_called_once_with(csv_path=mock_args.csv_file)
    # Ensure the failing method was called
    mock_analyzer_instance.plot_generation_time.assert_called_once()
    # Ensure subsequent methods might not be called (depending on exact script structure)
    # mock_analyzer_instance.plot_peak_vram.assert_not_called()
    mock_log.error.assert_called_with("Analysis failed: Cannot save plot")
    mock_sys_exit.assert_called_once_with(1)
