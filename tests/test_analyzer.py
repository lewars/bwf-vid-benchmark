# tests/test_analyzer.py
"""
Unit tests for the ResultsAnalyzer class from analyzer.py.
"""
import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Assuming ResultsAnalyzer is in 'analyzer.py' and src is in PYTHONPATH
from analyzer import ResultsAnalyzer

# Define the required columns based on the ResultsAnalyzer implementation
# This should match _REQUIRED_COLUMNS in ResultsAnalyzer
REQUIRED_COLS_FOR_TESTS = [
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


@pytest.fixture
def valid_csv_content() -> str:
    """Provides content for a valid CSV file."""
    lines = [
        ",".join(REQUIRED_COLS_FOR_TESTS),
        "tc001,completed,10.5,1024,2048,results/tc001/video.mp4,50000,0.95,",
        "tc002,failed,5.2,512,1024,results/tc002/video.mp4,25000,0.80,ModelError",
        "tc003,completed,20.0,2048,4096,results/tc003/video.mp4,100000,0.98,",
        "tc004,completed,15.3,1500,3000,results/tc004/video.mp4,75000,NaN,",  # Test with NaN
    ]
    return "\n".join(lines)


@pytest.fixture
def minimal_csv_content() -> str:
    """Provides content for a CSV with only the absolute minimum required columns for plotting."""
    cols = [
        "test_case_id",
        "status",
        "generation_time_secs",
        "peak_vram_mb",
        "peak_ram_mb",
    ]
    lines = [
        ",".join(cols),
        "tc001,completed,10.5,1024,2048",
        "tc002,failed,5.2,512,1024",
    ]
    return "\n".join(lines)


@pytest.fixture
def valid_csv_file(tmp_path: Path, valid_csv_content: str) -> Path:
    """Creates a temporary valid CSV file and returns its path."""
    p = tmp_path / "summary.csv"
    p.write_text(valid_csv_content)
    return p


@pytest.fixture
def minimal_csv_file(tmp_path: Path, minimal_csv_content: str) -> Path:
    """Creates a temporary minimal CSV file."""
    p = tmp_path / "minimal_summary.csv"
    p.write_text(minimal_csv_content)
    return p


class TestResultsAnalyzerInitialization:
    """Tests for ResultsAnalyzer.__init__ and _load_and_validate_data."""

    def test_init_success(self, valid_csv_file: Path):
        """Test successful initialization with a valid CSV file."""
        analyzer = ResultsAnalyzer(csv_path=valid_csv_file)
        assert analyzer.csv_path == valid_csv_file
        assert not analyzer.data.empty
        assert "tc001" in analyzer.data["test_case_id"].values
        # Check if numeric columns are loaded as numeric
        assert pd.api.types.is_numeric_dtype(
            analyzer.data["generation_time_secs"]
        )
        assert pd.api.types.is_numeric_dtype(analyzer.data["peak_vram_mb"])
        assert pd.api.types.is_numeric_dtype(analyzer.data["peak_ram_mb"])

    def test_init_type_error_invalid_path_type(self):
        """Test __init__ raises TypeError if csv_path is not a Path object."""
        with pytest.raises(TypeError, match="csv_path must be a Path object"):
            ResultsAnalyzer(csv_path="not_a_path_object.csv")  # type: ignore

    def test_init_file_not_found(self, tmp_path: Path):
        """Test __init__ raises FileNotFoundError for a non-existent CSV."""
        non_existent_file = tmp_path / "non_existent.csv"
        with pytest.raises(
            FileNotFoundError,
            match=f"CSV file not found: {non_existent_file.resolve()}",
        ):
            ResultsAnalyzer(csv_path=non_existent_file)

    def test_init_empty_csv_file_pandas_emptydataerror(self, tmp_path: Path):
        """Test __init__ raises pd.errors.EmptyDataError for an empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")  # Create an actual empty file
        with pytest.raises(
            pd.errors.EmptyDataError
        ):  # Pandas raises this before our custom check
            ResultsAnalyzer(csv_path=empty_file)

    def test_init_csv_with_only_headers_valueerror(self, tmp_path: Path):
        """Test __init__ raises ValueError if CSV has headers but no data rows."""
        header_only_file = tmp_path / "header_only.csv"
        header_only_file.write_text(",".join(REQUIRED_COLS_FOR_TESTS))
        with pytest.raises(
            ValueError,
            match=f"No data found in CSV file: {header_only_file.resolve()}",
        ):
            ResultsAnalyzer(csv_path=header_only_file)

    def test_init_missing_required_columns(self, tmp_path: Path):
        """Test __init__ raises ValueError if CSV is missing required columns."""
        content = "test_case_id,status\ntc001,completed"
        missing_cols_file = tmp_path / "missing_cols.csv"
        missing_cols_file.write_text(content)
        # Identify which columns from _REQUIRED_COLUMNS are actually missing
        # The error message will list the ones defined in ResultsAnalyzer._REQUIRED_COLUMNS
        # For this test, we assume some core ones like 'generation_time_secs' are missing.
        with pytest.raises(ValueError, match="is missing required columns:"):
            ResultsAnalyzer(csv_path=missing_cols_file)

    def test_init_malformed_csv_parsererror(self, tmp_path: Path):
        """Test __init__ raises pd.errors.ParserError for a malformed CSV."""
        # Example: CSV with inconsistent number of columns per row
        malformed_content = "col1,col2\ndata1,\"data2',data3\ndata4,data5"
        malformed_file = tmp_path / "malformed.csv"
        malformed_file.write_text(malformed_content)
        with pytest.raises(pd.errors.ParserError):
            ResultsAnalyzer(csv_path=malformed_file)

    def test_init_minimal_columns_success(self, minimal_csv_file: Path):
        """Test successful initialization with only the absolute minimum required columns for plotting."""
        # This test relies on ResultsAnalyzer._REQUIRED_COLUMNS being flexible enough
        # or this test using a CSV that satisfies its current definition.
        # The current ResultsAnalyzer._REQUIRED_COLUMNS includes more than just plotting ones.
        # This test is more to show that if _REQUIRED_COLUMNS was smaller, it would pass.
        # For the current implementation, this might fail if _REQUIRED_COLUMNS is strict.
        # Let's adjust the expectation based on the provided ResultsAnalyzer code.
        # The provided ResultsAnalyzer._REQUIRED_COLUMNS includes all columns.
        # So, this test should actually expect a ValueError.
        with pytest.raises(ValueError, match="is missing required columns:"):
            ResultsAnalyzer(csv_path=minimal_csv_file)


class TestResultsAnalyzerPlotting:
    """Tests for plotting methods in ResultsAnalyzer."""

    @pytest.fixture
    def analyzer(self, valid_csv_file: Path) -> ResultsAnalyzer:
        """Fixture to provide an initialized ResultsAnalyzer instance."""
        return ResultsAnalyzer(csv_path=valid_csv_file)

    @patch("analyzer.plt")  # Mock matplotlib.pyplot
    @patch("analyzer.sns")  # Mock seaborn
    def test_plot_metric_barplot_success(
        self,
        mock_sns: MagicMock,
        mock_plt: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
    ):
        """Test successful generation of a metric bar plot."""
        output_plot_path = tmp_path / "plots" / "test_plot.png"
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        analyzer.plot_metric_barplot(
            metric_col="generation_time_secs",
            y_label="Time (s)",
            title="Generation Time",
            output_path=output_plot_path,
            filter_status="completed",
        )

        mock_plt.style.use.assert_called_once_with("seaborn-v0_8-whitegrid")
        mock_plt.subplots.assert_called_once_with(figsize=(12, 7))
        mock_sns.barplot.assert_called_once()
        # Check some sns.barplot arguments (data is a DataFrame, so check its content)
        _, sns_kwargs = mock_sns.barplot.call_args
        assert sns_kwargs["x"] == "test_case_id"
        assert sns_kwargs["y"] == "generation_time_secs"
        assert len(sns_kwargs["data"]) == 3
        assert "tc001" in sns_kwargs["data"]["test_case_id"].values
        assert "tc003" in sns_kwargs["data"]["test_case_id"].values

        mock_ax.set_xlabel.assert_called_once_with("Test Case ID", fontsize=10)
        mock_ax.set_ylabel.assert_called_once_with("Time (s)", fontsize=10)
        mock_ax.set_title.assert_called_once_with(
            "Generation Time", fontsize=12, fontweight="bold"
        )
        mock_plt.xticks.assert_called_once_with(
            rotation=45, ha="right", fontsize=8
        )
        mock_plt.yticks.assert_called_once_with(fontsize=8)
        mock_ax.grid.assert_called_once_with(True, linestyle="--", alpha=0.6)
        mock_plt.tight_layout.assert_called_once()
        assert output_plot_path.parent.exists()  # Check directory creation
        mock_plt.savefig.assert_called_once_with(output_plot_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_metric_barplot_type_error_output_path(
        self, analyzer: ResultsAnalyzer
    ):
        """Test plot_metric_barplot raises TypeError for invalid output_path type."""
        with pytest.raises(
            TypeError, match="output_path must be a Path object"
        ):
            analyzer.plot_metric_barplot("generation_time_secs", "Y", "T", "not_a_path.png")  # type: ignore

    def test_plot_metric_barplot_metric_col_not_found(
        self, analyzer: ResultsAnalyzer, tmp_path: Path
    ):
        """Test plot_metric_barplot raises ValueError if metric_col is not in data."""
        output_plot_path = tmp_path / "test_plot.png"
        with pytest.raises(
            ValueError, match="Metric column 'non_existent_col' not found"
        ):
            analyzer.plot_metric_barplot(
                metric_col="non_existent_col",
                y_label="Y",
                title="T",
                output_path=output_plot_path,
            )

    @patch(
        "analyzer.plt.savefig"
    )  # Only need to mock savefig to check if called
    @patch("analyzer.plt.close")  # And close
    @patch("analyzer.sns.barplot")
    def test_plot_metric_barplot_no_data_after_filter(
        self,
        mock_barplot: MagicMock,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
        caplog,
    ):
        """Test plot_metric_barplot logs warning and doesn't plot if no data after filtering."""
        output_plot_path = tmp_path / "test_plot.png"
        analyzer.plot_metric_barplot(
            metric_col="generation_time_secs",
            y_label="Y",
            title="T",
            output_path=output_plot_path,
            filter_status="non_existent_status",  # This status will result in empty data
        )
        assert (
            "No data available for metric 'generation_time_secs' after filtering by status 'non_existent_status'"
            in caplog.text
        )
        mock_barplot.assert_not_called()
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()  # Figure isn't even created if data is empty

    @patch("analyzer.plt.savefig")
    @patch("analyzer.plt.close")
    @patch("analyzer.sns.barplot")
    def test_plot_metric_barplot_metric_all_nans(
        self,
        mock_barplot: MagicMock,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
        caplog,
    ):
        """Test plot_metric_barplot logs warning if metric column is all NaNs."""
        # Modify data to make a metric all NaNs for 'completed' status
        analyzer.data.loc[
            analyzer.data["status"] == "completed", "peak_ram_mb"
        ] = pd.NA
        output_plot_path = tmp_path / "test_plot_nan.png"

        analyzer.plot_metric_barplot(
            metric_col="peak_ram_mb",
            y_label="RAM (MB)",
            title="Peak RAM",
            output_path=output_plot_path,
            filter_status="completed",
        )
        assert (
            "Metric column 'peak_ram_mb' contains only NaN values after filtering."
            in caplog.text
        )
        mock_barplot.assert_not_called()
        mock_savefig.assert_not_called()

    @patch("analyzer.plt.savefig", side_effect=IOError("Disk full"))
    @patch("analyzer.plt.close")  # Mock close to check it's called in finally
    @patch("analyzer.sns.barplot")  # Mock barplot
    @patch(
        "analyzer.plt.subplots", return_value=(MagicMock(), MagicMock())
    )  # Mock subplots
    def test_plot_metric_barplot_savefig_ioerror(
        self,
        mock_subplots: MagicMock,
        mock_barplot: MagicMock,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
    ):
        """Test plot_metric_barplot raises IOError if plt.savefig fails."""
        output_plot_path = tmp_path / "test_plot.png"
        with pytest.raises(
            IOError,
            match="Could not generate or save plot test_plot.png: Disk full",
        ):
            analyzer.plot_metric_barplot(
                metric_col="generation_time_secs",
                y_label="Y",
                title="T",
                output_path=output_plot_path,
            )
        mock_close.assert_called_once()  # Ensure plt.close is called even on error

    @patch.object(
        ResultsAnalyzer, "plot_metric_barplot"
    )  # Mock the method on the class instance
    def test_plot_generation_time(
        self,
        mock_plot_metric_barplot: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
    ):
        """Test plot_generation_time calls plot_metric_barplot with correct args."""
        output_dir = tmp_path / "plots_gen_time"
        analyzer.plot_generation_time(output_dir=output_dir)
        expected_output_path = output_dir / "generation_time.png"
        mock_plot_metric_barplot.assert_called_once_with(
            metric_col="generation_time_secs",
            y_label="Generation Time (seconds)",
            title="Video Generation Time per Test Case (Completed)",
            output_path=expected_output_path,
            filter_status="completed",
        )

    def test_plot_generation_time_type_error_output_dir(
        self, analyzer: ResultsAnalyzer
    ):
        """Test plot_generation_time raises TypeError for invalid output_dir type."""
        with pytest.raises(TypeError, match="output_dir must be a Path object"):
            analyzer.plot_generation_time(output_dir="not_a_path")  # type: ignore

    @patch.object(ResultsAnalyzer, "plot_metric_barplot")
    def test_plot_peak_vram(
        self,
        mock_plot_metric_barplot: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
    ):
        """Test plot_peak_vram calls plot_metric_barplot with correct args."""
        output_dir = tmp_path / "plots_vram"
        analyzer.plot_peak_vram(output_dir=output_dir)
        expected_output_path = output_dir / "peak_vram.png"
        mock_plot_metric_barplot.assert_called_once_with(
            metric_col="peak_vram_mb",
            y_label="Peak VRAM Usage (MB)",
            title="Peak VRAM Usage per Test Case (Completed)",
            output_path=expected_output_path,
            filter_status="completed",
        )

    @patch.object(ResultsAnalyzer, "plot_metric_barplot")
    def test_plot_peak_ram(
        self,
        mock_plot_metric_barplot: MagicMock,
        analyzer: ResultsAnalyzer,
        tmp_path: Path,
    ):
        """Test plot_peak_ram calls plot_metric_barplot with correct args."""
        output_dir = tmp_path / "plots_ram"
        analyzer.plot_peak_ram(output_dir=output_dir)
        expected_output_path = output_dir / "peak_ram.png"
        mock_plot_metric_barplot.assert_called_once_with(
            metric_col="peak_ram_mb",
            y_label="Peak System RAM Usage (MB)",
            title="Peak System RAM Usage per Test Case (Completed)",
            output_path=expected_output_path,
            filter_status="completed",
        )


class TestResultsAnalyzerStatistics:
    """Tests for get_summary_statistics method."""

    @pytest.fixture
    def analyzer(self, valid_csv_file: Path) -> ResultsAnalyzer:
        return ResultsAnalyzer(csv_path=valid_csv_file)

    def test_get_summary_statistics_success(self, analyzer: ResultsAnalyzer):
        """Test successful calculation of summary statistics."""
        stats = analyzer.get_summary_statistics(
            metric_col="generation_time_secs", filter_status="completed"
        )
        assert isinstance(stats, pd.Series)
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(15.2666666)
        assert stats["min"] == pytest.approx(10.5)
        assert stats["max"] == pytest.approx(20.0)

    def test_get_summary_statistics_metric_col_not_found(
        self, analyzer: ResultsAnalyzer
    ):
        """Test raises ValueError if metric_col is not in data."""
        with pytest.raises(
            ValueError, match="Metric column 'non_existent' not found"
        ):
            analyzer.get_summary_statistics(metric_col="non_existent")

    def test_get_summary_statistics_no_data_after_filter(
        self, analyzer: ResultsAnalyzer, caplog
    ):
        """Test returns empty Series if no data after filtering."""
        stats = analyzer.get_summary_statistics(
            metric_col="generation_time_secs", filter_status="unknown_status"
        )
        assert isinstance(stats, pd.Series)
        assert stats.empty
        assert (
            "No valid data for metric 'generation_time_secs' after filtering by status 'unknown_status'"
            in caplog.text
        )

    def test_get_summary_statistics_all_nans(
        self, analyzer: ResultsAnalyzer, caplog
    ):
        """Test returns empty Series if metric column is all NaNs after filtering."""
        # From valid_csv_content, avg_frame_ssim for 'completed' tc004 is NaN.
        # Let's make tc001 and tc003 also NaN for this metric.
        analyzer.data.loc[
            analyzer.data["test_case_id"] == "tc001", "avg_frame_ssim"
        ] = pd.NA
        analyzer.data.loc[
            analyzer.data["test_case_id"] == "tc003", "avg_frame_ssim"
        ] = pd.NA

        stats = analyzer.get_summary_statistics(
            metric_col="avg_frame_ssim", filter_status="completed"
        )
        assert isinstance(stats, pd.Series)
        assert stats.empty  # describe() on all-NaN series returns count 0, etc.
        assert (
            "No valid data for metric 'avg_frame_ssim' after filtering by status 'completed'"
            in caplog.text
        )
