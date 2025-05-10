import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Analyzes benchmark summary data from a CSV file and generates plots.
    """

    # Define required columns for the input CSV file.
    # These columns are essential for the analyzer to function correctly.
    _REQUIRED_COLUMNS = [
        "test_case_id",
        "status",
        "generation_time_secs",
        "peak_vram_mb",
        "peak_ram_mb",
        "output_video_path",  # Not strictly required for plotting these metrics
        "video_file_size_bytes",  # Not strictly required for plotting these metrics
        "avg_frame_ssim",  # Not strictly required for plotting these metrics
        "error_message",  # Useful for filtering/reporting but not for basic plots
    ]

    def __init__(self, csv_path: Path):
        """
        Initializes the analyzer by loading and validating the summary data.

        Args:
            csv_path: Path to the summary CSV file.

        Raises:
            FileNotFoundError: If csv_path does not exist or is not a file.
            ValueError: If CSV is empty, missing required columns, or other data issues.
            pd.errors.ParserError: If CSV parsing fails.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        if not isinstance(csv_path, Path):
            raise TypeError("csv_path must be a Path object.")

        self.csv_path = csv_path
        log.info(
            f"Initializing ResultsAnalyzer with CSV: {self.csv_path.resolve()}"
        )
        self.data: pd.DataFrame = self._load_and_validate_data()
        log.info(
            f"Successfully loaded and validated data from {self.csv_path.resolve()}. Shape: {self.data.shape}"
        )

    def _load_and_validate_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file and performs essential validations.

        Returns:
            A pandas DataFrame containing the loaded and validated data.

        Raises:
            FileNotFoundError: If the CSV file does not exist or is not a file.
            pd.errors.EmptyDataError: If the CSV file is empty.
            pd.errors.ParserError: If there's an error parsing the CSV file.
            ValueError: If the CSV is missing required columns or is empty after loading.
        """
        if not self.csv_path.is_file():
            log.error(f"CSV file not found: {self.csv_path.resolve()}")
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path.resolve()}"
            )

        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            log.error(f"CSV file is empty: {self.csv_path.resolve()}")
            raise  # Re-raise the specific pandas error
        except pd.errors.ParserError as e:
            log.error(
                f"Failed to parse CSV file {self.csv_path.resolve()}: {e}"
            )
            raise  # Re-raise the specific pandas error
        except Exception as e:
            log.error(
                f"An unexpected error occurred while reading CSV {self.csv_path.resolve()}: {e}"
            )
            raise ValueError(f"Could not read CSV file: {e}") from e

        if df.empty:
            log.warning(
                f"CSV file {self.csv_path.resolve()} loaded as an empty DataFrame."
            )
            raise ValueError(
                f"No data found in CSV file: {self.csv_path.resolve()}"
            )

        missing_columns = [
            col for col in self._REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_columns:
            msg = f"CSV file {self.csv_path.resolve()} is missing required columns: {', '.join(missing_columns)}"
            log.error(msg)
            raise ValueError(msg)

        log.debug(f"Data loaded successfully. Columns: {df.columns.tolist()}")
        return df

    def plot_metric_barplot(
        self,
        metric_col: str,
        y_label: str,
        title: str,
        output_path: Path,
        filter_status: str = "completed",
    ):
        """
        Generates and saves a bar plot for a specific metric vs. test_case_id.
        Filters data by status before plotting.

        Args:
            metric_col: The column name in the DataFrame to plot (e.g., 'generation_time_secs').
            y_label: The label for the Y-axis.
            title: The title for the plot.
            output_path: The full path (including filename, e.g., 'plot.png') to save the plot image.
            filter_status: The status to filter test cases by (e.g., "completed").
                           If None, no status filtering is applied.

        Raises:
            ValueError: If metric_col is not found in the data, or if no data remains after filtering.
            IOError: If the plot cannot be saved.
        """
        if not isinstance(output_path, Path):
            raise TypeError("output_path must be a Path object.")
        if metric_col not in self.data.columns:
            msg = f"Metric column '{metric_col}' not found in loaded data."
            log.error(msg)
            raise ValueError(msg)

        plot_data = self.data.copy()
        if filter_status:
            if "status" not in plot_data.columns:
                log.warning(
                    f"Status column not found for filtering by '{filter_status}'. Plotting all data for {metric_col}."
                )
            else:
                plot_data = plot_data[plot_data["status"] == filter_status]

        if plot_data.empty:
            log.warning(
                f"No data available for metric '{metric_col}' after filtering by status '{filter_status}'. Plot not generated."
            )
            return  # Do not generate an empty plot

        # Ensure the metric column is numeric and handle NaNs for plotting
        # Convert to numeric, coercing errors to NaN. This helps if some values are non-numeric.
        plot_data[metric_col] = pd.to_numeric(
            plot_data[metric_col], errors="coerce"
        )
        if plot_data[metric_col].isnull().all():
            log.warning(
                f"Metric column '{metric_col}' contains only NaN values after filtering. Plot not generated for {output_path.name}."
            )
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            sns.barplot(
                x="test_case_id",
                y=metric_col,
                data=plot_data,
                ax=ax,
                palette="viridis",
            )
            ax.set_xlabel("Test Case ID", fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")

            # Rotate x-axis labels for better readability, especially with many test cases
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)

            ax.grid(True, linestyle="--", alpha=0.6)

            plt.tight_layout()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            log.info(
                f"Successfully generated and saved plot: {output_path.resolve()}"
            )

        except Exception as e:
            log.error(
                f"Failed to generate or save plot {output_path.name}: {e}",
                exc_info=True,
            )
            raise IOError(
                f"Could not generate or save plot {output_path.name}: {e}"
            ) from e
        finally:
            plt.close(fig)  # Close the figure to free up memory

    def plot_generation_time(self, output_dir: Path):
        """
        Generates and saves a bar plot for 'generation_time_secs' vs. 'test_case_id'.
        Filters for 'completed' status by default.

        Args:
            output_dir: The directory where the plot 'generation_time.png' will be saved.
        """
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a Path object.")
        output_file = output_dir / "generation_time.png"
        self.plot_metric_barplot(
            metric_col="generation_time_secs",
            y_label="Generation Time (seconds)",
            title="Video Generation Time per Test Case (Completed)",
            output_path=output_file,
            filter_status="completed",
        )

    def plot_peak_vram(self, output_dir: Path):
        """
        Generates and saves a bar plot for 'peak_vram_mb' vs. 'test_case_id'.
        Filters for 'completed' status by default.

        Args:
            output_dir: The directory where the plot 'peak_vram.png' will be saved.
        """
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a Path object.")
        output_file = output_dir / "peak_vram.png"
        self.plot_metric_barplot(
            metric_col="peak_vram_mb",
            y_label="Peak VRAM Usage (MB)",
            title="Peak VRAM Usage per Test Case (Completed)",
            output_path=output_file,
            filter_status="completed",
        )

    def plot_peak_ram(self, output_dir: Path):
        """
        Generates and saves a bar plot for 'peak_ram_mb' vs. 'test_case_id'.
        Filters for 'completed' status by default.

        Args:
            output_dir: The directory where the plot 'peak_ram.png' will be saved.
        """
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a Path object.")
        output_file = output_dir / "peak_ram.png"
        self.plot_metric_barplot(
            metric_col="peak_ram_mb",
            y_label="Peak System RAM Usage (MB)",
            title="Peak System RAM Usage per Test Case (Completed)",
            output_path=output_file,
            filter_status="completed",
        )

    # --- Potentially add more plotting methods or analysis functions below ---
    # Example: A method to get summary statistics
    def get_summary_statistics(
        self, metric_col: str, filter_status: str = "completed"
    ) -> pd.Series:
        """
        Calculates and returns descriptive statistics for a given metric column,
        filtered by status.

        Args:
            metric_col: The column name for which to calculate statistics.
            filter_status: The status to filter by (e.g., "completed").

        Returns:
            A pandas Series containing descriptive statistics (count, mean, std, min, max, etc.).

        Raises:
            ValueError: If metric_col is not found or no data after filtering.
        """
        if metric_col not in self.data.columns:
            msg = f"Metric column '{metric_col}' not found in loaded data."
            log.error(msg)
            raise ValueError(msg)

        plot_data = self.data.copy()
        if filter_status:
            if "status" not in plot_data.columns:
                log.warning(
                    f"Status column not found for filtering by '{filter_status}'. Calculating stats on all data for {metric_col}."
                )
            else:
                plot_data = plot_data[plot_data["status"] == filter_status]

        if plot_data.empty or plot_data[metric_col].isnull().all():
            log.warning(
                f"No valid data for metric '{metric_col}' after filtering by status '{filter_status}'. Cannot compute statistics."
            )
            # Return an empty Series or raise error, depending on desired behavior
            return pd.Series(dtype=float)

        return plot_data[metric_col].describe()
