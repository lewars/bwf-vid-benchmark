"""
Defines the TestCase data structure and the function to load test cases from YAML.
"""  # noqa: E501

import yaml
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)


class TestCaseLoadError(Exception):
    """Custom exception for errors during test case loading or validation."""

    pass


@dataclass(frozen=True)
class TestCase:
    """
    Immutable data structure representing a single benchmark test case configuration.

    Attributes:
        id: Unique identifier for the test case (string, auto-generated if not provided).
        model_name: Name of the AI model to use (string).
        prompt: Text prompt for video generation (string).
        resolution: Output video resolution (width, height) in pixels (tuple of two positive ints).
        duration_secs: Desired video duration in seconds (positive float).
        fps: Frames per second for the output video (positive integer).
        seed: Optional random seed for reproducibility (integer or None).
        extra_params: Optional dictionary for model-specific parameters (dict or None, defaults to {}).
    """  # noqa: E501

    id: str
    model_name: str
    prompt: str
    resolution: Tuple[int, int]
    duration_secs: float
    fps: int
    seed: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Performs validation checks after standard dataclass initialization."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string.")

        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string.")

        if not isinstance(self.resolution, tuple) or len(self.resolution) != 2:
            raise TypeError(
                "resolution must be a tuple of two elements (width, height)."
            )
        if not all(isinstance(dim, int) and dim > 0 for dim in self.resolution):
            raise ValueError(
                "resolution width and height must be positive integers."
            )

        if (
            not isinstance(self.duration_secs, (int, float))
            or self.duration_secs <= 0
        ):
            raise ValueError(
                "duration_secs must be a positive number (int or float)."
            )
        # Ensure duration_secs is stored as float internally, even if input was int
        if isinstance(self.duration_secs, int):
            object.__setattr__(self, "duration_secs", float(self.duration_secs))

        if not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError("fps must be a positive integer.")

        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an integer or None.")

        if self.extra_params is None:
            object.__setattr__(self, "extra_params", {})

        if not isinstance(self.extra_params, dict):
            raise TypeError("extra_params must be a dictionary.")


def load_test_cases_from_yaml(file_path: Path) -> List[TestCase]:
    """
    Loads, validates, and instantiates TestCase objects from a YAML configuration file.

    Args:
        file_path: The path to the YAML file containing test case definitions.

    Returns:
        A list of validated TestCase objects.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the YAML file has syntax errors.
        TestCaseLoadError: If the YAML structure is invalid, a definition fails
                           validation, or other loading issues occur.
    """  # noqa: E501
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Test case configuration file not found: {file_path}"
        )

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}") from e
    except Exception as e:
        raise IOError(f"Could not read file {file_path}: {e}") from e

    if data is None:
        log.warning(f"Warning: Test case file is empty: {file_path}")
        return []

    if not isinstance(data, dict):
        raise TestCaseLoadError(
            f"YAML content in {file_path} must be a dictionary (mapping)."
        )

    test_case_definitions = data.get("test_cases")

    if test_case_definitions is None:
        # Handle case where 'test_cases' key exists but is null
        if "test_cases" in data:
            log.warning(
                f"Warning: 'test_cases' key is null in {file_path}. No test cases loaded."
            )
            return []
        else:
            raise TestCaseLoadError(
                f"YAML file {file_path} must contain a top-level 'test_cases' key."
            )

    if not isinstance(test_case_definitions, list):
        raise TestCaseLoadError(
            f"The 'test_cases' key in {file_path} must contain a list of test case definitions."
        )

    if not test_case_definitions:
        log.warning(
            f"Warning: No test case definitions found under 'test_cases' key in {file_path}."
        )
        return []

    loaded_cases: List[TestCase] = []
    for i, item in enumerate(test_case_definitions):
        if not isinstance(item, dict):
            raise TestCaseLoadError(
                f"Item #{i+1} under 'test_cases' in {file_path} is not a dictionary."
            )

        # Prepare arguments for TestCase instantiation
        case_args = item.copy()  # Work on a copy

        # Handle ID: Use provided or generate UUID
        if "id" not in case_args or case_args["id"] is None:
            case_args["id"] = str(uuid.uuid4())
        elif not isinstance(case_args["id"], str) or not case_args["id"]:
            raise TestCaseLoadError(
                f"Item #{i+1} has an invalid 'id' (must be a non-empty string): {case_args['id']}"
            )

        # Handle extra_params: Use provided or default to empty dict
        if "extra_params" not in case_args or case_args["extra_params"] is None:
            case_args["extra_params"] = {}
        elif not isinstance(case_args["extra_params"], dict):
            raise TestCaseLoadError(
                f"Item #{i+1} has invalid 'extra_params' (must be a dictionary or null): {case_args['extra_params']}"
            )

        try:
            # Instantiate and validate the TestCase
            test_case = TestCase(**case_args)
            loaded_cases.append(test_case)
        except (TypeError, ValueError) as e:
            # Catch errors from TestCase __init__ (missing args) or
            raise TestCaseLoadError(
                f"Error validating test case #{i+1} (ID: {case_args.get('id', 'N/A')}) in {file_path}: {e}"
            ) from e
        except Exception as e:
            # Catch unexpected errors during instantiation
            raise TestCaseLoadError(
                f"Unexpected error creating test case #{i+1} (ID: {case_args.get('id', 'N/A')}) in {file_path}: {e}"
            ) from e

    log.info(
        f"Successfully loaded {len(loaded_cases)} test cases from {file_path}."
    )
    return loaded_cases
