# tests/test_test_case.py
"""
Unit tests for the TestCase data class.
"""

import pytest
from dataclasses import (
    FrozenInstanceError,
    is_dataclass,
)  # Keep necessary imports
from src.test_case import TestCase


def test_testcase_is_dataclass():
    """Verify that TestCase is correctly defined as a dataclass."""
    assert is_dataclass(TestCase)


def test_testcase_successful_initialization(valid_test_case_data):
    """Test successful creation of a TestCase instance with valid data."""
    try:
        # Use the fixture data provided by pytest
        case = TestCase(**valid_test_case_data)
        assert case.id == valid_test_case_data["id"]
        assert case.model_name == valid_test_case_data["model_name"]
        assert case.prompt == valid_test_case_data["prompt"]
        assert case.resolution == valid_test_case_data["resolution"]
        assert case.duration_secs == valid_test_case_data["duration_secs"]
        assert case.fps == valid_test_case_data["fps"]
        assert case.seed == valid_test_case_data["seed"]
        assert case.extra_params == valid_test_case_data["extra_params"]
    except (ValueError, TypeError) as e:
        pytest.fail(f"Valid TestCase initialization failed: {e}")


def test_testcase_successful_initialization_minimal(
    valid_test_case_data_minimal,
):
    """Test successful creation with minimal valid data (using defaults)."""
    try:
        # Use the minimal fixture data
        case = TestCase(**valid_test_case_data_minimal)
        assert case.id == valid_test_case_data_minimal["id"]
        assert case.model_name == valid_test_case_data_minimal["model_name"]
        assert case.prompt == valid_test_case_data_minimal["prompt"]
        assert case.resolution == valid_test_case_data_minimal["resolution"]
        # Assuming __post_init__ converts int duration to float if needed
        assert isinstance(case.duration_secs, float)
        assert case.duration_secs == 10.0
        assert case.fps == valid_test_case_data_minimal["fps"]
        assert case.seed is None  # Default
        assert case.extra_params == {}  # Default (empty dict)
    except (ValueError, TypeError) as e:
        pytest.fail(f"Minimal valid TestCase initialization failed: {e}")


def test_testcase_immutability(valid_test_case_data):
    """Test that TestCase attributes cannot be modified after creation."""
    case = TestCase(**valid_test_case_data)
    with pytest.raises(FrozenInstanceError):
        case.prompt = "A dog riding a bicycle"
    with pytest.raises(FrozenInstanceError):
        case.resolution = (256, 256)


def test_testcase_validation_invalid_id(valid_test_case_data):
    """Test validation failure for invalid id."""
    invalid_data = valid_test_case_data.copy()
    invalid_data["id"] = ""
    with pytest.raises(ValueError, match="id.*non-empty string"):
        TestCase(**invalid_data)

    invalid_data["id"] = None
    with pytest.raises(ValueError, match="id.*non-empty string"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_model_name(valid_test_case_data):
    """Test validation failure for invalid model_name."""
    invalid_data = valid_test_case_data.copy()
    invalid_data["model_name"] = ""
    with pytest.raises(ValueError, match="model_name.*non-empty string"):
        TestCase(**invalid_data)

    invalid_data["model_name"] = None
    with pytest.raises(ValueError, match="model_name.*non-empty string"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_prompt(valid_test_case_data):
    """Test validation failure for invalid prompt."""
    invalid_data = valid_test_case_data.copy()
    invalid_data["prompt"] = ""
    with pytest.raises(ValueError, match="prompt.*non-empty string"):
        TestCase(**invalid_data)

    invalid_data["prompt"] = None
    with pytest.raises(ValueError, match="prompt.*non-empty string"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_resolution(valid_test_case_data):
    """Test validation failure for invalid resolution."""
    invalid_data = valid_test_case_data.copy()

    # Incorrect type
    invalid_data["resolution"] = [512, 512]
    with pytest.raises(TypeError, match="resolution.*tuple"):
        TestCase(**invalid_data)

    # Incorrect number of elements
    invalid_data["resolution"] = (512,)
    with pytest.raises(TypeError, match="resolution.*tuple of two"):
        TestCase(**invalid_data)

    # Non-integer elements
    invalid_data["resolution"] = (512.0, 512)
    with pytest.raises(ValueError, match="resolution.*positive integers"):
        TestCase(**invalid_data)

    # Non-positive elements
    invalid_data["resolution"] = (0, 512)
    with pytest.raises(ValueError, match="resolution.*positive integers"):
        TestCase(**invalid_data)
    invalid_data["resolution"] = (512, -10)
    with pytest.raises(ValueError, match="resolution.*positive integers"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_duration(valid_test_case_data):
    """Test validation failure for invalid duration_secs."""
    invalid_data = valid_test_case_data.copy()

    # Non-positive value
    invalid_data["duration_secs"] = 0
    with pytest.raises(ValueError, match="duration_secs.*positive number"):
        TestCase(**invalid_data)
    invalid_data["duration_secs"] = -5.0
    with pytest.raises(ValueError, match="duration_secs.*positive number"):
        TestCase(**invalid_data)

    # Incorrect type
    invalid_data["duration_secs"] = "5 seconds"
    with pytest.raises(
        (TypeError, ValueError), match="duration_secs.*positive number"
    ):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_fps(valid_test_case_data):
    """Test validation failure for invalid fps."""
    invalid_data = valid_test_case_data.copy()

    # Non-positive value
    invalid_data["fps"] = 0
    with pytest.raises(ValueError, match="fps.*positive integer"):
        TestCase(**invalid_data)
    invalid_data["fps"] = -15
    with pytest.raises(ValueError, match="fps.*positive integer"):
        TestCase(**invalid_data)

    # Incorrect type
    invalid_data["fps"] = 15.5
    with pytest.raises((TypeError, ValueError), match="fps.*positive integer"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_seed(valid_test_case_data):
    """Test validation failure for invalid seed type."""
    invalid_data = valid_test_case_data.copy()
    invalid_data["seed"] = "not a seed"
    with pytest.raises(TypeError, match="seed.*integer or None"):
        TestCase(**invalid_data)


def test_testcase_validation_invalid_extra_params(valid_test_case_data):
    """Test validation failure for invalid extra_params type."""
    invalid_data = valid_test_case_data.copy()
    invalid_data["extra_params"] = ["list", "not", "dict"]
    with pytest.raises(TypeError, match="extra_params.*dictionary."):
        TestCase(**invalid_data)


def test_testcase_extra_params_default(valid_test_case_data_minimal):
    """Test that extra_params defaults to an empty dict if not provided."""
    # Use minimal data which omits extra_params
    case = TestCase(**valid_test_case_data_minimal)
    assert case.extra_params == {}


def test_testcase_extra_params_accepts_none_becomes_empty_dict(
    valid_test_case_data_minimal,
):
    """Test that providing None for extra_params results in an empty dict."""
    data_with_none = valid_test_case_data_minimal.copy()
    data_with_none["extra_params"] = None
    case = TestCase(**data_with_none)
    # This assumes the __post_init__ handles None correctly
    assert case.extra_params == {}
