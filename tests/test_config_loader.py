# tests/test_config_loader.py
"""
Unit tests for the test case loading functionality.
"""

import pytest
import yaml
from pathlib import Path
from typing import List, Dict, Any, Callable

from test_case import TestCase, TestCaseLoadError, load_test_cases_from_yaml

def test_load_valid_yaml(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    valid_test_case_data: Dict[str, Any],
    valid_test_case_data_minimal: Dict[str, Any]
):
    """Test loading a correctly formatted YAML file with multiple test cases."""
    # Prepare YAML content using data from conftest fixtures
    yaml_content = {
        "test_cases": [
            valid_test_case_data,
            valid_test_case_data_minimal
        ]
    }
    # Use the fixture to create the file
    file_path = yaml_file_creator("valid_cases.yaml", yaml_content)

    loaded_cases = load_test_cases_from_yaml(file_path)

    assert len(loaded_cases) == 2
    # Verify first test case (using deep comparison might be fragile, check key attrs)
    assert isinstance(loaded_cases[0], TestCase)
    assert loaded_cases[0].id == valid_test_case_data["id"]
    assert loaded_cases[0].model_name == valid_test_case_data["model_name"]
    assert loaded_cases[0].prompt == valid_test_case_data["prompt"]
    assert loaded_cases[0].resolution == valid_test_case_data["resolution"]
    assert loaded_cases[0].duration_secs == float(valid_test_case_data["duration_secs"])
    assert loaded_cases[0].fps == valid_test_case_data["fps"]
    assert loaded_cases[0].seed == valid_test_case_data["seed"]
    assert loaded_cases[0].extra_params == valid_test_case_data["extra_params"]

    # Verify second test case (minimal)
    assert isinstance(loaded_cases[1], TestCase)
    assert loaded_cases[1].id == valid_test_case_data_minimal["id"]
    assert loaded_cases[1].model_name == valid_test_case_data_minimal["model_name"]
    assert loaded_cases[1].prompt == valid_test_case_data_minimal["prompt"]
    assert loaded_cases[1].resolution == valid_test_case_data_minimal["resolution"]
    assert loaded_cases[1].duration_secs == float(valid_test_case_data_minimal["duration_secs"]) # Check conversion
    assert loaded_cases[1].fps == valid_test_case_data_minimal["fps"]
    assert loaded_cases[1].seed is None # Check default
    assert loaded_cases[1].extra_params == {} # Check default


def test_load_yaml_syntax_error(tmp_path: Path):
    """Test loading a YAML file with syntax errors."""
    # Can't use the fixture here as we need malformed content
    file_path = tmp_path / "invalid_syntax.yaml"
    file_path.write_text("test_cases: [\n  key: value\n") # Malformed YAML

    with pytest.raises(yaml.YAMLError):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_missing_test_cases_key(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path]
):
    """Test loading YAML missing the top-level 'test_cases' key."""
    yaml_content = {"other_key": []}
    file_path = yaml_file_creator("missing_key.yaml", yaml_content)

    with pytest.raises(TestCaseLoadError, match="must contain a top-level 'test_cases' key"):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_test_cases_not_a_list(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path]
):
    """Test loading YAML where 'test_cases' is not a list."""
    yaml_content = {"test_cases": {"not": "a list"}}
    file_path = yaml_file_creator("not_a_list.yaml", yaml_content)

    with pytest.raises(TestCaseLoadError, match="'test_cases' key.*must contain a list"):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_item_not_a_dict(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    valid_test_case_data: Dict[str, Any]
):
    """Test loading YAML where an item in 'test_cases' is not a dictionary."""
    yaml_content = {
        "test_cases": [
            valid_test_case_data,
            "not_a_dictionary" # Invalid item
        ]
    }
    file_path = yaml_file_creator("item_not_dict.yaml", yaml_content)

    with pytest.raises(TestCaseLoadError, match="Item #2.*is not a dictionary"):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_test_case_validation_error(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    valid_test_case_data: Dict[str, Any]
):
    """Test loading YAML where a test case fails TestCase internal validation."""
    invalid_case_data = valid_test_case_data.copy()
    invalid_case_data["fps"] = -10 # Invalid FPS

    yaml_content = {"test_cases": [invalid_case_data]}
    file_path = yaml_file_creator("invalid_data.yaml", yaml_content)

    # Expect TestCaseLoadError wrapping the original ValueError from TestCase.__post_init__
    with pytest.raises(TestCaseLoadError, match="Error validating test case #1.*fps must be a positive integer"):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_test_case_missing_required_key(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    valid_test_case_data: Dict[str, Any]
):
    """Test loading YAML where a test case misses a required key (e.g., prompt)."""
    invalid_case_data = valid_test_case_data.copy()
    del invalid_case_data["prompt"] # Remove required key

    yaml_content = {"test_cases": [invalid_case_data]}
    file_path = yaml_file_creator("missing_req_key.yaml", yaml_content)

    # Expect TestCaseLoadError wrapping the TypeError from TestCase.__init__
    with pytest.raises(TestCaseLoadError, match="Error validating test case #1.*__init__.*missing.*required positional argument: 'prompt'"):
        load_test_cases_from_yaml(file_path)

def test_load_yaml_empty_test_cases_list(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    capsys
):
    """Test loading YAML with an empty 'test_cases' list."""
    yaml_content = {"test_cases": []}
    file_path = yaml_file_creator("empty_list.yaml", yaml_content)

    loaded_cases = load_test_cases_from_yaml(file_path)
    captured = capsys.readouterr() # Capture print statements (warnings)

    assert loaded_cases == []
    assert "Warning: No test case definitions found" in captured.err or \
           "Warning: No test case definitions found" in captured.out # Check stdout/stderr

def test_load_yaml_empty_file(tmp_path: Path, capsys):
    """Test loading an empty YAML file."""
    # Need tmp_path directly here to create an empty file
    file_path = tmp_path / "empty_file.yaml"
    file_path.touch() # Create empty file

    loaded_cases = load_test_cases_from_yaml(file_path)
    captured = capsys.readouterr()

    assert loaded_cases == []
    assert "Warning: Test case file is empty" in captured.err or \
           "Warning: Test case file is empty" in captured.out

def test_load_yaml_file_not_found(tmp_path: Path):
    """Test loading a non-existent YAML file."""
    # Need tmp_path directly here to construct a non-existent path
    non_existent_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_test_cases_from_yaml(non_existent_path)

def test_load_yaml_with_explicit_id_and_uuid_fallback(
    yaml_file_creator: Callable[[str, Dict[str, Any]], Path],
    valid_test_case_data: Dict[str, Any]
):
    """Test loading cases with explicit IDs and checking UUID fallback."""
    case_no_id = valid_test_case_data.copy()
    del case_no_id["id"] # Remove explicit ID to test fallback

    yaml_content = {
        "test_cases": [
            valid_test_case_data, # Has explicit ID "test001"
            case_no_id            # Should get a UUID
        ]
    }
    file_path = yaml_file_creator("ids.yaml", yaml_content)
    loaded_cases = load_test_cases_from_yaml(file_path)

    assert len(loaded_cases) == 2
    assert loaded_cases[0].id == "test001" # Check explicit ID
    assert isinstance(loaded_cases[1].id, str) # Check UUID is a string
    # A more robust check might involve regex for UUID format, but str check is basic
    assert len(loaded_cases[1].id) == 36 # Standard UUID length with hyphens

def test_load_yaml_handles_null_test_cases(
    tmp_path: Path, # Need tmp_path to write raw text
    capsys
):
    """Test loading YAML where test_cases key is null."""
    # Can't use fixture easily for non-dict content
    file_path = tmp_path / "null_test_cases.yaml"
    file_path.write_text("test_cases: null\n")

    loaded_cases = load_test_cases_from_yaml(file_path)
    captured = capsys.readouterr()

    assert loaded_cases == []
    assert "Warning: 'test_cases' key is null" in captured.err or \
           "Warning: 'test_cases' key is null" in captured.out
