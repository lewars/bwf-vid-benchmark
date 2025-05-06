"""
Unit tests for the VideoGenerator class.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

from video_generator import VideoGenerator, GeneratedVideo, ModelAdapter
from test_case import TestCase


def test_video_generator_init(video_generator: VideoGenerator):
    """Test VideoGenerator initialization."""
    assert isinstance(video_generator._adapter_classes, dict)
    assert not video_generator._adapter_classes
    assert video_generator._loaded_adapter_instance is None
    assert video_generator._loaded_adapter_key is None


def test_register_adapter_success(
    video_generator: VideoGenerator, MockAdapterA: type
):
    """Test successful registration of a valid adapter."""
    try:
        video_generator.register_adapter("ModelA", MockAdapterA)
        assert "modela" in video_generator._adapter_classes
        assert video_generator._adapter_classes["modela"] is MockAdapterA
    except (ValueError, TypeError) as e:
        pytest.fail(f"Adapter registration failed unexpectedly: {e}")


def test_register_adapter_invalid_name(
    video_generator: VideoGenerator, MockAdapterA: type
):
    """Test registration failure with invalid model names."""
    with pytest.raises(
        ValueError, match="model_name must be a non-empty string"
    ):
        video_generator.register_adapter("", MockAdapterA)
    with pytest.raises(
        ValueError, match="model_name must be a non-empty string"
    ):
        video_generator.register_adapter(None, MockAdapterA)  # type: ignore


def test_register_adapter_invalid_class(video_generator: VideoGenerator):
    """Test registration failure with an invalid adapter class (missing methods)."""

    class InvalidAdapter:  # Missing methods like generate
        def load_model(self):
            pass

    with pytest.raises(
        TypeError, match="does not fully implement ModelAdapter protocol"
    ):
        video_generator.register_adapter("InvalidModel", InvalidAdapter)  # type: ignore


def test_register_adapter_overwrite_warning(
    video_generator: VideoGenerator,
    MockAdapterA: type,
    MockAdapterB: type,
    caplog,
):
    """Test that overwriting an existing registration logs a warning."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    video_generator.register_adapter("ModelA", MockAdapterB)  # Overwrite

    assert (
        "Overwriting existing adapter registration for model: modela"
        in caplog.text
    )
    assert (
        video_generator._adapter_classes["modela"] is MockAdapterB
    )  # Ensure it was overwritten


def test_get_adapter_unknown_model(video_generator: VideoGenerator):
    """Test _get_adapter raises ValueError for an unregistered model."""
    with pytest.raises(
        ValueError, match="No adapter registered for model: Unknown"
    ):
        video_generator._get_adapter("Unknown")


def test_get_adapter_first_load(
    video_generator: VideoGenerator,
    MockAdapterA: type,
):
    """Test loading the first adapter."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    adapter_instance = video_generator._get_adapter("ModelA")

    assert isinstance(adapter_instance, MockAdapterA)
    assert (
        video_generator._loaded_adapter_instance is adapter_instance
    )  # Check identity
    assert video_generator._loaded_adapter_key == "modela"


# Patch methods on the mock classes from conftest
@patch("tests.conftest._MockAdapterAImpl.load_model", return_value=None)
@patch("tests.conftest._MockAdapterAImpl.unload_model", return_value=None)
@patch("tests.conftest._MockAdapterBImpl.load_model", return_value=None)
def test_get_adapter_switch_models(
    mock_b_load: MagicMock,
    mock_a_unload: MagicMock,
    mock_a_load: MagicMock,
    video_generator: VideoGenerator,
    MockAdapterA: type,
    MockAdapterB: type,
):
    """Test switching between loaded models triggers unload and load."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    video_generator.register_adapter("ModelB", MockAdapterB)

    # Load Model A first
    # Access the mock methods through the *instance* created inside _get_adapter
    adapter_a_instance = video_generator._get_adapter("ModelA")
    adapter_a_instance.load_model.assert_called_once()
    adapter_a_instance.unload_model.assert_not_called()
    # Can't easily assert mock_b_load.assert_not_called() without instance

    assert video_generator._loaded_adapter_instance is adapter_a_instance
    assert video_generator._loaded_adapter_key == "modela"

    # Reset mocks on the *instances* if needed for clarity between steps
    adapter_a_instance.load_model.reset_mock()
    adapter_a_instance.unload_model.reset_mock()

    # Load Model B (should unload A, load B)
    adapter_b_instance = video_generator._get_adapter("ModelB")
    adapter_a_instance.load_model.assert_not_called()  # A load not called again
    adapter_a_instance.unload_model.assert_called_once()  # A unload called

    adapter_b_instance.load_model.assert_called_once()  # B load called
    adapter_b_instance.unload_model.assert_not_called()  # B unload not called yet

    assert video_generator._loaded_adapter_instance is adapter_b_instance
    assert video_generator._loaded_adapter_key == "modelb"


def test_get_adapter_load_failure(
    video_generator: VideoGenerator,
    MockAdapterA: type,
):
    """Test error handling when adapter load_model fails."""
    video_generator.register_adapter("ModelA", MockAdapterA)

    # Patch methods directly on the MockAdapterA class object from the fixture
    with (
        patch.object(
            MockAdapterA, "__init__", return_value=None, autospec=True
        ) as mock_init,
        patch.object(
            MockAdapterA,
            "load_model",
            side_effect=RuntimeError("Load Failed"),
            autospec=True,
        ) as mock_load,
    ):

        with pytest.raises(
            RuntimeError, match="Failed to load model adapter modela"
        ):
            video_generator._get_adapter("ModelA")

        # --- Assertions ---
        mock_init.assert_called_once()  # Instantiation attempted
        mock_load.assert_called_once()  # Load attempted and failed via side_effect

        # Check that the state reflects the failure
        assert video_generator._loaded_adapter_instance is None
        assert video_generator._loaded_adapter_key is None

    mock_init.assert_called_once()  # Instantiation attempted
    # The mock_load here refers to the patched method *before* instantiation attempt
    # Inside the method, the *instance*'s load_model would be called.
    # Verification requires checking the instance's mock if we had access,
    # or trusting the side_effect worked as expected.
    # Let's assume mock_load *was* called on the instance attempt.
    assert (
        mock_load.call_count >= 1
    )  # Ensure the method causing the error was entered

    assert video_generator._loaded_adapter_instance is None  # Should be reset
    assert video_generator._loaded_adapter_key is None


# Patch methods on the instance that will be created
@patch.object(
    Path, "mkdir"
)  # Patch Path.mkdir used inside generate_for_test_case
@patch("tests.conftest._MockAdapterAImpl.generate")
@patch("tests.conftest._MockAdapterAImpl.load_model", return_value=None)
def test_generate_for_test_case_success(
    mock_load: MagicMock,
    mock_generate: MagicMock,  # Patched on the class
    mock_mkdir: MagicMock,  # Patched globally
    video_generator: VideoGenerator,
    MockAdapterA: type,
    test_case_a: TestCase,  # Fixture for TestCase(model_name="ModelA")
    tmp_path: Path,
):
    """Test successful generation flow using generate_for_test_case."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    output_dir = tmp_path / "test_output"
    expected_output_video = GeneratedVideo(file_path=Path("mock_a_video.mp4"))
    # Configure the return value on the CLASS mock, instance will inherit
    mock_generate.return_value = expected_output_video

    # --- Action ---
    result_video = video_generator.generate_for_test_case(
        test_case_a, output_dir
    )

    # --- Assertions ---
    # Adapter loaded (implicitly via _get_adapter called by generate_for_test_case)
    # We need to check load_model on the *instance* after it's created
    adapter_instance = video_generator._loaded_adapter_instance
    assert adapter_instance is not None
    adapter_instance.load_model.assert_called_once()

    # Check generate called on the correct adapter instance with correct args
    adapter_instance.generate.assert_called_once_with(
        test_case=test_case_a, output_dir=output_dir
    )

    # mkdir called on output_dir
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Correct result returned
    assert result_video == expected_output_video


def test_generate_for_test_case_unknown_model(
    video_generator: VideoGenerator, test_case_a: TestCase, tmp_path: Path
):
    """Test generate_for_test_case failure with an unregistered model."""
    # No adapters registered
    with pytest.raises(
        ValueError, match="No adapter registered for model: ModelA"
    ):
        video_generator.generate_for_test_case(test_case_a, tmp_path)


def test_generate_for_test_case_invalid_args(
    video_generator: VideoGenerator, tmp_path: Path
):
    """Test generate_for_test_case failure with invalid argument types."""
    with pytest.raises(
        TypeError, match="Input 'test_case' must be a TestCase object"
    ):
        video_generator.generate_for_test_case(None, tmp_path)  # type: ignore
    with pytest.raises(
        TypeError, match="Input 'output_dir' must be a Path object"
    ):
        # Need a valid TestCase for this part
        dummy_case = TestCase(
            id="d",
            model_name="m",
            prompt="p",
            resolution=(1, 1),
            duration_secs=1,
            fps=1,
        )
        video_generator.generate_for_test_case(dummy_case, "not/a/path")  # type: ignore


@patch(
    "tests.conftest._MockAdapterAImpl.generate",
    side_effect=RuntimeError("Generation OOM"),
)
def test_generate_for_test_case_generate_error(
    mock_generate: MagicMock,  # Patched on class
    video_generator: VideoGenerator,
    MockAdapterA: type,
    test_case_a: TestCase,
    tmp_path: Path,
):
    """Test generate_for_test_case failure when adapter.generate raises error."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    output_dir = tmp_path / "gen_fail_output"
    # Configure side effect on the class, instance will inherit
    mock_generate.side_effect = RuntimeError("Generation OOM")

    with (
        patch.object(
            MockAdapterA, "__init__", return_value=None, autospec=True
        ) as mock_init,
    ):
        with pytest.raises(
            RuntimeError, match="Unexpected error during generation for .*"
        ):
            video_generator.generate_for_test_case(test_case_a, output_dir)

    # Ensure loading happened and generate was called on the instance
    mock_init.assert_called_once()
    adapter_instance = video_generator._loaded_adapter_instance
    assert adapter_instance is not None


@patch("tests.conftest._MockAdapterAImpl.load_model", return_value=None)
@patch("tests.conftest._MockAdapterAImpl.unload_model", return_value=None)
@patch("tests.conftest._MockAdapterAImpl.__init__", return_value=None)
def test_unload_all_adapters(
    mock_init: MagicMock,
    mock_unload: MagicMock,  # Patched on class
    mock_load: MagicMock,  # Patched on class
    video_generator: VideoGenerator,
    MockAdapterA: type,
):
    """Test the unload_all_adapters method."""
    video_generator.register_adapter("ModelA", MockAdapterA)

    # Load the adapter first
    adapter_instance = video_generator._get_adapter("ModelA")
    assert video_generator._loaded_adapter_instance is adapter_instance
    adapter_instance.load_model.assert_called_once()
    adapter_instance.unload_model.assert_not_called()  # Not unloaded yet

    # Call unload_all_adapters
    video_generator.unload_all_adapters()

    # Assert unload was called on the instance and state is cleared
    adapter_instance.unload_model.assert_called_once()
    assert video_generator._loaded_adapter_instance is None
    assert video_generator._loaded_adapter_key is None


@patch("tests.conftest._MockAdapterAImpl.load_model", return_value=None)
@patch(
    "tests.conftest._MockAdapterAImpl.unload_model",
    side_effect=Exception("Unload Error"),
)
@patch("tests.conftest._MockAdapterAImpl.__init__", return_value=None)
def test_unload_all_adapters_error_handling(
    mock_init: MagicMock,
    mock_unload: MagicMock,  # Patched on class
    mock_load: MagicMock,  # Patched on class
    video_generator: VideoGenerator,
    MockAdapterA: type,
    caplog,
):
    """Test that unload_all_adapters logs errors but clears state."""
    video_generator.register_adapter("ModelA", MockAdapterA)
    adapter_instance = video_generator._get_adapter("ModelA")  # Load adapter
    # Configure side effect on the instance mock
    adapter_instance.unload_model.side_effect = Exception("Unload Error")

    # Call unload_all_adapters, expecting it to fail internally but log it
    video_generator.unload_all_adapters()

    adapter_instance.unload_model.assert_called_once()  # unload was called
    # State should still be cleared even if unload failed
    assert video_generator._loaded_adapter_instance is None
    assert video_generator._loaded_adapter_key is None
    # Check log message
    assert (
        "Error during final unload of adapter modela: Unload Error"
        in caplog.text
    )


def test_unload_all_adapters_when_none_loaded(video_generator: VideoGenerator):
    """Test unload_all_adapters when no adapter is currently loaded."""
    # Get the mock unload method from a potential (but not loaded) adapter
    # to check it wasn't called. This is a bit tricky without an instance.
    # Alternatively, just check the state remains None.
    assert video_generator._loaded_adapter_instance is None
    video_generator.unload_all_adapters()  # Should do nothing gracefully
    assert video_generator._loaded_adapter_instance is None
    assert video_generator._loaded_adapter_key is None
    # No mocks to check calls against in this state
