# tests/test_generated_video.py
"""
Unit tests for the GeneratedVideo data class.
"""

import pytest
from dataclasses import dataclass, field, FrozenInstanceError, is_dataclass

from video_generator import GeneratedVideo

# --- Test Functions ---

def test_generatedvideo_is_dataclass():
    """Verify GeneratedVideo is a dataclass."""
    assert is_dataclass(GeneratedVideo)

def test_generatedvideo_successful_initialization():
    """Test successful creation with a valid file path."""
    valid_path = "/path/to/output/video.mp4"
    try:
        video = GeneratedVideo(file_path=valid_path)
        assert video.file_path == valid_path
    except ValueError as e:
        pytest.fail(f"Valid GeneratedVideo initialization failed: {e}")

def test_generatedvideo_immutability():
    """Test that GeneratedVideo attributes cannot be modified."""
    video = GeneratedVideo(file_path="/path/to/video.mp4")
    with pytest.raises(FrozenInstanceError):
        video.file_path = "/new/path/video.mp4" # type: ignore

def test_generatedvideo_validation_empty_path():
    """Test validation failure for an empty file_path."""
    with pytest.raises(ValueError, match="file_path.*non-empty string"):
        GeneratedVideo(file_path="")

def test_generatedvideo_validation_invalid_type():
    """Test validation failure for non-string file_path."""
    with pytest.raises(ValueError, match="file_path.*non-empty string"):
        GeneratedVideo(file_path=None) # type: ignore
    with pytest.raises(ValueError, match="file_path.*non-empty string"):
        GeneratedVideo(file_path=123) # type: ignore
