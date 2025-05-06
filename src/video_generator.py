"""
Defines the video generation components: Generator, Adapters, and data structures.
"""

import logging
import time
from pathlib import Path
from typing import Protocol, Dict, Any, Type, Optional
from dataclasses import dataclass, field
from test_case import TestCase


log = logging.getLogger(__name__)

@dataclass(frozen=True)
class GeneratedVideo:
    """
    Immutable data structure representing a successfully generated video file.

    Attributes:
        file_path: The path to the saved video file.
    """
    file_path: Path

    def __post_init__(self):
        """Basic validation for the file path."""
        if not isinstance(self.file_path, Path):
            raise TypeError("GeneratedVideo file_path must be a Path object.")
        # Note: Existence check is deferred to the consumer.


class ModelAdapter(Protocol):
    """
    Protocol defining the standard interface for video generation model adapters.
    Ensures consistency in how the VideoGenerator interacts with different models.
    """

    def load_model(self) -> None:
        """
        Loads the model into memory (e.g., onto GPU).
        Should handle resource allocation and raise exceptions on failure.
        """
        ...

    def unload_model(self) -> None:
        """
        Unloads the model from memory, freeing resources.
        Optional but recommended for efficient resource management.
        """
        # Default implementation does nothing if unloading isn't needed/supported
        pass

    def generate(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        """
        Generates a video based on the TestCase parameters and saves it.

        Args:
            test_case: The TestCase object containing generation parameters.
            output_dir: The directory where the generated video file should be saved.
                        The adapter is responsible for naming the file within this dir.

        Returns:
            A GeneratedVideo object containing the path to the saved video file.

        Raises:
            Exception: If video generation fails (e.g., model error, OOM, save error).
                       Specific exception types depend on the implementation.
        """
        ...

class Mochi1Adapter:
    def __init__(self):
        self._model = None
        log.info("Mochi1Adapter: Initialized (model not loaded).")

    def load_model(self) -> None:
        log.info("Mochi1Adapter: Loading model...")
        # --- Placeholder for actual Mochi1 model loading ---
        # Example using hypothetical diffusers pipeline:
        # try:
        #     # from diffusers import DiffusionPipeline
        #     # self._pipeline = DiffusionPipeline.from_pretrained("path/to/mochi1_model")
        #     # self._pipeline.to("cuda") # Move to GPU if available
        #     time.sleep(1) # Simulate load time
        #     self._model = "loaded_mochi1_pipeline" # Represent loaded state
        #     log.info("Mochi1Adapter: Model loaded successfully.")
        # except Exception as e:
        #     log.error(f"Mochi1Adapter: Failed to load model - {e}", exc_info=True)
        #     raise RuntimeError("Mochi1 model loading failed") from e
        # --- End Placeholder ---
        time.sleep(1) # Simulate load time
        self._model = "loaded_mochi1_model_placeholder"
        log.info("Mochi1Adapter: Model loaded (placeholder).")


    def unload_model(self) -> None:
        log.info("Mochi1Adapter: Unloading model...")
        # --- Placeholder for actual Mochi1 model unloading ---
        # Example:
        # if self._model:
        #     del self._model
        #     self._model = None
        #     # Attempt to clear GPU memory if using PyTorch/CUDA
        #     try:
        #         import torch
        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #             log.info("Mochi1Adapter: Cleared CUDA cache.")
        #     except ImportError:
        #         pass # Ignore if torch isn't available
        #     except Exception as e:
        #         log.warning(f"Mochi1Adapter: Error during CUDA cache clear - {e}")
        # --- End Placeholder ---
        self._model = None
        log.info("Mochi1Adapter: Model unloaded (placeholder).")

    def generate(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        log.info(f"Mochi1Adapter: Generating video for prompt: '{test_case.prompt}'")
        if self._model is None:
            # This check ensures load_model was called successfully before generate
            raise RuntimeError("Mochi1Adapter: Model not loaded before calling generate.")

        # --- Placeholder for actual Mochi1 video generation ---
        # Example:
        # try:
        #     # Map TestCase parameters to the model's expected arguments
        #     # kwargs = {
        #     #     "prompt": test_case.prompt,
        #     #     "height": test_case.resolution[1],
        #     #     "width": test_case.resolution[0],
        #     #     "num_frames": int(test_case.duration_secs * test_case.fps),
        #     #     "generator": torch.Generator("cuda").manual_seed(test_case.seed) if test_case.seed is not None else None,
        #     #     **test_case.extra_params # Pass through extra params
        #     # }
        #     # video_frames = self._model(**kwargs).frames # Assuming model returns frames
        #
        #     # Save the frames to a video file
        #     output_filename = f"{test_case.id}_mochi1.mp4"
        #     output_file_path = output_dir / output_filename
        #     # from some_video_export_library import export_to_video
        #     # export_to_video(video_frames, str(output_file_path), fps=test_case.fps)
        #
        #     time.sleep(2) # Simulate generation time
        #     output_file_path.touch() # Create dummy file for placeholder
        #
        #     log.info(f"Mochi1Adapter: Video saved to {output_file_path}")
        #     return GeneratedVideo(file_path=output_file_path)
        #
        # except Exception as e:
        #     log.error(f"Mochi1Adapter: Video generation failed - {e}", exc_info=True)
        #     # Re-raise a more specific error or the original one
        #     raise RuntimeError("Mochi1 video generation failed") from e
        # --- End Placeholder ---

        time.sleep(2) # Simulate generation time
        output_filename = f"{test_case.id}_mochi1.mp4"
        output_file_path = output_dir / output_filename
        output_file_path.touch() # Create dummy file for placeholder
        log.info(f"Mochi1Adapter: Video saved to {output_file_path} (placeholder).")
        return GeneratedVideo(file_path=output_file_path)


class HunyuanAdapter:
    def __init__(self):
        self._model = None
        log.info("HunyuanAdapter: Initialized (model not loaded).")

    def load_model(self) -> None:
        log.info("HunyuanAdapter: Loading model...")
        # --- Placeholder for actual Hunyuan model loading ---
        time.sleep(1.5) # Simulate load time
        self._model = "loaded_hunyuan_model_placeholder"
        log.info("HunyuanAdapter: Model loaded (placeholder).")

    def unload_model(self) -> None:
        log.info("HunyuanAdapter: Unloading model...")
        # --- Placeholder for actual Hunyuan model unloading ---
        self._model = None
        log.info("HunyuanAdapter: Model unloaded (placeholder).")

    def generate(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        log.info(f"HunyuanAdapter: Generating video for prompt: '{test_case.prompt}'")
        if self._model is None:
            raise RuntimeError("HunyuanAdapter: Model not loaded before calling generate.")

        # --- Placeholder for actual Hunyuan video generation ---
        time.sleep(3) # Simulate generation time
        output_filename = f"{test_case.id}_hunyuan.mp4"
        output_file_path = output_dir / output_filename
        output_file_path.touch() # Create dummy file for placeholder
        log.info(f"HunyuanAdapter: Video saved to {output_file_path} (placeholder).")
        return GeneratedVideo(file_path=output_file_path)


class VideoGenerator:
    """
    Orchestrates video generation using registered model adapters.
    Manages adapter loading, unloading, and execution based on TestCase.
    """

    def __init__(self):
        """Initializes the VideoGenerator with registries for adapter classes and instances."""
        self._adapter_classes: Dict[str, Type[ModelAdapter]] = {}
        # Stores the currently instantiated and loaded adapter instance
        self._loaded_adapter_instance: Optional[ModelAdapter] = None
        self._loaded_adapter_key: Optional[str] = None
        log.debug("VideoGenerator initialized.")

    def register_adapter(self, model_name: str, adapter_class: Type[ModelAdapter]):
        """
        Registers a model adapter class for a given model name.

        Args:
            model_name: The case-insensitive name of the model (e.g., "Mochi1").
            adapter_class: The class implementing the ModelAdapter protocol.
        """
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not (hasattr(adapter_class, 'load_model') and
                hasattr(adapter_class, 'unload_model') and
                hasattr(adapter_class, 'generate')):
            raise TypeError(f"Adapter class {adapter_class.__name__} does not fully implement ModelAdapter protocol.")

        adapter_key = model_name.lower()
        if adapter_key in self._adapter_classes:
            log.warning(f"Overwriting existing adapter registration for model: {adapter_key}")
        self._adapter_classes[adapter_key] = adapter_class
        log.info(f"Registered adapter '{adapter_class.__name__}' for model: {model_name}")

    def _get_adapter(self, model_name: str) -> ModelAdapter:
        """
        Retrieves the appropriate adapter instance, handling loading/unloading.

        Instantiates and loads the adapter if it's not the currently loaded one.
        Unloads the previous adapter if switching models.

        Args:
            model_name: The case-insensitive name of the model required.

        Returns:
            The loaded ModelAdapter instance for the requested model.

        Raises:
            ValueError: If no adapter is registered for the model name.
            RuntimeError: If adapter instantiation or loading fails.
        """
        adapter_key = model_name.lower()
        if adapter_key not in self._adapter_classes:
            log.error(f"Attempted to get adapter for unregistered model: {model_name}")
            raise ValueError(f"No adapter registered for model: {model_name}")

        # If the requested adapter is already loaded, return it
        if self._loaded_adapter_key == adapter_key and self._loaded_adapter_instance:
            log.debug(f"Using already loaded adapter for model: {adapter_key}")
            return self._loaded_adapter_instance

        # --- Need to load a new or different adapter ---

        # Unload the previous adapter if one is loaded
        if self._loaded_adapter_instance:
            log.info(f"Unloading previous model adapter: {self._loaded_adapter_key}")
            try:
                self._loaded_adapter_instance.unload_model()
            except Exception as e:
                # Log warning but continue, as the goal is to load the new one
                log.warning(f"Error unloading adapter {self._loaded_adapter_key}: {e}", exc_info=True)
            finally:
                # Ensure state is cleared even if unload fails
                self._loaded_adapter_instance = None
                self._loaded_adapter_key = None

        # Instantiate and load the new adapter
        log.info(f"Loading model adapter for: {adapter_key}")
        try:
            adapter_class = self._adapter_classes[adapter_key]
            adapter_instance = adapter_class() # Instantiate
            adapter_instance.load_model()      # Load model resources (can raise errors)
            self._loaded_adapter_instance = adapter_instance
            self._loaded_adapter_key = adapter_key
            log.info(f"Successfully loaded adapter for model: {adapter_key}")
            return self._loaded_adapter_instance
        except Exception as e:
            log.error(f"Failed to instantiate or load model adapter {adapter_key}: {e}", exc_info=True)
            # Clear state in case of partial load failure
            self._loaded_adapter_instance = None
            self._loaded_adapter_key = None
            # Re-raise as a runtime error to signal critical failure
            raise RuntimeError(f"Failed to load model adapter {adapter_key}") from e

    def generate_for_test_case(self, test_case: TestCase, output_dir: Path) -> GeneratedVideo:
        """
        Generates a video for a given TestCase using the appropriate adapter.

        Handles adapter selection, loading (via _get_adapter), execution,
        and error propagation.

        Args:
            test_case: The TestCase defining the generation parameters.
            output_dir: The directory where the adapter should save the output video.

        Returns:
            A GeneratedVideo object referencing the created video file upon success.

        Raises:
            ValueError: If no adapter is registered for the test_case.model_name.
            RuntimeError: If adapter loading or video generation fails.
            TypeError: If input arguments are invalid types.
        """
        if not isinstance(test_case, TestCase):
             raise TypeError("Input 'test_case' must be a TestCase object.")
        if not isinstance(output_dir, Path):
             raise TypeError("Input 'output_dir' must be a Path object.")

        log.info(f"Processing generation request for test case: {test_case.id} (Model: {test_case.model_name})")

        try:
            # Get (and potentially load/switch) the correct adapter
            adapter = self._get_adapter(test_case.model_name)

            # Ensure the specific output directory for this test case exists
            # The adapter expects this directory to be ready.
            output_dir.mkdir(parents=True, exist_ok=True)

            # Delegate the actual generation task to the adapter
            log.debug(f"Calling generate on adapter {type(adapter).__name__} for test case {test_case.id}")
            generated_video = adapter.generate(test_case=test_case, output_dir=output_dir)

            log.info(f"Successfully generated video for test case {test_case.id} at: {generated_video.file_path}")
            return generated_video

        except (ValueError, RuntimeError, TypeError) as e:
             # Catch errors from _get_adapter or adapter.generate or type errors
             log.error(f"Video generation failed for test case {test_case.id}: {e}", exc_info=True)
             raise # Re-raise the caught exception for the orchestrator to handle
        except Exception as e:
             # Catch any other unexpected errors during the process
             log.error(f"Unexpected error during video generation for test case {test_case.id}: {e}", exc_info=True)
             # Wrap in a RuntimeError to indicate a general failure
             raise RuntimeError(f"Unexpected error during generation for {test_case.id}") from e

    def unload_all_adapters(self):
        """
        Unloads any currently loaded adapter. Should be called during cleanup.
        """
        if self._loaded_adapter_instance:
             log.info(f"Performing final unload for adapter: {self._loaded_adapter_key}")
             try:
                 self._loaded_adapter_instance.unload_model()
             except Exception as e:
                 log.warning(f"Error during final unload of adapter {self._loaded_adapter_key}: {e}", exc_info=True)
             finally:
                 # Clear state regardless of unload success/failure
                 self._loaded_adapter_instance = None
                 self._loaded_adapter_key = None
                 log.debug("Adapter state cleared.")
        else:
             log.debug("No adapter currently loaded, skipping final unload.")
