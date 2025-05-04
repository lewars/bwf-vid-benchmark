import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging # Added for logging potential issues

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# --- Helper for JSON Serialization ---

def _to_dict_for_json(obj: Any) -> Any:
    """
    Recursively converts dataclasses and Path objects to JSON-serializable types.
    """
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        # Convert nested dataclasses recursively
        return {k: _to_dict_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict_for_json(v) for k, v in obj.items()}
    # Handle other basic types directly
    return obj


@dataclass(frozen=True)
class PeakResourceUsage:
    """Immutable dataclass holding peak resource usage."""
    pass

@dataclass
class TestMetrics:
    """Mutable dataclass holding results for a single test case run."""
    pass

class MetricsRecorder:
    """
    Handles saving benchmark metrics to disk (JSON details, CSV summary).
    """
    pass
