from dataclasses import dataclass

@dataclass(frozen=True)
class PeakResourceUsage:
    """Immutable dataclass holding peak resource usage."""
    pass

@dataclass
class TestMetrics:
    """Mutable dataclass holding results for a single test case run."""
    pass
