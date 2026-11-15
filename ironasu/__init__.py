"""ironasu: adaptive local LLM gateway and experimental runtime components."""
from .core import AdaptiveEngine, Config, HTTPBackend, InferenceError
from .scheduling import (
    AdaptiveSpeculationScheduler,
    LayerSpec,
    NgramDraftIndex,
    OrderedLayerPrefetcher,
    SpeculationChoice,
)
from .speculative_state import (
    BusySpeculationError,
    Snapshot,
    SpeculationResult,
    SpeculativeInitializer,
    StateStore,
)

__version__ = "1.5.0"
__all__ = [
    "AdaptiveEngine", "Config", "HTTPBackend", "InferenceError",
    "AdaptiveSpeculationScheduler", "LayerSpec", "NgramDraftIndex",
    "OrderedLayerPrefetcher", "SpeculationChoice",
    "BusySpeculationError", "Snapshot", "SpeculationResult",
    "SpeculativeInitializer", "StateStore",
]
