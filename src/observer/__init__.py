"""Observer module for truth recovery and belief inference.

Provides two types of observers:
1. ICLObserver - In-context learning baseline for evaluating LLM truth recovery
2. Observer - Game observer with belief tracking and oracle queries
"""

from .icl import ICLObserver, ObserverResponse, ANSWER_TOOL
from .prompts import Condition, PromptBuilder, FullContextPromptBuilder
from .observer import (
    Observer,
    InferenceObserver,
    ObserverBeliefs,
    InferredRule,
    ObserverSelectionResult,
    run_observer,
)

__all__ = [
    # ICL observer (baseline experiments)
    "ICLObserver",
    "ObserverResponse",
    "ANSWER_TOOL",
    "Condition",
    "PromptBuilder",
    "FullContextPromptBuilder",
    # Game observer
    "Observer",
    "InferenceObserver",
    "ObserverBeliefs",
    "InferredRule",
    "ObserverSelectionResult",
    "run_observer",
]
