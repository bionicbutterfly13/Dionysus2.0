"""
Flux Data Models
Core entities for the Self-Teaching Consciousness Emulator.
"""

from .user_profile import UserProfile
from .autobiographical_journey import AutobiographicalJourney
from .event_node import EventNode
from .document_artifact import DocumentArtifact
from .concept_node import ConceptNode
from .thoughtseed_trace import ThoughtSeedTrace
from .curiosity_mission import CuriosityMission
from .evaluation_frame import EvaluationFrame
from .visualization_state import VisualizationState

__all__ = [
    "UserProfile",
    "AutobiographicalJourney",
    "EventNode",
    "DocumentArtifact",
    "ConceptNode",
    "ThoughtSeedTrace",
    "CuriosityMission",
    "EvaluationFrame",
    "VisualizationState",
]