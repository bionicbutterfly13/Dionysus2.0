"""
Simple thoughtseed competition system - minimal TDD implementation
"""
from typing import List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import random

class ThoughtType(Enum):
    GOAL = "goal"
    ACTION = "action"
    BELIEF = "belief"
    PERCEPTION = "perception"

@dataclass
class Thought:
    """A thought in the competition workspace"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    type: ThoughtType = ThoughtType.ACTION
    energy: float = 0.5
    confidence: float = 0.5
    parent_ids: Set[str] = field(default_factory=set)

class ThoughtGenerator:
    """Generates thoughts for the workspace"""

    def __init__(self, workspace):
        self.workspace = workspace

    def generate_thought(self, content: str, thought_type: ThoughtType, parent_ids: List[str] = None) -> Thought:
        """Generate a new thought"""
        thought = Thought(
            content=content,
            type=thought_type,
            parent_ids=set(parent_ids or [])
        )
        return thought

class InnerWorkspace:
    """Workspace where thoughts compete for dominance"""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.thoughts: dict[str, Thought] = {}
        self.watching_enabled = False  # Toggle for detailed state logging
        self.state_logger = None  # Will be set if watching is enabled

    def add_thought(self, thought: Thought):
        """Add a thought to the workspace"""
        self.thoughts[thought.id] = thought

    def update(self):
        """Update the competition - simple energy decay and random fluctuation"""
        if self.watching_enabled and self.state_logger:
            # Capture pre-update state
            pre_state = self._capture_state()
            self.state_logger.log_competition_cycle(pre_state, "pre_update")

        for thought in self.thoughts.values():
            # Simple competition dynamics
            thought.energy = max(0.0, thought.energy + random.uniform(-0.1, 0.1))

        if self.watching_enabled and self.state_logger:
            # Capture post-update state
            post_state = self._capture_state()
            self.state_logger.log_competition_cycle(post_state, "post_update")

    def get_dominant_thought(self) -> Optional[Thought]:
        """Get the thought with highest energy"""
        if not self.thoughts:
            return None

        return max(self.thoughts.values(), key=lambda t: t.energy * t.confidence)

    def enable_watching(self, state_logger):
        """Enable detailed state watching with a logger"""
        self.watching_enabled = True
        self.state_logger = state_logger

    def disable_watching(self):
        """Disable detailed state watching"""
        self.watching_enabled = False
        self.state_logger = None

    def _capture_state(self) -> dict:
        """Capture current workspace state for logging"""
        return {
            "timestamp": self._get_timestamp(),
            "workspace_id": id(self),
            "thought_count": len(self.thoughts),
            "thoughts": {
                thought.id: {
                    "content": thought.content,
                    "type": thought.type.value,
                    "energy": thought.energy,
                    "confidence": thought.confidence,
                    "parent_ids": list(thought.parent_ids)
                } for thought in self.thoughts.values()
            },
            "dominant_thought_id": self.get_dominant_thought().id if self.get_dominant_thought() else None
        }

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp with milliseconds"""
        from datetime import datetime
        return datetime.now().isoformat(timespec='milliseconds') + 'Z'