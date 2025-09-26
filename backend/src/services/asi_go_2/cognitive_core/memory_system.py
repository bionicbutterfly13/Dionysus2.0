"""
Simple memory system for ASI-GO-2 - minimal TDD implementation
"""
from typing import List, Optional, Dict
from dataclasses import dataclass

@dataclass
class Pattern:
    """A problem-solving pattern"""
    name: str
    description: str
    success_rate: float
    confidence: float

class CognitionBase:
    """Basic pattern storage and retrieval"""

    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}

    def add_pattern(self, pattern: Pattern):
        """Store a pattern"""
        self.patterns[pattern.name] = pattern

    def get_pattern(self, name: str) -> Optional[Pattern]:
        """Get a pattern by name"""
        return self.patterns.get(name)

    def get_relevant_patterns(self, goal: str) -> List[Pattern]:
        """Get patterns relevant to a goal - simple keyword matching for now"""
        relevant = []
        goal_lower = goal.lower()

        for pattern in self.patterns.values():
            if any(word in pattern.name.lower() or word in pattern.description.lower()
                   for word in goal_lower.split()):
                relevant.append(pattern)

        # Sort by success_rate * confidence for simple relevance ranking
        relevant.sort(key=lambda p: p.success_rate * p.confidence, reverse=True)
        return relevant