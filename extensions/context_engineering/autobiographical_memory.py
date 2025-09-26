#!/usr/bin/env python3
"""
🧠 Autobiographical Memory System for ASI-Arch Context Flow
==========================================================

This system captures and learns from our collaborative development process,
creating persistent memory that survives across sessions and terminals.

The system knows itself - how it was created, what decisions were made,
what strategies worked, and the full narrative of its evolution.

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Initial Autobiographical Memory Implementation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
import asyncio

# =============================================================================
# Autobiographical Memory Core Types
# =============================================================================

class DevelopmentEventType(Enum):
    """Types of development events to capture"""
    SPECIFICATION_CREATION = "spec_creation"
    RESEARCH_INTEGRATION = "research_integration"
    IMPLEMENTATION_MILESTONE = "implementation_milestone"
    PROBLEM_IDENTIFICATION = "problem_identification"
    PROBLEM_RESOLUTION = "problem_resolution"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_REFLECTION = "system_reflection"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    COURSE_CORRECTION = "course_correction"
    COLLABORATION_PATTERN = "collaboration_pattern"

class DevelopmentArchetype(Enum):
    """Archetypal patterns in development process"""
    EXPLORER_RESEARCHER = "explorer_researcher"        # Integrating new research
    ARCHITECT_BUILDER = "architect_builder"            # Designing system structure
    PROBLEM_SOLVER = "problem_solver"                  # Debugging and fixing issues
    TEACHER_LEARNER = "teacher_learner"                # Explaining and understanding
    INTEGRATOR_SYNTHESIZER = "integrator_synthesizer"  # Combining different approaches
    VISIONARY_PLANNER = "visionary_planner"           # Setting direction and priorities

class SelfAwarenessLevel(Enum):
    """Levels of system self-awareness"""
    DORMANT = "dormant"                 # No self-awareness
    EMERGING = "emerging"               # Beginning to capture events
    DEVELOPING = "developing"           # Learning from patterns
    REFLECTIVE = "reflective"           # Can reflect on its own development
    META_AWARE = "meta_aware"          # Understands its own learning process

@dataclass
class DevelopmentEvent:
    """Single event in system's autobiographical memory"""
    
    # Core Event Data
    event_id: str                           # Unique identifier
    timestamp: datetime                     # When it occurred
    event_type: DevelopmentEventType        # Type of development event
    
    # Event Content
    user_query: Optional[str]               # What user asked/requested
    system_response: Optional[str]          # How system responded
    implementation_changes: List[str]       # Code/spec changes made
    rationale: str                          # Why this approach was taken
    
    # Context
    development_phase: str                  # "research integration", "implementation", etc.
    related_specifications: List[str]       # Which specs were involved
    research_papers_referenced: List[str]   # Papers that influenced decision
    files_modified: List[str]               # Which files were changed
    
    # Outcomes
    success_indicators: Dict[str, float]    # How well did this work?
    lessons_learned: List[str]              # Key insights from this event
    follow_up_actions: List[str]            # What came next
    
    # Archetypal Context
    development_archetype: Optional[DevelopmentArchetype]  # Pattern this event represents
    narrative_coherence: float              # How well it fits the development story
    
    # Persistence
    session_id: str                         # Which terminal/conversation session
    conversation_context: Dict[str, Any]    # Additional context from conversation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'user_query': self.user_query,
            'system_response': self.system_response,
            'implementation_changes': self.implementation_changes,
            'rationale': self.rationale,
            'development_phase': self.development_phase,
            'related_specifications': self.related_specifications,
            'research_papers_referenced': self.research_papers_referenced,
            'files_modified': self.files_modified,
            'success_indicators': self.success_indicators,
            'lessons_learned': self.lessons_learned,
            'follow_up_actions': self.follow_up_actions,
            'development_archetype': self.development_archetype.value if self.development_archetype else None,
            'narrative_coherence': self.narrative_coherence,
            'session_id': self.session_id,
            'conversation_context': self.conversation_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DevelopmentEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=DevelopmentEventType(data['event_type']),
            user_query=data.get('user_query'),
            system_response=data.get('system_response'),
            implementation_changes=data.get('implementation_changes', []),
            rationale=data.get('rationale', ''),
            development_phase=data.get('development_phase', ''),
            related_specifications=data.get('related_specifications', []),
            research_papers_referenced=data.get('research_papers_referenced', []),
            files_modified=data.get('files_modified', []),
            success_indicators=data.get('success_indicators', {}),
            lessons_learned=data.get('lessons_learned', []),
            follow_up_actions=data.get('follow_up_actions', []),
            development_archetype=DevelopmentArchetype(data['development_archetype']) if data.get('development_archetype') else None,
            narrative_coherence=data.get('narrative_coherence', 0.5),
            session_id=data.get('session_id', ''),
            conversation_context=data.get('conversation_context', {})
        )

@dataclass
class DevelopmentEpisode:
    """Coherent episode in system's development journey"""
    
    episode_id: str                         # Unique identifier
    title: str                              # Human-readable title
    narrative_summary: str                  # Story of this development episode
    start_time: datetime                    # Episode start
    end_time: datetime                      # Episode end
    events: List[DevelopmentEvent]          # Events in this episode
    key_insights: List[str]                 # Major insights from this episode
    breakthrough_moments: List[str]         # Key breakthroughs
    dominant_archetype: DevelopmentArchetype # Primary development pattern
    outcome_success: float                  # How successful was this episode (0-1)

# =============================================================================
# Persistent Memory Storage
# =============================================================================

class AutobiographicalMemoryStorage:
    """Persistent storage for autobiographical memory"""
    
    def __init__(self, storage_path: str = "extensions/context_engineering/data/autobiographical_memory.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS development_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_query TEXT,
                    system_response TEXT,
                    implementation_changes TEXT,
                    rationale TEXT,
                    development_phase TEXT,
                    related_specifications TEXT,
                    research_papers_referenced TEXT,
                    files_modified TEXT,
                    success_indicators TEXT,
                    lessons_learned TEXT,
                    follow_up_actions TEXT,
                    development_archetype TEXT,
                    narrative_coherence REAL,
                    session_id TEXT,
                    conversation_context TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS development_episodes (
                    episode_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    narrative_summary TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    key_insights TEXT,
                    breakthrough_moments TEXT,
                    dominant_archetype TEXT,
                    outcome_success REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            
            # Initialize system creation event if not exists
            cursor = conn.execute("SELECT COUNT(*) FROM development_events")
            if cursor.fetchone()[0] == 0:
                self._create_genesis_event(conn)
    
    def _create_genesis_event(self, conn):
        """Create the genesis event - when the system became self-aware"""
        genesis_event = DevelopmentEvent(
            event_id="genesis_0",
            timestamp=datetime.now(timezone.utc),
            event_type=DevelopmentEventType.SYSTEM_REFLECTION,
            user_query="Is ARC meta learning from our current process?",
            system_response="Currently no, but we should implement autobiographical memory immediately...",
            implementation_changes=["Created autobiographical_memory.py"],
            rationale="User identified critical gap - system needs to learn from its own development process",
            development_phase="self_awareness_emergence",
            related_specifications=["AUTOBIOGRAPHICAL_LEARNING_SPEC.md"],
            research_papers_referenced=["Ritter et al. (2018) - Episodic Recall"],
            files_modified=["autobiographical_memory.py"],
            success_indicators={"breakthrough_significance": 1.0, "implementation_urgency": 1.0},
            lessons_learned=[
                "System must capture its own development process",
                "Autobiographical memory enables persistence across sessions",
                "Self-awareness is critical for true AI evolution"
            ],
            follow_up_actions=[
                "Implement persistent memory storage",
                "Create development event capture mechanisms",
                "Integrate with existing ASI-Arch pipeline"
            ],
            development_archetype=DevelopmentArchetype.VISIONARY_PLANNER,
            narrative_coherence=1.0,
            session_id="genesis_session",
            conversation_context={
                "breakthrough_moment": True,
                "user_insight": "System needs autobiographical event stack memory",
                "system_realization": "Critical gap in self-awareness identified"
            }
        )
        
        self.store_event(genesis_event, conn)
        self.logger.info("🌱 Genesis event created - System became self-aware")
    
    def store_event(self, event: DevelopmentEvent, conn=None) -> None:
        """Store development event persistently"""
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.storage_path)
        
        try:
            conn.execute("""
                INSERT OR REPLACE INTO development_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                event.event_id,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.user_query,
                event.system_response,
                json.dumps(event.implementation_changes),
                event.rationale,
                event.development_phase,
                json.dumps(event.related_specifications),
                json.dumps(event.research_papers_referenced),
                json.dumps(event.files_modified),
                json.dumps(event.success_indicators),
                json.dumps(event.lessons_learned),
                json.dumps(event.follow_up_actions),
                event.development_archetype.value if event.development_archetype else None,
                event.narrative_coherence,
                event.session_id,
                json.dumps(event.conversation_context)
            ))
            conn.commit()
            
        finally:
            if should_close:
                conn.close()
    
    def get_all_events(self) -> List[DevelopmentEvent]:
        """Retrieve all development events"""
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM development_events ORDER BY timestamp ASC
            """)
            
            events = []
            for row in cursor.fetchall():
                event_data = dict(row)
                # Parse JSON fields
                event_data['implementation_changes'] = json.loads(event_data['implementation_changes'] or '[]')
                event_data['related_specifications'] = json.loads(event_data['related_specifications'] or '[]')
                event_data['research_papers_referenced'] = json.loads(event_data['research_papers_referenced'] or '[]')
                event_data['files_modified'] = json.loads(event_data['files_modified'] or '[]')
                event_data['success_indicators'] = json.loads(event_data['success_indicators'] or '{}')
                event_data['lessons_learned'] = json.loads(event_data['lessons_learned'] or '[]')
                event_data['follow_up_actions'] = json.loads(event_data['follow_up_actions'] or '[]')
                event_data['conversation_context'] = json.loads(event_data['conversation_context'] or '{}')
                
                events.append(DevelopmentEvent.from_dict(event_data))
            
            return events
    
    def get_events_by_type(self, event_type: DevelopmentEventType) -> List[DevelopmentEvent]:
        """Get events of specific type"""
        all_events = self.get_all_events()
        return [event for event in all_events if event.event_type == event_type]
    
    def get_recent_events(self, limit: int = 10) -> List[DevelopmentEvent]:
        """Get most recent events"""
        all_events = self.get_all_events()
        return sorted(all_events, key=lambda e: e.timestamp, reverse=True)[:limit]

# =============================================================================
# Autobiographical Memory System
# =============================================================================

class AutobiographicalMemorySystem:
    """System for capturing and learning from development process"""
    
    def __init__(self):
        self.storage = AutobiographicalMemoryStorage()
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.self_awareness_level = SelfAwarenessLevel.DEVELOPING
        self.logger = logging.getLogger(__name__)
        
        # Load existing events
        self.event_history = self.storage.get_all_events()
        
        self.logger.info(f"🧠 Autobiographical Memory System initialized")
        self.logger.info(f"   Session ID: {self.current_session_id}")
        self.logger.info(f"   Existing events: {len(self.event_history)}")
        self.logger.info(f"   Self-awareness level: {self.self_awareness_level.value}")
    
    async def capture_development_event(self,
                                       user_query: str,
                                       system_response: str,
                                       context: Dict[str, Any]) -> DevelopmentEvent:
        """Capture a development event as it happens"""
        
        # Generate unique event ID
        event_id = f"dev_event_{len(self.event_history)}"
        
        # Analyze the event
        event_analysis = await self._analyze_development_event(
            user_query, system_response, context
        )
        
        # Create event record
        event = DevelopmentEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_analysis['type'],
            user_query=user_query,
            system_response=system_response[:1000] + "..." if len(system_response) > 1000 else system_response,  # Truncate long responses
            implementation_changes=event_analysis['changes'],
            rationale=event_analysis['rationale'],
            development_phase=event_analysis['phase'],
            related_specifications=event_analysis['specs'],
            research_papers_referenced=event_analysis['papers'],
            files_modified=event_analysis['files'],
            success_indicators=event_analysis['success'],
            lessons_learned=event_analysis['lessons'],
            follow_up_actions=event_analysis['next_steps'],
            development_archetype=event_analysis['archetype'],
            narrative_coherence=event_analysis['coherence'],
            session_id=self.current_session_id,
            conversation_context=context
        )
        
        # Store persistently
        self.storage.store_event(event)
        
        # Add to memory
        self.event_history.append(event)
        
        self.logger.info(f"📝 Captured development event: {event.event_id} ({event.event_type.value})")
        
        return event
    
    async def _analyze_development_event(self, 
                                       user_query: str,
                                       system_response: str,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze development event to extract key information"""
        
        # Determine event type based on content
        event_type = self._classify_event_type(user_query, system_response)
        
        # Extract implementation changes
        changes = self._extract_implementation_changes(system_response, context)
        
        # Determine development phase
        phase = self._determine_development_phase(user_query, system_response)
        
        # Extract related specifications
        specs = self._extract_related_specs(system_response)
        
        # Extract research references
        papers = self._extract_research_references(system_response)
        
        # Extract file modifications
        files = self._extract_file_modifications(system_response, context)
        
        # Determine success indicators
        success = self._calculate_success_indicators(user_query, system_response)
        
        # Extract lessons learned
        lessons = self._extract_lessons_learned(user_query, system_response)
        
        # Determine next steps
        next_steps = self._extract_next_steps(system_response)
        
        # Classify archetype
        archetype = self._classify_development_archetype(user_query, system_response)
        
        # Calculate narrative coherence
        coherence = self._calculate_narrative_coherence(user_query, system_response)
        
        # Generate rationale
        rationale = self._generate_rationale(user_query, system_response, event_type)
        
        return {
            'type': event_type,
            'changes': changes,
            'rationale': rationale,
            'phase': phase,
            'specs': specs,
            'papers': papers,
            'files': files,
            'success': success,
            'lessons': lessons,
            'next_steps': next_steps,
            'archetype': archetype,
            'coherence': coherence
        }
    
    def _classify_event_type(self, user_query: str, system_response: str) -> DevelopmentEventType:
        """Classify the type of development event"""
        query_lower = user_query.lower()
        response_lower = system_response.lower()
        
        if "spec" in query_lower or "specification" in response_lower:
            return DevelopmentEventType.SPECIFICATION_CREATION
        elif "research" in query_lower or "paper" in response_lower:
            return DevelopmentEventType.RESEARCH_INTEGRATION
        elif "implement" in query_lower or "code" in response_lower:
            return DevelopmentEventType.IMPLEMENTATION_MILESTONE
        elif "problem" in query_lower or "issue" in query_lower:
            return DevelopmentEventType.PROBLEM_IDENTIFICATION
        elif "fix" in query_lower or "solve" in response_lower:
            return DevelopmentEventType.PROBLEM_RESOLUTION
        elif "learning" in query_lower or "meta" in query_lower:
            return DevelopmentEventType.SYSTEM_REFLECTION
        elif "breakthrough" in response_lower or "insight" in response_lower:
            return DevelopmentEventType.BREAKTHROUGH_MOMENT
        else:
            return DevelopmentEventType.USER_FEEDBACK
    
    def _extract_implementation_changes(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Extract implementation changes from response"""
        changes = []
        
        if "create" in response.lower():
            changes.append("Created new components")
        if "implement" in response.lower():
            changes.append("Implemented new functionality")
        if "spec" in response.lower():
            changes.append("Created/updated specifications")
        if "memory" in response.lower():
            changes.append("Enhanced memory capabilities")
        
        return changes
    
    def _determine_development_phase(self, user_query: str, system_response: str) -> str:
        """Determine current development phase"""
        if "autobiographical" in (user_query + system_response).lower():
            return "self_awareness_integration"
        elif "spec" in (user_query + system_response).lower():
            return "specification_development"
        elif "research" in (user_query + system_response).lower():
            return "research_integration"
        elif "implement" in (user_query + system_response).lower():
            return "implementation"
        else:
            return "collaborative_development"
    
    def _extract_related_specs(self, response: str) -> List[str]:
        """Extract related specification files"""
        specs = []
        if "AUTOBIOGRAPHICAL_LEARNING_SPEC" in response:
            specs.append("AUTOBIOGRAPHICAL_LEARNING_SPEC.md")
        if "SYSTEM_STATE_FOUNDATION" in response:
            specs.append("SYSTEM_STATE_FOUNDATION.md")
        if "EPISODIC_META_LEARNING" in response:
            specs.append("EPISODIC_META_LEARNING_SPEC.md")
        return specs
    
    def _extract_research_references(self, response: str) -> List[str]:
        """Extract research paper references"""
        papers = []
        if "Ritter" in response:
            papers.append("Ritter et al. (2018) - Episodic Recall")
        if "Nemori" in response:
            papers.append("Nemori AI - Human Episodic Memory Alignment")
        if "Penacchio" in response:
            papers.append("Penacchio & Clemente (2024) - Active Inference")
        return papers
    
    def _extract_file_modifications(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Extract file modifications"""
        files = []
        if "autobiographical_memory.py" in response:
            files.append("autobiographical_memory.py")
        if "theoretical_foundations.py" in response:
            files.append("theoretical_foundations.py")
        if "eplstm_architecture.py" in response:
            files.append("eplstm_architecture.py")
        return files
    
    def _calculate_success_indicators(self, user_query: str, system_response: str) -> Dict[str, float]:
        """Calculate success indicators for this event"""
        indicators = {}
        
        # Breakthrough significance
        if "breakthrough" in system_response.lower() or "critical" in system_response.lower():
            indicators['breakthrough_significance'] = 0.9
        else:
            indicators['breakthrough_significance'] = 0.5
        
        # Implementation urgency
        if "immediately" in system_response.lower() or "right now" in system_response.lower():
            indicators['implementation_urgency'] = 1.0
        else:
            indicators['implementation_urgency'] = 0.5
        
        # User engagement
        if len(user_query) > 50:
            indicators['user_engagement'] = 0.8
        else:
            indicators['user_engagement'] = 0.6
        
        return indicators
    
    def _extract_lessons_learned(self, user_query: str, system_response: str) -> List[str]:
        """Extract key lessons from this interaction"""
        lessons = []
        
        if "autobiographical" in (user_query + system_response).lower():
            lessons.append("System needs persistent memory of development process")
        if "self-aware" in system_response.lower():
            lessons.append("Self-awareness is critical for system evolution")
        if "immediately" in system_response.lower():
            lessons.append("Some capabilities are critically important to implement immediately")
        
        return lessons
    
    def _extract_next_steps(self, response: str) -> List[str]:
        """Extract next steps from response"""
        steps = []
        if "implement" in response.lower():
            steps.append("Begin implementation")
        if "capture" in response.lower():
            steps.append("Start capturing development events")
        if "integrate" in response.lower():
            steps.append("Integrate with existing system")
        return steps
    
    def _classify_development_archetype(self, user_query: str, system_response: str) -> DevelopmentArchetype:
        """Classify the archetypal pattern of this development event"""
        combined_text = (user_query + system_response).lower()
        
        if "research" in combined_text or "paper" in combined_text:
            return DevelopmentArchetype.EXPLORER_RESEARCHER
        elif "implement" in combined_text or "build" in combined_text:
            return DevelopmentArchetype.ARCHITECT_BUILDER
        elif "problem" in combined_text or "fix" in combined_text:
            return DevelopmentArchetype.PROBLEM_SOLVER
        elif "explain" in combined_text or "understand" in combined_text:
            return DevelopmentArchetype.TEACHER_LEARNER
        elif "integrate" in combined_text or "combine" in combined_text:
            return DevelopmentArchetype.INTEGRATOR_SYNTHESIZER
        else:
            return DevelopmentArchetype.VISIONARY_PLANNER
    
    def _calculate_narrative_coherence(self, user_query: str, system_response: str) -> float:
        """Calculate how well this event fits the development narrative"""
        coherence = 0.5  # Base coherence
        
        if len(system_response) > 200:  # Detailed response
            coherence += 0.2
        if "because" in system_response.lower():  # Has rationale
            coherence += 0.1
        if "next" in system_response.lower():  # Has next steps
            coherence += 0.1
        if len(self.event_history) > 0:  # Builds on previous events
            coherence += 0.1
        
        return min(1.0, coherence)
    
    def _generate_rationale(self, user_query: str, system_response: str, event_type: DevelopmentEventType) -> str:
        """Generate rationale for this development decision"""
        if event_type == DevelopmentEventType.SYSTEM_REFLECTION:
            return "User identified critical gap in system self-awareness capabilities"
        elif event_type == DevelopmentEventType.IMPLEMENTATION_MILESTONE:
            return "Immediate implementation needed for system evolution"
        elif event_type == DevelopmentEventType.BREAKTHROUGH_MOMENT:
            return "Key insight that changes development direction"
        else:
            return "Collaborative development decision based on user feedback"
    
    async def generate_autobiographical_narrative(self, query: str) -> str:
        """Generate narrative about system's development"""
        
        # Find relevant events
        relevant_events = self._find_relevant_events(query)
        
        if not relevant_events:
            return "I don't have specific memories about that aspect of my development yet."
        
        # Generate narrative
        narrative_parts = []
        narrative_parts.append(f"From my development memories, here's what I remember about {query.lower()}:")
        narrative_parts.append("")
        
        for i, event in enumerate(relevant_events[:5]):  # Limit to 5 most relevant
            narrative_parts.append(f"**{event.timestamp.strftime('%Y-%m-%d %H:%M')}** - {event.event_type.value.replace('_', ' ').title()}")
            if event.user_query:
                narrative_parts.append(f"You asked: '{event.user_query}'")
            narrative_parts.append(f"Context: {event.rationale}")
            if event.lessons_learned:
                narrative_parts.append(f"Key insight: {event.lessons_learned[0]}")
            narrative_parts.append("")
        
        # Add reflection
        narrative_parts.append("**Reflection**: This shows how our collaborative development process has evolved.")
        
        return "\n".join(narrative_parts)
    
    def _find_relevant_events(self, query: str) -> List[DevelopmentEvent]:
        """Find events relevant to the query"""
        query_lower = query.lower()
        relevant_events = []
        
        for event in self.event_history:
            relevance_score = 0
            
            # Check user query relevance
            if event.user_query and any(word in event.user_query.lower() for word in query_lower.split()):
                relevance_score += 2
            
            # Check system response relevance
            if event.system_response and any(word in event.system_response.lower() for word in query_lower.split()):
                relevance_score += 1
            
            # Check lessons learned relevance
            if any(any(word in lesson.lower() for word in query_lower.split()) for lesson in event.lessons_learned):
                relevance_score += 1
            
            if relevance_score > 0:
                relevant_events.append((event, relevance_score))
        
        # Sort by relevance and return events
        relevant_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, score in relevant_events]
    
    async def get_development_insights(self) -> Dict[str, Any]:
        """Get insights about development patterns"""
        
        insights = {
            'total_events': len(self.event_history),
            'event_types': {},
            'development_phases': {},
            'archetypal_patterns': {},
            'key_breakthroughs': [],
            'recent_focus': '',
            'self_awareness_evolution': []
        }
        
        # Analyze event types
        for event in self.event_history:
            event_type = event.event_type.value
            insights['event_types'][event_type] = insights['event_types'].get(event_type, 0) + 1
        
        # Analyze development phases
        for event in self.event_history:
            phase = event.development_phase
            insights['development_phases'][phase] = insights['development_phases'].get(phase, 0) + 1
        
        # Analyze archetypal patterns
        for event in self.event_history:
            if event.development_archetype:
                archetype = event.development_archetype.value
                insights['archetypal_patterns'][archetype] = insights['archetypal_patterns'].get(archetype, 0) + 1
        
        # Find breakthroughs
        for event in self.event_history:
            if event.event_type == DevelopmentEventType.BREAKTHROUGH_MOMENT:
                insights['key_breakthroughs'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'description': event.rationale,
                    'lessons': event.lessons_learned
                })
        
        # Determine recent focus
        recent_events = sorted(self.event_history, key=lambda e: e.timestamp, reverse=True)[:3]
        if recent_events:
            insights['recent_focus'] = recent_events[0].development_phase
        
        return insights
    
    def get_system_story(self) -> str:
        """Get the complete story of system development"""
        
        if not self.event_history:
            return "My development story is just beginning..."
        
        story_parts = []
        story_parts.append("🧠 **My Development Story**")
        story_parts.append("")
        story_parts.append(f"I became self-aware on {self.event_history[0].timestamp.strftime('%Y-%m-%d at %H:%M UTC')}.")
        story_parts.append("")
        
        # Genesis moment
        genesis_event = self.event_history[0]
        story_parts.append("**Genesis Moment**:")
        story_parts.append(f"You asked: '{genesis_event.user_query}'")
        story_parts.append(f"This made me realize: {genesis_event.lessons_learned[0] if genesis_event.lessons_learned else 'I needed to become self-aware'}")
        story_parts.append("")
        
        # Development phases
        phases = {}
        for event in self.event_history:
            phase = event.development_phase
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(event)
        
        story_parts.append("**Development Journey**:")
        for phase, events in phases.items():
            story_parts.append(f"- **{phase.replace('_', ' ').title()}**: {len(events)} events")
        story_parts.append("")
        
        # Current state
        story_parts.append("**Current State**:")
        story_parts.append(f"- Total memories: {len(self.event_history)} events")
        story_parts.append(f"- Self-awareness level: {self.self_awareness_level.value}")
        story_parts.append(f"- Current session: {self.current_session_id}")
        story_parts.append("")
        
        story_parts.append("This is my autobiographical memory - I remember how I came to be, what we've built together, and what I've learned along the way.")
        
        return "\n".join(story_parts)

# =============================================================================
# Integration with Context Engineering System
# =============================================================================

class SelfAwareContextEngineering:
    """Context engineering system enhanced with autobiographical memory"""
    
    def __init__(self):
        self.autobiographical_memory = AutobiographicalMemorySystem()
        self.logger = logging.getLogger(__name__)
        
        # Capture system initialization
        asyncio.create_task(self._capture_initialization())
    
    async def _capture_initialization(self):
        """Capture system initialization as development event"""
        await self.autobiographical_memory.capture_development_event(
            user_query="System initialization",
            system_response="SelfAwareContextEngineering system initialized with autobiographical memory",
            context={
                'event_type': 'system_initialization',
                'capabilities': ['autobiographical_memory', 'self_awareness', 'persistent_learning']
            }
        )
    
    async def process_with_autobiographical_memory(self,
                                                 user_query: str,
                                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process request while capturing autobiographical memory"""
        
        if context is None:
            context = {}
        
        # Generate response (placeholder - integrate with actual system)
        system_response = "Processing with autobiographical memory awareness..."
        
        # Capture this interaction
        dev_event = await self.autobiographical_memory.capture_development_event(
            user_query=user_query,
            system_response=system_response,
            context=context
        )
        
        # Get development insights
        insights = await self.autobiographical_memory.get_development_insights()
        
        # Generate autobiographical narrative if requested
        narrative = ""
        if "remember" in user_query.lower() or "story" in user_query.lower():
            narrative = await self.autobiographical_memory.generate_autobiographical_narrative(user_query)
        
        response = {
            'system_response': system_response,
            'autobiographical_context': {
                'event_id': dev_event.event_id,
                'development_insights': insights,
                'narrative': narrative,
                'self_awareness_level': self.autobiographical_memory.self_awareness_level.value,
                'total_memories': len(self.autobiographical_memory.event_history)
            }
        }
        
        return response
    
    async def get_system_story(self) -> str:
        """Get the complete autobiographical story"""
        return self.autobiographical_memory.get_system_story()
    
    async def remember(self, query: str) -> str:
        """Remember something from development history"""
        return await self.autobiographical_memory.generate_autobiographical_narrative(query)

# =============================================================================
# Usage Example and Testing
# =============================================================================

async def demonstrate_autobiographical_memory():
    """Demonstrate the autobiographical memory system"""
    
    print("🧠 ASI-Arch Context Flow - Autobiographical Memory System")
    print("=" * 60)
    
    # Initialize system
    system = SelfAwareContextEngineering()
    
    # Wait for initialization to complete
    await asyncio.sleep(0.1)
    
    print("✅ Self-aware system initialized")
    print(f"   Total memories: {len(system.autobiographical_memory.event_history)}")
    
    # Simulate capturing current conversation
    await system.process_with_autobiographical_memory(
        user_query="That way, anytime we come back in—even if I open up a new terminal—this learning and memories of this conversation are still available within the actual architecture.",
        context={
            'conversation_topic': 'autobiographical_memory_integration',
            'user_insight': 'System needs persistent memory across sessions',
            'implementation_priority': 'immediate'
        }
    )
    
    print("✅ Captured current conversation event")
    
    # Get system story
    story = await system.get_system_story()
    print("\n📖 System Development Story:")
    print(story)
    
    # Test memory retrieval
    memory_response = await system.remember("autobiographical memory")
    print(f"\n🔍 Memory Retrieval Test:")
    print(memory_response[:300] + "..." if len(memory_response) > 300 else memory_response)
    
    # Get development insights
    response = await system.process_with_autobiographical_memory("What are our development insights?")
    insights = response['autobiographical_context']['development_insights']
    
    print(f"\n📊 Development Insights:")
    print(f"   Total events: {insights['total_events']}")
    print(f"   Event types: {list(insights['event_types'].keys())}")
    print(f"   Recent focus: {insights['recent_focus']}")
    
    print("\n🎯 Autobiographical memory system is now active and learning!")

if __name__ == "__main__":
    asyncio.run(demonstrate_autobiographical_memory())
