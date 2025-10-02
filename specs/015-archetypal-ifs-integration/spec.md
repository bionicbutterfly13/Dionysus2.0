# Archetypal Models & Internal Family Systems Integration

**Feature Branch**: `015-archetypal-ifs-integration`  
**Created**: 2025-09-27  
**Status**: Draft  
**Integration**: Narrative Map Extraction + Consciousness Processing

## Overview

Integration specification for archetypal pattern recognition and Internal Family Systems (IFS) analysis within the Flux CE narrative extraction system. This enhancement enables deep psychological pattern matching, episodic memory analysis, and isomorphic metaphor detection for therapeutic and transformational insights.

## Theoretical Foundation

### Archetypal Patterns (Jungian Model)
Core archetypal roles in narrative structures:
- **Hero**: The protagonist on a transformational journey
- **Mentor**: Wise guide providing knowledge and tools
- **Shadow**: Repressed or denied aspects, antagonistic forces
- **Anima/Animus**: Contrasexual aspects, soul figures
- **Trickster**: Boundary-crosser, catalyst for change
- **Mother**: Nurturing, life-giving, protective force
- **Father**: Authority, structure, discipline
- **Child**: Innocence, potential, new beginnings

### Internal Family Systems (IFS) Model
Core parts dynamics in psychological narratives:
- **Protector Parts**: Shield vulnerable parts, manage external relationships
- **Exile Parts**: Carry pain, trauma, unmet needs from the past
- **Firefighter Parts**: Emergency responders, impulsive reaction patterns
- **Self**: Core, undamaged essence with natural leadership qualities

## Core Requirements

### FR-001: Archetypal Pattern Recognition Engine
- System MUST identify archetypal roles within narrative structures
- System MUST track archetypal transformations across story progression
- System MUST correlate user experiences with universal archetypal patterns
- System MUST enable cross-cultural archetypal pattern matching
- System MUST support archetypal constellation analysis (multiple archetypes interacting)

### FR-002: IFS Parts Detection System
- System MUST identify Protector, Exile, and Firefighter patterns in narratives
- System MUST detect Self-leadership moments in user stories
- System MUST recognize parts blending and polarization dynamics
- System MUST track internal conflicts between parts
- System MUST identify protective strategies and their effectiveness

### FR-003: Episodic Memory Pattern Analysis
- System MUST analyze user episodic memories for archetypal and IFS patterns
- System MUST correlate personal stories with universal narrative structures
- System MUST identify recurring patterns across user's life experiences
- System MUST detect therapeutic opportunities through pattern recognition
- System MUST map emotional patterns to archetypal energies

### FR-004: Isomorphic Metaphor Detection
- System MUST identify similar patterns across different narrative domains
- System MUST detect metaphorical relationships between stories
- System MUST enable pattern transfer from one context to another
- System MUST recognize symbolic representations of internal dynamics
- System MUST support therapeutic metaphor construction

### FR-005: Pattern Evolution Tracking
- System MUST track archetypal pattern changes over time
- System MUST monitor IFS parts integration and healing progression
- System MUST identify transformational moments in user narratives
- System MUST detect pattern maturation and development
- System MUST recognize stuck patterns and growth opportunities

## Technical Implementation

### Archetypal Pattern Database

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/models/archetypal_patterns.py

from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

class ArchetypalRole(str, Enum):
    HERO = "hero"
    MENTOR = "mentor" 
    SHADOW = "shadow"
    ANIMA = "anima"
    ANIMUS = "animus"
    TRICKSTER = "trickster"
    MOTHER = "mother"
    FATHER = "father"
    CHILD = "child"
    WISE_OLD_MAN = "wise_old_man"
    WISE_OLD_WOMAN = "wise_old_woman"
    INNOCENT = "innocent"
    EXPLORER = "explorer"
    SAGE = "sage"
    OUTLAW = "outlaw"
    MAGICIAN = "magician"
    LOVER = "lover"
    JESTER = "jester"
    CAREGIVER = "caregiver"
    CREATOR = "creator"
    RULER = "ruler"

class IFSPartType(str, Enum):
    PROTECTOR = "protector"
    EXILE = "exile"
    FIREFIGHTER = "firefighter"
    SELF = "self"

class ArchetypalPattern(BaseModel):
    archetype: ArchetypalRole
    characteristics: List[str]
    typical_behaviors: List[str]
    growth_patterns: List[str]
    shadow_aspects: List[str]
    integration_markers: List[str]
    narrative_functions: List[str]

class IFSPattern(BaseModel):
    part_type: IFSPartType
    protective_strategies: List[str]
    emotional_markers: List[str]
    behavioral_indicators: List[str]
    healing_indicators: List[str]
    self_leadership_markers: List[str]

class PersonalPattern(BaseModel):
    user_id: str
    pattern_type: str  # archetypal or ifs
    pattern_id: str
    confidence: float
    evidence_events: List[str]  # Event IDs
    first_detected: str
    evolution_timeline: List[Dict[str, Any]]
    therapeutic_insights: List[str]
```

### Pattern Recognition Engine

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/services/pattern_recognition/

class ArchetypalPatternRecognizer:
    """Detect archetypal patterns in narratives"""
    
    def __init__(self):
        self.archetype_embeddings = self._load_archetype_embeddings()
        self.pattern_library = self._load_pattern_library()
        
    async def identify_archetypal_roles(self, narrative: Narrative) -> List[ArchetypalMatch]:
        """Identify archetypal roles in narrative characters and events"""
        
        # Extract character actions and motivations
        character_profiles = await self._extract_character_profiles(narrative)
        
        # Match against archetypal patterns
        archetypal_matches = []
        for character in character_profiles:
            matches = await self._match_to_archetypes(character)
            archetypal_matches.extend(matches)
            
        return archetypal_matches
    
    async def detect_archetypal_transformations(self, narrative: Narrative) -> List[Transformation]:
        """Track archetypal transformations throughout narrative"""
        
        # Analyze narrative progression
        story_stages = await self._identify_story_stages(narrative)
        
        # Track archetypal changes across stages
        transformations = []
        for stage_pair in zip(story_stages[:-1], story_stages[1:]):
            transformation = await self._analyze_archetypal_shift(stage_pair)
            if transformation:
                transformations.append(transformation)
                
        return transformations

class IFSPatternRecognizer:
    """Detect Internal Family Systems patterns in narratives"""
    
    def __init__(self):
        self.ifs_patterns = self._load_ifs_patterns()
        self.emotional_lexicon = self._load_emotional_lexicon()
        
    async def identify_parts_dynamics(self, narrative: Narrative) -> List[IFSMatch]:
        """Identify IFS parts and their dynamics in narratives"""
        
        # Extract emotional and behavioral indicators
        emotional_markers = await self._extract_emotional_markers(narrative)
        behavioral_patterns = await self._extract_behavioral_patterns(narrative)
        
        # Match to IFS parts patterns
        parts_matches = []
        for pattern in self.ifs_patterns:
            match = await self._match_to_ifs_pattern(
                pattern, emotional_markers, behavioral_patterns
            )
            if match.confidence > 0.7:
                parts_matches.append(match)
                
        return parts_matches
    
    async def detect_internal_conflicts(self, narrative: Narrative) -> List[InternalConflict]:
        """Detect conflicts between different parts"""
        
        parts = await self.identify_parts_dynamics(narrative)
        
        conflicts = []
        for part1 in parts:
            for part2 in parts:
                if self._are_parts_in_conflict(part1, part2):
                    conflict = InternalConflict(
                        part1=part1,
                        part2=part2,
                        conflict_type=self._classify_conflict(part1, part2),
                        intensity=self._calculate_conflict_intensity(part1, part2)
                    )
                    conflicts.append(conflict)
                    
        return conflicts

class IsomorphicMetaphorDetector:
    """Detect similar patterns across different narrative domains"""
    
    def __init__(self):
        self.metaphor_mappings = self._load_metaphor_mappings()
        self.domain_patterns = self._load_domain_patterns()
        
    async def detect_cross_domain_patterns(
        self, 
        narratives: List[Narrative]
    ) -> List[IsomorphicPattern]:
        """Detect similar patterns across different narrative domains"""
        
        # Extract structural patterns from each narrative
        structural_patterns = []
        for narrative in narratives:
            pattern = await self._extract_structural_pattern(narrative)
            structural_patterns.append(pattern)
            
        # Find isomorphic relationships
        isomorphic_patterns = []
        for pattern1 in structural_patterns:
            for pattern2 in structural_patterns:
                if pattern1.narrative_id != pattern2.narrative_id:
                    similarity = await self._calculate_pattern_similarity(pattern1, pattern2)
                    if similarity > 0.8:
                        isomorphic = IsomorphicPattern(
                            pattern1=pattern1,
                            pattern2=pattern2,
                            similarity_score=similarity,
                            metaphorical_mapping=await self._generate_metaphor_mapping(
                                pattern1, pattern2
                            )
                        )
                        isomorphic_patterns.append(isomorphic)
                        
        return isomorphic_patterns
```

### Integration with Consciousness System

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/services/consciousness_pattern_bridge.py

class ConsciousnessPatternBridge:
    """Bridge between consciousness system and pattern recognition"""
    
    async def enhance_pattern_recognition_with_consciousness(
        self,
        narrative: Narrative,
        consciousness_level: str
    ) -> EnhancedPatternAnalysis:
        """Enhance pattern recognition with consciousness insights"""
        
        # Get consciousness state during narrative events
        consciousness_states = await self._get_consciousness_states(narrative)
        
        # Correlate consciousness with archetypal patterns
        archetypal_consciousness = await self._correlate_archetypes_consciousness(
            narrative, consciousness_states
        )
        
        # Analyze IFS parts through consciousness lens
        ifs_consciousness = await self._analyze_ifs_through_consciousness(
            narrative, consciousness_states
        )
        
        # Detect consciousness-guided transformations
        transformations = await self._detect_consciousness_transformations(
            narrative, consciousness_states
        )
        
        return EnhancedPatternAnalysis(
            archetypal_patterns=archetypal_consciousness,
            ifs_patterns=ifs_consciousness,
            consciousness_transformations=transformations,
            integration_opportunities=await self._identify_integration_opportunities(
                archetypal_consciousness, ifs_consciousness
            )
        )
    
    async def generate_therapeutic_insights(
        self,
        pattern_analysis: EnhancedPatternAnalysis,
        user_context: UserContext
    ) -> List[TherapeuticInsight]:
        """Generate therapeutic insights from pattern analysis"""
        
        insights = []
        
        # Archetypal insights
        for pattern in pattern_analysis.archetypal_patterns:
            if pattern.needs_integration:
                insight = await self._generate_archetypal_insight(pattern, user_context)
                insights.append(insight)
                
        # IFS insights
        for pattern in pattern_analysis.ifs_patterns:
            if pattern.indicates_healing_opportunity:
                insight = await self._generate_ifs_insight(pattern, user_context)
                insights.append(insight)
                
        # Integration insights
        for opportunity in pattern_analysis.integration_opportunities:
            insight = await self._generate_integration_insight(opportunity, user_context)
            insights.append(insight)
            
        return insights
```

### Database Schema Extensions

```sql
-- Neo4j Cypher for archetypal and IFS pattern storage

CREATE (a:ArchetypalPattern {
    id: $pattern_id,
    archetype: $archetype_type,
    confidence: $confidence,
    narrative_function: $function,
    transformation_stage: $stage
})

CREATE (i:IFSPattern {
    id: $pattern_id,
    part_type: $part_type,
    confidence: $confidence,
    protective_strategy: $strategy,
    emotional_signature: $emotions
})

CREATE (m:IsomorphicMapping {
    id: $mapping_id,
    source_domain: $source,
    target_domain: $target,
    similarity_score: $similarity,
    metaphor_type: $type
})

CREATE (t:TherapeuticInsight {
    id: $insight_id,
    insight_type: $type,
    content: $content,
    user_relevance: $relevance,
    therapeutic_value: $value
})

-- Relationships
CREATE (narrative)-[:CONTAINS_ARCHETYPE]->(a)
CREATE (narrative)-[:CONTAINS_IFS_PATTERN]->(i)
CREATE (narrative1)-[:ISOMORPHIC_TO]->(narrative2)
CREATE (pattern)-[:SUGGESTS_INSIGHT]->(t)
CREATE (user)-[:HAS_PATTERN]->(pattern)
CREATE (pattern)-[:EVOLVES_TO]->(pattern2)
```

### API Specifications

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/api/pattern_analysis.py

@app.post("/patterns/archetypal/analyze")
async def analyze_archetypal_patterns(
    narrative_id: str,
    include_transformations: bool = True
) -> List[ArchetypalMatch]:
    """Analyze archetypal patterns in narrative"""
    pass

@app.post("/patterns/ifs/analyze")
async def analyze_ifs_patterns(
    narrative_id: str,
    include_conflicts: bool = True
) -> List[IFSMatch]:
    """Analyze Internal Family Systems patterns"""
    pass

@app.post("/patterns/isomorphic/detect")
async def detect_isomorphic_patterns(
    narrative_ids: List[str],
    similarity_threshold: float = 0.8
) -> List[IsomorphicPattern]:
    """Detect isomorphic patterns across narratives"""
    pass

@app.get("/patterns/therapeutic/insights/{user_id}")
async def get_therapeutic_insights(
    user_id: str,
    pattern_types: Optional[List[str]] = None
) -> List[TherapeuticInsight]:
    """Get therapeutic insights from pattern analysis"""
    pass

@app.post("/patterns/evolution/track")
async def track_pattern_evolution(
    user_id: str,
    time_range: str = "6_months"
) -> PatternEvolutionReport:
    """Track evolution of patterns over time"""
    pass
```

## Agent Delegation Tasks

### Task 1: Archetypal Pattern Recognition Agent
**Specialization**: Jungian archetypal analysis and pattern matching  
**Deliverables**:
- Implement archetypal role identification
- Create archetypal transformation tracking
- Build cross-cultural pattern matching
- Develop archetypal constellation analysis

### Task 2: IFS Pattern Detection Agent
**Specialization**: Internal Family Systems analysis  
**Deliverables**:
- Implement IFS parts identification
- Create internal conflict detection
- Build Self-leadership recognition
- Develop parts integration tracking

### Task 3: Isomorphic Metaphor Agent
**Specialization**: Cross-domain pattern matching and metaphor detection  
**Deliverables**:
- Implement pattern similarity algorithms
- Create metaphorical mapping generation
- Build domain transfer mechanisms
- Develop therapeutic metaphor construction

### Task 4: Consciousness-Pattern Integration Agent
**Specialization**: Bridging consciousness system with pattern recognition  
**Deliverables**:
- Integrate consciousness states with pattern analysis
- Create therapeutic insight generation
- Build pattern evolution tracking
- Develop integration opportunity detection

## Updated Clarification Questions

**13. Archetypal Pattern Priorities**
- Which archetypal roles should be prioritized for detection first?
- Should we focus on classical Jungian archetypes or expand to include cultural variants?

**14. IFS Therapeutic Integration**
- How deeply should the system integrate therapeutic insights?
- Should therapeutic suggestions be provided or just pattern recognition?

**15. Pattern Evolution Tracking**
- What timeframe should be used for tracking pattern evolution?
- How should the system handle conflicting or contradictory patterns?

**16. Isomorphic Metaphor Applications** 
- Should metaphor detection focus on therapeutic applications or general insight?
- How should the system validate the accuracy of metaphorical mappings?

---

**Enhanced narrative extraction now includes deep psychological pattern analysis for therapeutic and transformational insights through archetypal and IFS integration!**