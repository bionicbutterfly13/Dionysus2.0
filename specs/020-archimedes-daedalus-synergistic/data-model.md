# Data Model: Archimedes-Daedalus Synergistic System

## Core Entities

### EvolutionaryPattern
**Purpose**: Represents problem-solving patterns that evolve over time
**Fields**:
- `pattern_id`: String (UUID, unique identifier)
- `name`: String (human-readable pattern name)
- `description`: String (pattern description)
- `pattern_signature`: SemanticSignature (high-dimensional vector representation)
- `solution_template`: SolutionTemplate (structured solution approach)
- `success_rate`: Float (0.0-1.0, pattern effectiveness)
- `confidence_score`: Float (0.0-1.0, pattern reliability)
- `generation`: Integer (evolution generation number)
- `parent_patterns`: List[String] (IDs of parent patterns)
- `mutation_history`: List[Mutation] (record of evolutionary changes)
- `performance_metrics`: PerformanceMetrics (speed, accuracy, resource usage)
- `last_accessed`: Timestamp (for capacity management)
- `basin_activation_count`: Integer (frequency of archetype activation)
- `created_at`: Timestamp
- `last_updated`: Timestamp

**Relationships**:
- Many-to-many with SpecializedAgent (agents can use multiple patterns)
- Self-referencing (parent-child pattern evolution)

### SpecializedAgent
**Purpose**: Represents domain-specific agents with refined contexts
**Fields**:
- `agent_id`: String (UUID, unique identifier)
- `name`: String (agent name)
- `subspecialty_domain`: String (domain expertise area)
- `capability_profile`: CapabilityProfile (skills and tool access)
- `context_boundaries`: ContextBoundaries (domain scope and limits)
- `performance_history`: PerformanceHistory (past task outcomes)
- `available_tools`: List[String] (accessible tool identifiers)
- `creation_context`: ProblemContext (situation that triggered creation)
- `last_accessed`: Timestamp (for capacity management)
- `usage_frequency`: Integer (how often agent is selected)
- `created_at`: Timestamp

**Relationships**:
- Many-to-many with EvolutionaryPattern
- Many-to-many with ReasoningCommittee (committee membership)

### ProblemAgentMatch
**Purpose**: Represents semantic matches between problems and agents
**Fields**:
- `match_id`: String (UUID, unique identifier)
- `problem_signature`: SemanticSignature (problem representation)
- `agent_id`: String (foreign key to SpecializedAgent)
- `similarity_score`: Float (0.0-1.0, semantic similarity)
- `affordance_compatibility`: Float (0.0-1.0, capability match)
- `confidence_level`: Float (0.0-1.0, match reliability)
- `expected_performance`: Float (0.0-1.0, predicted success rate)
- `contextual_factors`: List[ContextualFactor] (environmental considerations)
- `reasoning_explanation`: String (match rationale)
- `match_timestamp`: Timestamp

**Relationships**:
- Many-to-one with SpecializedAgent
- One-to-one with Problem (if stored)

### ReasoningCommittee
**Purpose**: Represents collaborative agent groups for problem-solving
**Fields**:
- `committee_id`: String (UUID, unique identifier)
- `problem_context`: ProblemContext (problem being addressed)
- `member_agents`: List[String] (agent IDs in committee)
- `formation_strategy`: String (how committee was assembled)
- `coordination_protocol`: String (communication method)
- `consensus_mechanism`: String (decision-making approach)
- `session_state`: String (active, completed, failed)
- `created_at`: Timestamp
- `completed_at`: Timestamp (nullable)

**Relationships**:
- Many-to-many with SpecializedAgent

### CognitiveToolset
**Purpose**: Represents reasoning enhancement tools
**Fields**:
- `tool_id`: String (unique identifier)
- `tool_name`: String (understand_question, recall_related, examine_answer, backtracking)
- `implementation`: String (tool implementation reference)
- `applicability_context`: List[String] (when to use this tool)
- `effectiveness_metrics`: ToolMetrics (success rates, performance)

### AffordanceMapping
**Purpose**: Maps problem characteristics to agent capabilities
**Fields**:
- `mapping_id`: String (UUID, unique identifier)
- `problem_affordances`: List[Affordance] (what problem offers/requires)
- `agent_capabilities`: List[Capability] (what agent can do)
- `compatibility_score`: Float (0.0-1.0, how well they match)
- `mapping_context`: String (domain or situation context)
- `created_at`: Timestamp

## Supporting Data Structures

### SemanticSignature
- `vector`: List[Float] (high-dimensional embedding)
- `dimensionality`: Integer (vector size)
- `encoding_method`: String (how vector was generated)

### SolutionTemplate
- `approach`: String (general solution strategy)
- `steps`: List[String] (ordered solution steps)
- `required_tools`: List[String] (needed capabilities)
- `success_criteria`: List[String] (how to measure success)

### PerformanceMetrics
- `accuracy`: Float (0.0-1.0)
- `speed`: Float (milliseconds)
- `resource_usage`: Float (computational cost)
- `user_satisfaction`: Float (0.0-1.0)

### CapabilityProfile
- `available_tools`: List[String] (tool identifiers)
- `skill_levels`: Dict[String, Float] (skill → proficiency mapping)
- `domain_knowledge`: List[String] (knowledge areas)
- `processing_limits`: ProcessingLimits (computational constraints)

### ProblemContext
- `problem_description`: String
- `domain`: String
- `complexity_level`: String (simple, medium, complex)
- `required_expertise`: List[String]
- `time_constraints`: TimeConstraints (deadlines, urgency)

## Database Schema Considerations

### Primary Storage (Local)
- **SQLite** for structured data (entities, relationships)
- **JSON files** for complex nested structures (context, metrics)
- **Binary files** for semantic vectors (efficient storage/retrieval)

### Real-time Cache (Redis)
- Pattern matching results (100ms requirement)
- Agent availability status
- Active session state

### Knowledge Graph (Neo4j)
- Pattern evolution relationships
- Agent specialization hierarchies
- Problem-solution mappings
- Affordance networks

### Backup Storage (Cloud)
- Encrypted exports of all local data
- Selective sync with mobile app (future)
- Version history for pattern evolution

## Validation Rules

### Pattern Library
- Pattern signatures must be unique within 95% similarity threshold
- Success rates must be updated after each pattern usage
- Basin activation counts increment on pattern access

### Agent Population
- Agent specialization domains must be non-overlapping above 80% similarity
- Capability profiles must include at least one available tool
- Performance history required after 5+ task completions

### Committee Formation
- Maximum 5 agents per committee (cognitive load limit)
- Agent capabilities must complement each other
- Committee formation time must not exceed 2 seconds

### Capacity Management
- Pattern library size limit: 10,000 patterns
- Agent population limit: 1,000 agents
- Auto-purge triggers at 90% capacity
- User alert required before any purging

## Data Lifecycle

### Pattern Evolution
1. **Creation**: New patterns generated from successful solutions
2. **Mutation**: Patterns modified based on feedback
3. **Selection**: Successful patterns retained, failures removed
4. **Decay**: Unused patterns marked for removal (but basin activation resets)

### Agent Lifecycle
1. **Dynamic Creation**: Agents created on-demand for specific contexts
2. **Specialization**: Agents refined through task performance
3. **Retirement**: Agents removed when superseded or unused
4. **Archival**: Agent configurations preserved for future reference

### Knowledge Base Growth
1. **Personal Input**: User-provided ideas and knowledge
2. **Solution Learning**: Insights from problem-solving sessions
3. **Pattern Discovery**: Emergent patterns from repeated solutions
4. **Relationship Mapping**: Connections between ideas and concepts