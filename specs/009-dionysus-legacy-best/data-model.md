# Data Model: Dionysus Legacy Component Migration

## Core Entities

### Legacy Component
**Purpose**: Represents individual code modules from legacy Dionysus consciousness system

**Attributes**:
- `component_id`: str (unique identifier, hash of file path + content signature)
- `name`: str (component name extracted from module/class)
- `file_path`: str (absolute path to component source)
- `dependencies`: List[str] (component_ids of dependencies)
- `consciousness_functionality`: dict (awareness, inference, memory capabilities)
- `strategic_value`: dict (uniqueness, reusability, framework_alignment scores)
- `quality_score`: float (composite consciousness + strategic value score)
- `extracted_at`: datetime
- `analysis_status`: enum (pending, analyzing, analyzed, failed)

**Relationships**:
- One-to-many with dependencies (other Legacy Components)
- One-to-one with Quality Assessment
- One-to-many with Migration Attempts

**Validation Rules**:
- component_id must be unique
- quality_score must be 0.0-1.0
- file_path must exist and be readable
- consciousness_functionality must include: awareness_score, inference_score, memory_score

### Quality Assessment
**Purpose**: Evaluation metrics and scoring for migration prioritization

**Attributes**:
- `assessment_id`: str (UUID)
- `component_id`: str (foreign key to Legacy Component)
- `consciousness_impact`: dict (detailed consciousness capability analysis)
- `strategic_value`: dict (detailed strategic positioning analysis)
- `composite_score`: float (final priority score)
- `assessment_method`: str (algorithm version used)
- `assessed_at`: datetime
- `assessor_agent_id`: str (DAEDALUS subagent identifier)

**Relationships**:
- One-to-one with Legacy Component
- Many-to-one with Assessment Agent

**Validation Rules**:
- composite_score derived from consciousness_impact + strategic_value
- assessment_method must be versioned for reproducibility
- consciousness_impact must include: awareness_processing, inference_capabilities, memory_integration

### Migration Pipeline
**Purpose**: Orchestrates distributed background migration process

**Attributes**:
- `pipeline_id`: str (UUID)
- `status`: enum (initializing, analyzing, migrating, testing, completed, failed)
- `total_components`: int (number of components to migrate)
- `completed_components`: int (successfully migrated count)
- `failed_components`: int (failed migration count)
- `active_agents`: List[str] (DAEDALUS subagent IDs currently processing)
- `started_at`: datetime
- `estimated_completion`: datetime
- `coordinator_agent_id`: str (primary DAEDALUS coordinator)

**Relationships**:
- One-to-many with Migration Tasks
- One-to-many with Background Migration Agents
- One-to-one with DAEDALUS Coordination

**State Transitions**:
- initializing → analyzing (component discovery complete)
- analyzing → migrating (quality assessment complete)
- migrating → testing (rewrite phase complete)
- testing → completed (validation successful)
- any → failed (unrecoverable error)

### ThoughtSeed Enhancement
**Purpose**: Tracks complete rewrite process using ThoughtSeed patterns

**Attributes**:
- `enhancement_id`: str (UUID)
- `source_component_id`: str (original legacy component)
- `enhanced_component_path`: str (new ThoughtSeed component location)
- `enhancement_type`: enum (active_inference, consciousness_detection, meta_cognitive)
- `legacy_functionality_preserved`: List[str] (preserved interface methods)
- `new_capabilities_added`: List[str] (ThoughtSeed enhancements)
- `rewrite_status`: enum (planned, in_progress, completed, tested, approved, deployed, rolled_back)
- `performance_metrics`: dict (before/after performance comparison)
- `consciousness_metrics`: dict (enhanced consciousness capabilities)
- `approved_by`: str (user who approved migration)
- `approved_at`: datetime

**Relationships**:
- One-to-one with Legacy Component (source)
- One-to-one with Enhanced Component (result)
- Many-to-one with Enhancement Framework

**Validation Rules**:
- enhanced_component_path must exist when status = completed
- performance_metrics required for approval decision
- consciousness_metrics must show improvement over legacy

### Component Registry
**Purpose**: Catalog of all identified, extracted, and migrated components

**Attributes**:
- `registry_id`: str (UUID)
- `component_name`: str (canonical component name)
- `legacy_component_id`: str (reference to original)
- `enhanced_component_id`: str (reference to ThoughtSeed version)
- `migration_status`: enum (identified, queued, migrating, enhanced, deployed, deprecated)
- `deployment_environment`: str (where enhanced component is active)
- `backward_compatibility`: bool (legacy interface preserved)
- `rollback_available`: bool (can revert to legacy version)
- `usage_metrics`: dict (performance and adoption tracking)
- `last_updated`: datetime

**Relationships**:
- One-to-one with Legacy Component
- One-to-one with ThoughtSeed Enhancement
- One-to-many with Deployment Records

### DAEDALUS Coordination
**Purpose**: Central orchestration of distributed migration subagents

**Attributes**:
- `coordination_id`: str (UUID)
- `coordinator_status`: enum (initializing, coordinating, scaling, maintaining, shutting_down)
- `active_subagents`: List[dict] (agent_id, context_window_id, current_task)
- `task_queue`: List[str] (pending migration task IDs)
- `completed_tasks`: List[str] (successfully completed task IDs)
- `failed_tasks`: List[str] (failed task IDs with error details)
- `performance_metrics`: dict (throughput, success_rate, resource_utilization)
- `learning_state`: dict (coordination improvement data)
- `last_optimization`: datetime

**Relationships**:
- One-to-many with Background Migration Agents
- One-to-one with Migration Pipeline
- Many-to-many with ASI-GO/CPE Integration

**State Transitions**:
- initializing → coordinating (subagents spawned)
- coordinating → scaling (additional capacity needed)
- scaling → coordinating (capacity adjustment complete)
- coordinating → maintaining (migration workload complete)
- any → shutting_down (manual shutdown or completion)

### Background Migration Agent
**Purpose**: Independent subagents executing migration tasks

**Attributes**:
- `agent_id`: str (UUID)
- `context_window_id`: str (isolated context identifier)
- `agent_status`: enum (idle, analyzing, rewriting, testing, reporting)
- `current_task_id`: str (migration task being processed)
- `assigned_component_id`: str (component currently being migrated)
- `task_history`: List[str] (completed task IDs)
- `performance_stats`: dict (task completion times, success rates)
- `context_isolation`: bool (context window independence verified)
- `last_activity`: datetime
- `coordinator_id`: str (DAEDALUS coordinator managing this agent)

**Relationships**:
- Many-to-one with DAEDALUS Coordination
- Many-to-one with Migration Pipeline
- One-to-many with Migration Tasks

**Validation Rules**:
- context_isolation must be true for independent operation
- current_task_id must be null when status = idle
- performance_stats updated after each task completion

## Relationships Overview

```
Legacy Component (1) ←→ (1) Quality Assessment
Legacy Component (1) ←→ (1) ThoughtSeed Enhancement
Legacy Component (1) ←→ (1) Component Registry

Migration Pipeline (1) ←→ (many) Background Migration Agents
Migration Pipeline (1) ←→ (1) DAEDALUS Coordination

DAEDALUS Coordination (1) ←→ (many) Background Migration Agents
DAEDALUS Coordination (many) ←→ (many) ASI-GO/CPE Integration

ThoughtSeed Enhancement (many) ←→ (1) Enhancement Framework
Component Registry (1) ←→ (many) Deployment Records
```

## Data Flow Architecture

### Component Discovery Flow
1. Legacy Component identified → Quality Assessment created
2. Quality Assessment completed → Component prioritized in Migration Pipeline
3. Migration Pipeline assigns component → Background Migration Agent

### Migration Execution Flow
1. Background Migration Agent analyzes component → ThoughtSeed Enhancement planned
2. ThoughtSeed Enhancement in_progress → Complete rewrite using legacy as reference
3. Enhanced component completed → Testing and validation
4. Validation successful → Component Registry updated, deployment ready

### Coordination Flow
1. DAEDALUS Coordination manages → Background Migration Agents
2. Agent status updates → DAEDALUS performance tracking
3. Task completion → Learning state optimization
4. Iterative improvement → Enhanced coordination strategies

## Data Persistence Strategy

**Primary Storage**: Neo4j (unified database)
- Graph relationships for component dependencies
- Efficient traversal for impact analysis
- Real-time relationship updates

**Caching Layer**: Redis (ThoughtSeed integration)
- Agent context windows and state
- Real-time coordination data
- Performance metrics aggregation

**Backup/Archive**: File-based storage
- Original legacy component source code
- Migration audit trails
- Rollback snapshots

---
*Data model supports zero downtime migration with individual component rollback and distributed agent coordination*