# Implementation Plan: Modular Consciousness Pipeline Architecture

**Branch**: `024-modular-consciousness-pipeline` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/024-modular-consciousness-pipeline/spec.md`

## Summary
Production-ready modular consciousness pipeline with clean separation of concerns, dependency injection, and pluggable consciousness stages. Each stage (perception, competition, attention, integration) implemented as independent module with standardized interfaces. LangGraph-based orchestration with conditional branching, parallel processing, and hot-swapping capabilities.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: LangGraph, langgraph-checkpoint-redis, pydantic 2.x, pytest, NumPy 2.3.3
**Storage**: Redis (state checkpoints), Neo4j (consciousness traces), Qdrant (stage embeddings)
**Testing**: pytest with >90% coverage target, contract tests per stage, performance benchmarks
**Target Platform**: Linux server, containerized deployment
**Project Type**: Backend library + service
**Performance Goals**: <100ms per stage, 1000+ ThoughtSeeds/sec throughput, <200MB memory per pipeline instance
**Constraints**: Stateless stages (checkpoints in Redis), no side effects in stage logic, all stages independently testable
**Scale/Scope**: 6-10 consciousness stages, 100+ concurrent pipelines, hot-reload without restart

## Constitution Check
*Per constitution v1.0.0*

**✅ NumPy 2.0+ Compliance**: All consciousness stages use NumPy 2.3.3 frozen environment
**✅ TDD Standards**: Each stage requires contract tests BEFORE implementation
**✅ Environment Isolation**: Dedicated consciousness-pipeline-env with frozen deps
**✅ Code Complexity**: Single responsibility per stage, max 200 lines per stage module
**✅ Testing Protocols**: Unit tests per stage + integration tests for stage interactions + performance benchmarks

**No violations detected** - Proceed with Phase 0

## Project Structure

### Documentation (this feature)
```
specs/024-modular-consciousness-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 stage interface contracts
└── tasks.md             # Phase 2 output (/tasks command)
```

### Source Code (repository root - EXTRACT to separate package)
```
consciousness-pipeline/  # NEW separate package
├── src/
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseStage abstract class
│   │   ├── perception.py            # Perception stage
│   │   ├── competition.py           # ThoughtSeed competition
│   │   ├── attention.py             # Attention mechanism
│   │   ├── integration.py           # Integration stage
│   │   ├── reflection.py            # Meta-cognitive reflection
│   │   └── action.py                # Action selection
│   ├── pipeline/
│   │   ├── orchestrator.py          # LangGraph StateGraph
│   │   ├── plugin_loader.py         # Dynamic stage loading
│   │   ├── checkpoint_manager.py    # Redis checkpointing
│   │   └── conditional_router.py    # Stage branching logic
│   ├── models/
│   │   ├── consciousness_state.py   # Pipeline state model
│   │   └── stage_result.py          # Stage output model
│   └── utils/
│       ├── dependency_injection.py  # DI container
│       └── performance_monitor.py   # Stage benchmarking
├── tests/
│   ├── unit/
│   │   ├── test_perception_stage.py
│   │   ├── test_competition_stage.py
│   │   ├── test_attention_stage.py
│   │   └── ... (one per stage)
│   ├── integration/
│   │   ├── test_stage_interactions.py
│   │   └── test_pipeline_flow.py
│   └── performance/
│       └── test_stage_benchmarks.py
├── setup.py
├── pyproject.toml
└── README.md

backend/src/services/
└── consciousness_integration.py  # Import from consciousness-pipeline package
```

**Structure Decision**: Extract to separate `consciousness-pipeline` package for refinement and reuse across projects. Backend imports as dependency. Modular stage architecture with plugin system.

## Phase 0: Outline & Research

### Research Tasks

1. **LangGraph StateGraph patterns for consciousness**
   - Research: Best practices for multi-stage cognitive pipelines with conditional routing
   - Decision criteria: Checkpointing overhead, state size limits, branching complexity
   - Output: StateGraph architecture design

2. **Dependency injection frameworks for Python**
   - Research: Compare dependency-injector, injector, punq for stage configuration
   - Decision criteria: Performance overhead, configuration complexity, testing support
   - Output: DI framework selection

3. **Hot-swapping plugin systems**
   - Research: Python plugin architectures that support runtime reload without restart
   - Decision criteria: Safety, rollback on failure, version compatibility
   - Output: Plugin loading architecture

4. **Consciousness stage interface contracts**
   - Research: Standardized input/output schemas for cognitive processing stages
   - Decision criteria: Extensibility, type safety, validation performance
   - Output: BaseStage interface definition

5. **Performance benchmarking for cognitive pipelines**
   - Research: pytest-benchmark vs custom timing, memory profiling tools
   - Decision criteria: CI/CD integration, historical comparison, overhead
   - Output: Benchmark framework selection

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts

### Data Model Entities (`data-model.md`)

1. **ConsciousnessState**
   - Fields: state_id, thoughtseed_id, stage_history[], current_stage, metadata, created_at
   - Relationships: contains StageResults, persisted in Redis checkpoint
   - State transitions: PERCEPTION → COMPETITION → ATTENTION → INTEGRATION → REFLECTION → ACTION
   - Validation: Stage ordering must be valid, no duplicate stages in history

2. **BaseStage** (Abstract Interface)
   - Methods: `process(state: ConsciousnessState) -> StageResult`
   - Properties: stage_name, stage_version, dependencies[]
   - Validation: All stages must implement process(), must be stateless
   - Testing: Every stage must have contract test verifying interface compliance

3. **StageResult**
   - Fields: stage_name, success: bool, output_data, confidence: float, processing_time_ms, metadata
   - Relationships: produced by Stage, appended to ConsciousnessState history
   - Validation: Confidence in [0, 1], processing_time > 0

4. **PipelineConfig**
   - Fields: config_id, stage_sequence[], conditional_branches{}, parallel_stages[], checkpoint_interval
   - Relationships: defines Pipeline execution graph
   - Validation: No circular dependencies, all stages registered

5. **StagePlugin**
   - Fields: plugin_id, stage_class, version, compatibility[], enabled: bool
   - Relationships: loaded by PluginLoader, instantiated by DI container
   - Validation: Version compatibility check before loading

### Stage Interface Contracts (`contracts/`)

**BaseStage Interface**
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class StageResult(BaseModel):
    stage_name: str
    success: bool
    output_data: dict
    confidence: float
    processing_time_ms: int

class BaseStage(ABC):
    stage_name: str
    stage_version: str

    @abstractmethod
    def process(self, state: ConsciousnessState) -> StageResult:
        """Process consciousness state and return result."""
        pass

    def validate_input(self, state: ConsciousnessState) -> bool:
        """Validate input state before processing."""
        return True

    def get_dependencies(self) -> list[str]:
        """Return list of required previous stages."""
        return []
```

**Perception Stage Contract**
```python
class PerceptionStage(BaseStage):
    stage_name = "perception"

    def process(self, state: ConsciousnessState) -> StageResult:
        # Input: Raw sensory data in state.metadata["input"]
        # Output: Processed perceptual features in result.output_data
        # Confidence: Perception quality score
        pass
```

**Competition Stage Contract**
```python
class CompetitionStage(BaseStage):
    stage_name = "competition"

    def get_dependencies(self) -> list[str]:
        return ["perception"]

    def process(self, state: ConsciousnessState) -> StageResult:
        # Input: Perceptual features from previous stage
        # Output: Winning ThoughtSeed from competition
        # Confidence: Competition winner strength
        pass
```

### Contract Tests

**test_stage_interface_compliance.py**
```python
def test_all_stages_implement_base_interface():
    """All stages must implement BaseStage interface."""
    stages = discover_all_stages()
    for stage_class in stages:
        assert issubclass(stage_class, BaseStage)
        assert hasattr(stage_class, 'process')
        assert hasattr(stage_class, 'stage_name')
```

**test_perception_stage.py**
- Test perception stage accepts valid input
- Test perception stage produces valid StageResult
- Test perception stage validates input schema
- Test perception stage performance <10ms

**test_competition_stage.py**
- Test competition requires perception dependency
- Test competition selects winning ThoughtSeed
- Test competition handles empty input gracefully
- Test competition performance <20ms

**test_pipeline_orchestration.py**
- Test StateGraph executes stages in order
- Test conditional branching works
- Test parallel stages execute concurrently
- Test checkpointing saves/restores state

**test_hot_swapping.py**
- Test plugin reload without pipeline restart
- Test version compatibility check
- Test rollback on plugin load failure
- Test multiple plugin versions coexist

### Quickstart (`quickstart.md`)

1. Define custom consciousness stage implementing BaseStage
2. Register stage with plugin loader
3. Configure pipeline with stage sequence
4. Run pipeline with ThoughtSeed input
5. Monitor stage performance benchmarks

## Phase 2: Task Generation Approach

Tasks will be organized into TDD phases:

**Phase 3.1: Setup**
- Extract consciousness pipeline to separate package
- Setup package structure with setup.py
- Configure pytest with coverage requirements
- Setup Redis for checkpointing

**Phase 3.2: Tests First (TDD RED)**
- Contract test for BaseStage interface (MUST FAIL)
- Contract tests for each stage (MUST FAIL)
- Integration test for pipeline flow (MUST FAIL)
- Performance benchmark tests (MUST FAIL)

**Phase 3.3: Core Implementation (TDD GREEN)**
- Implement BaseStage abstract class
- Implement ConsciousnessState model
- Implement StageResult model
- Implement PerceptionStage
- Implement CompetitionStage
- Implement AttentionStage
- Implement IntegrationStage
- Implement ReflectionStage
- Implement ActionStage

**Phase 3.4: Pipeline Orchestration**
- Implement LangGraph StateGraph orchestrator
- Implement PluginLoader with hot-swap support
- Implement CheckpointManager with Redis backend
- Implement ConditionalRouter for stage branching
- Implement DependencyInjection container

**Phase 3.5: Testing & Polish**
- Achieve >90% test coverage
- Run performance benchmarks
- Optimize slow stages
- Package documentation
- Publish to PyPI as consciousness-pipeline

## Dependencies & Integration Points

**Depends on**:
- ThoughtSeed package (extracted)
- Spec 022: Consciousness orchestrator base
- Spec 023: Context engineering integration
- LangGraph StateGraph architecture

**Integrates with**:
- Backend consciousness_integration.py (consumer)
- Query system (Spec 006) for consciousness queries
- Bulk processing (Spec 008) for batch consciousness
- Knowledge processing (Spec 013) for narrative consciousness

**Package Extraction**:
This spec triggers extraction of consciousness pipeline to separate package for:
- Refinement and purity
- Reuse across projects
- Independent versioning
- Focused testing and development

## Progress Tracking

- [x] Initial constitution check - PASS
- [x] Technical context defined
- [ ] Phase 0: Research complete (awaiting /tasks command trigger)
- [ ] Phase 1: Data model + contracts generated
- [ ] Post-design constitution re-check
- [ ] Phase 2: Tasks generated via /tasks command
- [ ] Package extraction to consciousness-pipeline/

**Status**: Ready for Phase 0 research execution

**CRITICAL**: This spec requires package extraction - coordinate with separate package tasks
