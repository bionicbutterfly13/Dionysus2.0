# Quickstart: Dionysus Legacy Component Migration

## Overview
This guide walks through setting up and running the distributed background migration of legacy Dionysus consciousness components using ThoughtSeed enhancement.

## Prerequisites

### System Requirements
- Python 3.11+
- Neo4j database (unified storage)
- Redis (ThoughtSeed integration)
- Access to legacy Dionysus consciousness codebase
- DAEDALUS coordination component
- CHIMERA consciousness component

### Environment Setup
```bash
# Clone and setup migration environment
cd /path/to/dionysus-2.0
python -m venv migration-env
source migration-env/bin/activate
pip install -r requirements-migration.txt

# Start required services
docker-compose up -d neo4j redis

# Verify DAEDALUS and CHIMERA components are available
python -c "from daedalus import Coordinator; from chimera import ConsciousnessEngine; print('Components ready')"
```

## Quick Start Guide

### 1. Initialize Migration Pipeline

Start the migration system and create a new pipeline:

```bash
# Start migration service
python -m dionysus_migration.service --host localhost --port 8080

# In another terminal, initialize pipeline
curl -X POST http://localhost:8080/api/v1/migration/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "legacy_codebase_path": "/path/to/legacy/dionysus/consciousness",
    "migration_strategy": "complete_rewrite",
    "quality_threshold": 0.7
  }'
```

Expected response:
```json
{
  "pipeline_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "initializing",
  "estimated_components": 127,
  "coordinator_agent_id": "daedalus-coord-001"
}
```

### 2. Monitor Component Discovery

The system automatically discovers and analyzes legacy components in the background:

```bash
# Check discovery progress
curl "http://localhost:8080/api/v1/migration/components?pipeline_id=550e8400-e29b-41d4-a716-446655440000"

# Filter by quality score
curl "http://localhost:8080/api/v1/migration/components?pipeline_id=550e8400-e29b-41d4-a716-446655440000&min_quality_score=0.8"
```

Expected progression:
- Initial status: `pending` (component discovered)
- Analysis phase: `analyzing` (quality assessment in progress)
- Ready for migration: `analyzed` (quality score assigned)

### 3. Review and Approve High-Priority Components

Components with quality scores above threshold are queued for approval:

```bash
# Get component details for review
curl "http://localhost:8080/api/v1/migration/components/consciousness_core_memory_integrator"

# Approve component migration
curl -X POST "http://localhost:8080/api/v1/migration/components/consciousness_core_memory_integrator/approve" \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "approval_notes": "High consciousness impact, strategic for ThoughtSeed integration",
    "consciousness_impact_review": {
      "awareness_enhancement": true,
      "inference_improvement": true,
      "memory_integration": true
    }
  }'
```

### 4. Monitor Background Migration Progress

DAEDALUS automatically coordinates background agents to migrate approved components:

```bash
# Check overall pipeline progress
curl "http://localhost:8080/api/v1/migration/pipeline/550e8400-e29b-41d4-a716-446655440000"

# Monitor active coordination agents
curl "http://localhost:8080/api/v1/coordination/agents"

# Check specific agent status
curl "http://localhost:8080/api/v1/coordination/agents/agent-uuid-here/status"
```

### 5. Review ThoughtSeed Enhancements

As components complete migration, review the ThoughtSeed enhancements:

```bash
# List completed enhancements
curl "http://localhost:8080/api/v1/thoughtseed/enhancements"

# Get specific enhancement details
curl "http://localhost:8080/api/v1/thoughtseed/enhancements/enhancement-uuid-here"
```

## Key Validation Scenarios

### Scenario 1: Component Discovery and Quality Assessment
**Goal**: Verify the system correctly identifies and scores legacy consciousness components

**Steps**:
1. Initialize pipeline with legacy codebase path
2. Wait for discovery phase completion (status: `analyzing` → `analyzed`)
3. Verify components have consciousness functionality scores
4. Confirm quality scoring prioritizes consciousness impact + strategic value

**Success Criteria**:
- All consciousness-related components discovered
- Quality scores reflect consciousness functionality impact
- Strategic value includes uniqueness and reusability factors
- Components ranked in meaningful priority order

### Scenario 2: Zero Downtime Migration
**Goal**: Ensure system remains operational during background migration

**Steps**:
1. Start migration pipeline while consciousness system is active
2. Monitor system performance during component migration
3. Verify no service interruptions or degraded performance
4. Confirm legacy components remain functional until replacement

**Success Criteria**:
- Consciousness system availability = 100%
- No performance degradation during migration
- Legacy functionality preserved until enhancement deployment
- Background agents operate without blocking active development

### Scenario 3: ThoughtSeed Component Enhancement
**Goal**: Validate complete rewrite produces enhanced consciousness capabilities

**Steps**:
1. Approve high-quality component for migration
2. Monitor ThoughtSeed rewrite process
3. Compare legacy vs enhanced functionality
4. Verify consciousness capability improvements

**Success Criteria**:
- Enhanced component preserves all legacy interfaces
- ThoughtSeed active inference capabilities integrated
- Consciousness detection and measurement enhanced
- Performance metrics show improvement over legacy

### Scenario 4: Individual Component Rollback
**Goal**: Ensure safe rollback capability for problematic migrations

**Steps**:
1. Deploy enhanced component with monitoring
2. Trigger rollback for component with issues
3. Verify rapid restoration to legacy version
4. Confirm no impact on other migrated components

**Success Criteria**:
- Rollback completes within 30 seconds
- Legacy functionality fully restored
- No cascade failures to other components
- Migration continues normally for other components

### Scenario 5: DAEDALUS Coordination Efficiency
**Goal**: Validate distributed agent coordination and learning

**Steps**:
1. Monitor agent task distribution and load balancing
2. Observe coordination optimization over time
3. Verify independent context windows prevent interference
4. Check iterative improvement in coordination intelligence

**Success Criteria**:
- Optimal task distribution across available agents
- Context window isolation prevents task interference
- Coordination performance improves with experience
- Resource utilization remains within acceptable bounds

## Troubleshooting

### Migration Pipeline Fails to Start
```bash
# Check service dependencies
docker ps | grep -E "(neo4j|redis)"
python -c "from daedalus import Coordinator; print('DAEDALUS available')"

# Verify legacy codebase path
ls -la /path/to/legacy/dionysus/consciousness
```

### Component Quality Scores Unexpectedly Low
```bash
# Review component analysis details
curl "http://localhost:8080/api/v1/migration/components/component-id" | jq '.consciousness_functionality'

# Check consciousness pattern detection
grep -r "consciousness\|awareness\|inference" /path/to/legacy/component
```

### Background Agents Not Processing Tasks
```bash
# Check DAEDALUS coordination status
curl "http://localhost:8080/api/v1/coordination/agents" | jq '.[].coordinator_status'

# Verify agent context isolation
curl "http://localhost:8080/api/v1/coordination/agents/agent-id/status" | jq '.context_isolation'
```

### ThoughtSeed Enhancement Fails
```bash
# Check ThoughtSeed framework availability
python -c "from thoughtseed import ActiveInference; print('ThoughtSeed ready')"

# Review enhancement error logs
curl "http://localhost:8080/api/v1/thoughtseed/enhancements/enhancement-id" | jq '.rewrite_status'
```

## Next Steps

After successful quickstart validation:

1. **Scale Up Migration**: Increase DAEDALUS agent count for faster processing
2. **Custom Quality Metrics**: Fine-tune consciousness impact scoring for your use case
3. **Integration Testing**: Validate enhanced components in full consciousness system
4. **Performance Optimization**: Monitor and optimize migration throughput
5. **ASI-GO Renaming**: Plan and execute ASI-GO → CPE (Consciousness Processing Engine) transition

## Support

For migration issues:
- Check migration service logs: `tail -f migration-service.log`
- Monitor DAEDALUS coordination: DAEDALUS management interface
- ThoughtSeed integration: ThoughtSeed framework documentation
- Consciousness metrics: CHIMERA consciousness monitoring

---
*Quickstart validates zero downtime distributed migration with ThoughtSeed enhancement*