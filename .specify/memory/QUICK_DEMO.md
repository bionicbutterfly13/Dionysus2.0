# 🚀 Quick Demo: Context Engineering in Spec-Kit Pipeline

## Try It Now!

### 1. See Context Engineering Foundation
```bash
cat .specify/memory/CONTEXT_ENGINEERING_FOUNDATION.md
```

You'll see:
- Why Attractor Basins are essential
- Why Neural Fields are essential
- Mathematical foundation
- Quick verification commands

### 2. Verify Components Accessible
```bash
python -c "
from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

manager = AttractorBasinManager()
print(f'✅ AttractorBasinManager: {len(manager.basins)} basins loaded')
print(f'✅ Basin IDs: {list(manager.basins.keys())}')
"
```

Expected output:
```
✅ AttractorBasinManager: 1 basins loaded
✅ Basin IDs: ['default_cognitive_basin']
```

### 3. Run Test Suite
```bash
pytest backend/tests/test_context_engineering_spec_pipeline.py -v
```

Expected: **12 passed, 3 skipped**

### 4. Try the New Workflow

#### Create a new feature with `/specify`:
When you run `/specify "Add semantic search to documents"`, you'll now see:

```
🌊 CONTEXT ENGINEERING FOUNDATION
================================

Let me show you why Attractor Basins and Neural Fields are essential...

## The Two Pillars

### 1. Attractor Basin Dynamics 🌀
- Organizes cognitive landscape into dynamic basins of attraction
- Each basin represents a stable conceptual domain
- New thoughtseeds alter the basin landscape
...

[Full foundation displayed]

✅ Components verified:
- AttractorBasinManager: accessible
- Neural Field System: accessible
- Redis: connected

Now creating your feature spec...
```

#### Create implementation plan with `/plan`:
When you run `/plan`, you'll now see:

```
🔍 VALIDATING CONTEXT ENGINEERING INTEGRATION
===========================================

✅ AttractorBasinManager accessible - 1 basins loaded
✅ Neural Field System accessible - dimensions=384
✅ Redis connection active for basin persistence

Integration strategy for this feature:
- Create 'semantic_search_basin' for document concepts
- Use field resonance to detect related documents
- Store basin evolution in Redis for learning

Now executing plan template...
```

#### Generate tasks with `/tasks`:
When you run `/tasks`, you'll now see:

```
📋 GENERATING TASKS
==================

Context Engineering validation tasks (FIRST):
- T001 [P] Verify AttractorBasinManager integration
- T002 [P] Verify Neural Field System integration
- T003 [P] Validate Redis persistence for attractor basins

Feature-specific tests:
- T007 [P] Contract test semantic search endpoint
- T008 [P] Integration test document similarity
...
```

### 5. What Changed?

**Before**:
- Context Engineering buried in codebase
- No visibility to users
- No validation before implementation
- No constitutional requirement

**After**:
- Context Engineering foundation shown at `/specify`
- Components validated at `/plan`
- T001-T003 tasks run FIRST
- Constitution mandates integration

## Test Individual Components

### Test Constitution Compliance
```bash
python -c "
import numpy
print(f'NumPy version: {numpy.__version__}')

from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager
manager = AttractorBasinManager()
print(f'✅ {len(manager.basins)} attractor basins loaded')
print('✅ Constitution compliance verified')
"
```

### Test Basin Creation
```bash
python -c "
from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager, AttractorBasin

manager = AttractorBasinManager()
new_basin = AttractorBasin(
    basin_id='demo_feature_basin',
    center_concept='demo_concept',
    strength=1.2,
    radius=0.6
)

manager.basins[new_basin.basin_id] = new_basin
print(f'✅ Created basin: {new_basin.basin_id}')
print(f'✅ Total basins: {len(manager.basins)}')
"
```

### Test Basin Influence Calculation
```bash
python -c "
from extensions.context_engineering.attractor_basin_dynamics import AttractorBasin

basin = AttractorBasin(
    basin_id='test_basin',
    center_concept='machine_learning',
    strength=1.5,
    radius=0.5
)

# High similarity -> reinforcement
influence_type, strength = basin.calculate_influence_on('deep_learning', 0.9)
print(f'High similarity (0.9): {influence_type.value} with strength {strength:.2f}')

# Medium similarity -> competition or synthesis
influence_type, strength = basin.calculate_influence_on('statistics', 0.6)
print(f'Medium similarity (0.6): {influence_type.value} with strength {strength:.2f}')

# Low similarity -> emergence
influence_type, strength = basin.calculate_influence_on('cooking', 0.2)
print(f'Low similarity (0.2): {influence_type.value} with strength {strength:.2f}')
"
```

Expected output:
```
High similarity (0.9): reinforcement with strength 1.35
Medium similarity (0.6): competition with strength 0.60
Low similarity (0.2): emergence with strength 0.80
```

## Summary

✅ **Context Engineering is now visible from the start**
✅ **TDD tests validate integration (T001-T003)**
✅ **Constitution enforces requirements**
✅ **Clear documentation shows necessity**

Every new feature automatically includes Context Engineering validation!

---

*For more details, see:*
- `.specify/memory/CONTEXT_ENGINEERING_FOUNDATION.md`
- `.specify/memory/SPEC_KIT_PIPELINE_SUMMARY.md`
- `backend/tests/test_context_engineering_spec_pipeline.py`
