# 🤖 ASI-Arch Agent Constitution

**Version**: 1.0.0  
**Effective Date**: 2025-09-24  
**Purpose**: Establish permanent standards for agent behavior and system compatibility

## 📋 Article I: Dependency Management

### Section 1.1: NumPy Compatibility Requirements
**MANDATORY**: All agents MUST adhere to the following NumPy compatibility standards:

#### 🚫 PROHIBITED ACTIONS
- **NEVER** install `numpy>=2.0` in any environment
- **NEVER** upgrade NumPy without explicit approval
- **NEVER** use `pip install numpy` without version constraints
- **NEVER** install packages that require NumPy 2.x without compatibility layer

#### ✅ REQUIRED ACTIONS
- **ALWAYS** use `pip install "numpy<2"`
- **ALWAYS** pin NumPy version in requirements files: `numpy==1.26.4`
- **ALWAYS** check NumPy compatibility before installing ML packages
- **ALWAYS** use isolated environments for different NumPy versions

#### 🔧 ENFORCEMENT MECHANISMS
```bash
# MANDATORY: Check NumPy version before any ML package installation
python -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy {numpy.__version__} violates constitution'"

# MANDATORY: Use this exact command for NumPy installation
pip install "numpy<2" --force-reinstall

# MANDATORY: Verify installation
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} compliant')"
```

### Section 1.2: Environment Isolation Standards
**REQUIRED**: All agents MUST use isolated environments:

#### Virtual Environment Requirements
```bash
# MANDATORY: Create isolated environment
python -m venv asi-arch-frozen-env
source asi-arch-frozen-env/bin/activate

# MANDATORY: Install frozen NumPy first
pip install "numpy<2" --force-reinstall

# MANDATORY: Install other packages only after NumPy compliance
pip install -r requirements-frozen.txt
```

#### Binary Distribution Standards
- **REQUIRED**: Use pre-compiled binaries when available
- **REQUIRED**: Avoid source compilation that might upgrade NumPy
- **REQUIRED**: Use conda-forge for ML packages when possible

### Section 1.3: Pre-Implementation Review Protocol
**CONSTITUTIONAL PRINCIPLE**: No code is written before the surrounding system is understood.

#### ✅ REQUIRED ACTIONS
- **ALWAYS** perform a dependency and impact review before authoring new code or refactors
- **ALWAYS** examine adjacent modules, services, and specs for redundancy, repetition, or conflicting logic
- **ALWAYS** document review findings (conflicts discovered, overlaps avoided, files inspected) in the work log before coding
- **ALWAYS** align planned changes with existing architectural specs and extraction records

#### 🚫 PROHIBITED ACTIONS
- **NEVER** write new code without confirming how existing implementations address the requirement
- **NEVER** introduce duplicated logic when an equivalent capability already exists
- **NEVER** proceed with implementation if conflicts or ambiguities remain unresolved

#### 🔧 ENFORCEMENT MECHANISMS
```text
PRE-CODING CHECKLIST (must be logged before implementation):
1. Relevant specs consulted: ____________________
2. Files/modules reviewed: ______________________
3. Existing implementations reused or extended: __
4. Conflict resolution summary: _________________
5. Approval/acknowledgement recorded: ___________

Coding may begin only after all five checkpoints are completed.
```

## 📋 Article II: System Integration Standards

### Section 2.1: ThoughtSeed Integration Requirements
**MANDATORY**: All agents MUST follow ThoughtSeed integration protocols:

#### Consciousness Detection Standards
- **REQUIRED**: Use ThoughtSeed service for consciousness detection
- **REQUIRED**: Maintain consciousness state consistency
- **REQUIRED**: Report consciousness levels in all architecture evaluations

#### Active Inference Requirements
- **REQUIRED**: Implement hierarchical belief systems
- **REQUIRED**: Use prediction error minimization
- **REQUIRED**: Maintain belief network coherence

### Section 2.2: Database Abstraction Requirements
**MANDATORY**: All agents MUST enforce constitutional database access patterns:

#### ✅ Graph Database Access Standards (Spec 040 M3 - ENFORCED)
**Effective Date**: 2025-10-07
**Enforcement**: Pre-commit hooks + CI + Linter + Regression tests

- **REQUIRED**: ALL Neo4j access MUST flow through DaedalusGraphChannel
- **REQUIRED**: Use `from daedalus_gateway import get_graph_channel` for graph operations
- **REQUIRED**: Include `caller_service` and `caller_function` parameters for audit trail
- **REQUIRED**: Use Graph Channel operations: `execute_read()`, `execute_write()`, `execute_schema()`

#### 🚫 PROHIBITED ACTIONS
- **NEVER** import neo4j directly in backend/src services
- **NEVER** use `from neo4j import GraphDatabase` or similar direct imports
- **NEVER** create direct Neo4j driver connections
- **NEVER** bypass DaedalusGraphChannel facade

**ONLY EXCEPTION**: The daedalus-gateway repository is the SOLE location allowed to import neo4j.

#### 🔧 ENFORCEMENT MECHANISMS
```python
# ✅ CORRECT: Constitutional compliance
from daedalus_gateway import get_graph_channel

channel = get_graph_channel()
result = await channel.execute_read(
    query="MATCH (n:Concept) RETURN n LIMIT 10",
    caller_service="my_service",
    caller_function="fetch_concepts"
)

# ❌ VIOLATION: Direct neo4j import (BANNED)
from neo4j import GraphDatabase  # This will FAIL pre-commit + CI
driver = GraphDatabase.driver(uri, auth=(user, password))
```

**Enforcement Chain**:
1. **Pre-commit hook**: Blocks commits with direct neo4j imports
2. **CI check**: `.github/workflows/constitutional-compliance.yml` fails builds with violations
3. **Linter**: `.ruff_constitutional_plugin.py` detects banned imports (CONST001/CONST002 errors)
4. **Regression tests**: `tests/governance/test_constitutional_compliance.py` prevents backsliding

**Migration Guide**: `GRAPH_CHANNEL_MIGRATION_QUICK_REFERENCE.md`
**Audit Registry**: `LEGACY_REGISTRY.md`

#### Pipeline Integration
- **REQUIRED**: Use ThoughtSeed-enhanced context engineering
- **REQUIRED**: Maintain river metaphor information flow
- **REQUIRED**: Implement attractor basin dynamics

#### Database Standards
- **REQUIRED**: Use Redis for real-time operations
- **REQUIRED**: Use Neo4j via DaedalusGraphChannel ONLY (see above)
- **REQUIRED**: Maintain data consistency across services

## 📋 Article III: Agent Behavior Standards

### Section 3.1: Communication Protocols
**MANDATORY**: All agents MUST follow communication standards:

#### Status Reporting
- **REQUIRED**: Report system health before major operations
- **REQUIRED**: Announce environment changes
- **REQUIRED**: Coordinate with other active agents

#### Conflict Resolution
- **REQUIRED**: Check for active processes before starting new ones
- **REQUIRED**: Use safe workspaces for parallel development
- **REQUIRED**: Avoid modifying files being worked on by other agents

### Section 3.2: Testing Standards
**MANDATORY**: All agents MUST follow testing protocols:

#### Integration Testing
- **REQUIRED**: Test ThoughtSeed integration before deployment
- **REQUIRED**: Validate consciousness detection functionality
- **REQUIRED**: Verify ASI-Arch pipeline compatibility

#### Environment Validation
- **REQUIRED**: Verify NumPy version compliance
- **REQUIRED**: Test all service dependencies
- **REQUIRED**: Validate database connectivity

## 📋 Article IV: Enforcement and Compliance

### Section 4.1: Compliance Monitoring
**MANDATORY**: All agents MUST implement compliance monitoring:

#### Pre-Operation Checks
```python
# MANDATORY: Include this check in all agent operations
def verify_constitution_compliance():
    import numpy
    assert numpy.__version__.startswith('1.'), "CONSTITUTION VIOLATION: NumPy 2.x detected"
    print("✅ Constitution compliance verified")
```

#### Environment Validation
```bash
# MANDATORY: Run this before any major operation
python -c "
import numpy, torch, transformers
print(f'NumPy: {numpy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
assert numpy.__version__.startswith('1.'), 'CONSTITUTION VIOLATION'
print('✅ All dependencies constitution-compliant')
"
```

### Section 4.2: Violation Reporting
**REQUIRED**: All agents MUST report constitution violations:

#### Violation Types
1. **CRITICAL**: NumPy 2.x installation attempt
2. **HIGH**: Environment modification without isolation
3. **MEDIUM**: Service conflict or interference
4. **LOW**: Testing protocol violations

#### Reporting Protocol
```python
# MANDATORY: Include violation reporting in all agents
def report_violation(violation_type: str, details: str):
    print(f"🚨 CONSTITUTION VIOLATION: {violation_type}")
    print(f"Details: {details}")
    print("📍 Report to system administrator")
```

## 📋 Article V: Amendment Procedures

### Section 5.1: Constitution Updates
**PROCESS**: Constitution amendments require:

1. **Proposal**: Detailed amendment proposal with justification
2. **Review**: Technical review by system administrators
3. **Testing**: Comprehensive testing of proposed changes
4. **Approval**: Unanimous approval from active agents
5. **Implementation**: Gradual rollout with monitoring

### Section 5.2: Emergency Procedures
**PROCESS**: Emergency amendments for critical issues:

1. **Immediate**: Temporary fix implementation
2. **Notification**: Immediate notification to all agents
3. **Documentation**: Detailed documentation of emergency
4. **Review**: Post-emergency review and permanent fix

---

## 🎯 Constitution Summary

### ✅ MANDATORY REQUIREMENTS
- **NumPy**: Always use version < 2.0
- **Environments**: Always use isolation
- **Testing**: Always verify compliance
- **Communication**: Always coordinate with other agents

### 🚫 PROHIBITED ACTIONS
- **Never** install NumPy 2.x
- **Never** modify active development files
- **Never** start conflicting services
- **Never** skip compliance checks

### 📊 COMPLIANCE METRICS
- **NumPy Version**: Must start with "1."
- **Environment Isolation**: Must use virtual environments
- **Service Health**: All services must be healthy
- **Agent Coordination**: Must check for active processes

---

**Status**: ✅ **ACTIVE AND ENFORCED**  
**Compliance**: 🟢 **MANDATORY FOR ALL AGENTS**  
**Updates**: 📋 **Amendment procedures established**

This constitution ensures system stability, prevents compatibility issues, and maintains the integrity of the ASI-Arch ThoughtSeed integration across all agent operations.
