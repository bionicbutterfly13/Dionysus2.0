# 🤖 ASI-Arch Agent Constitution

**Version**: 1.0.0  
**Effective Date**: 2025-09-24  
**Purpose**: Establish permanent standards for agent behavior and system compatibility

## 📋 Article I: Dependency Management

### Section 1.1: NumPy 2.0+ Compatibility Requirements
**MANDATORY**: All agents MUST adhere to the following NumPy 2.0+ compatibility standards:

#### 🚫 PROHIBITED ACTIONS
- **NEVER** install `numpy<2.0` in any environment
- **NEVER** downgrade NumPy without explicit approval
- **NEVER** use legacy NumPy 1.x packages
- **NEVER** install packages that require NumPy 1.x without upgrading them first

#### ✅ REQUIRED ACTIONS
- **ALWAYS** use `pip install "numpy>=2.0"`
- **ALWAYS** pin NumPy version in requirements files: `numpy>=2.0.0`
- **ALWAYS** verify PyTorch + NumPy 2.x compatibility before installing ML packages
- **ALWAYS** use NumPy 2.0+ compatible versions of SentenceTransformers and PyTorch

#### 🔧 ENFORCEMENT MECHANISMS
```bash
# MANDATORY: Check NumPy version before any ML package installation
python -c "import numpy; assert numpy.__version__.startswith('2.'), f'NumPy {numpy.__version__} violates constitution (requires 2.0+)'"

# MANDATORY: Use this exact command for NumPy installation
pip install "numpy>=2.0" --upgrade

# MANDATORY: Verify installation with PyTorch compatibility
python -c "import numpy, torch; print(f'✅ NumPy {numpy.__version__} + PyTorch {torch.__version__} compatible')"
```

### Section 1.2: Environment Isolation Standards
**REQUIRED**: All agents MUST use isolated environments:

#### Virtual Environment Requirements
```bash
# MANDATORY: Create isolated environment
python -m venv asi-arch-frozen-env
source asi-arch-frozen-env/bin/activate

# MANDATORY: Install NumPy 2.0+ first
pip install "numpy>=2.0" --upgrade

# MANDATORY: Install other packages only after NumPy compliance
pip install -r requirements-frozen.txt
```

#### Binary Distribution Standards
- **REQUIRED**: Use pre-compiled binaries when available
- **REQUIRED**: Avoid source compilation that might upgrade NumPy
- **REQUIRED**: Use conda-forge for ML packages when possible

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

### Section 2.2: ASI-Arch Pipeline Standards
**MANDATORY**: All agents MUST maintain ASI-Arch compatibility:

#### Pipeline Integration
- **REQUIRED**: Use ThoughtSeed-enhanced context engineering
- **REQUIRED**: Maintain river metaphor information flow
- **REQUIRED**: Implement attractor basin dynamics

#### Database Standards
- **REQUIRED**: Use Redis for real-time operations
- **REQUIRED**: Use Neo4j for knowledge graph operations
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
    assert numpy.__version__.startswith('2.'), "CONSTITUTION VIOLATION: NumPy 1.x detected"
    print("✅ Constitution compliance verified - NumPy 2.0+ active")
```

#### Environment Validation
```bash
# MANDATORY: Run this before any major operation
python -c "
import numpy, torch, transformers
print(f'NumPy: {numpy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
assert numpy.__version__.startswith('2.'), 'CONSTITUTION VIOLATION: NumPy 1.x detected'
print('✅ All dependencies constitution-compliant')
"
```

### Section 4.2: Violation Reporting
**REQUIRED**: All agents MUST report constitution violations:

#### Violation Types
1. **CRITICAL**: NumPy 1.x installation attempt
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
- **NumPy**: Always use version >= 2.0
- **Environments**: Always use isolation
- **Testing**: Always verify compliance
- **Communication**: Always coordinate with other agents

### 🚫 PROHIBITED ACTIONS
- **Never** install NumPy 1.x
- **Never** modify active development files
- **Never** start conflicting services
- **Never** skip compliance checks

### 📊 COMPLIANCE METRICS
- **NumPy Version**: Must start with "2."
- **Environment Isolation**: Must use virtual environments
- **Service Health**: All services must be healthy
- **Agent Coordination**: Must check for active processes

---

**Status**: ✅ **ACTIVE AND ENFORCED**  
**Compliance**: 🟢 **MANDATORY FOR ALL AGENTS**  
**Updates**: 📋 **Amendment procedures established**

This constitution ensures system stability, prevents compatibility issues, and maintains the integrity of the ASI-Arch ThoughtSeed integration across all agent operations.
