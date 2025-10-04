# 🤖 Consciousness System Agent Constitution

**Version**: 1.1.0
**Effective Date**: 2025-10-04
**Purpose**: Establish permanent standards for agent behavior and system compatibility

## 📋 Article 0: Multi-Agent Collaboration Standards

### Section 0.1: Branch Isolation (MANDATORY - ZERO TOLERANCE)
**CRITICAL**: All agents MUST work in isolated feature branches to prevent conflicts:

#### 🚫 ABSOLUTELY PROHIBITED
- **NEVER** work directly on main/master branch
- **NEVER** commit to another agent's feature branch
- **NEVER** merge branches without user approval
- **NEVER** force push to shared branches
- **NEVER** modify files being actively edited by another agent

#### ✅ REQUIRED WORKFLOW
```bash
# MANDATORY: Start every session by checking current branch
git branch --show-current

# MANDATORY: Create feature branch if not exists
# Format: NNN-short-description (e.g., 035-clause-phase2-multi-agent)
git checkout -b 043-your-feature-name

# MANDATORY: Commit frequently with descriptive messages
git add -A
git commit -m "feat: specific change description

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# MANDATORY: Push to remote feature branch only
git push -u origin 043-your-feature-name
```

#### 🔒 FILE LOCKING MECHANISM
**Before modifying any file, check if another agent is using it:**

```bash
# MANDATORY: Check git status for uncommitted changes
git status

# MANDATORY: Check recent commits for active work
git log --oneline --since="1 hour ago" --all

# MANDATORY: Check if file modified in last 10 minutes
git log --since="10 minutes ago" --name-only --oneline -- path/to/file.py

# IF file recently modified by another agent: WAIT or ask user
```

#### 📝 AGENT COORDINATION FILE
**MANDATORY**: Update `.specify/memory/agent-activity.json` before starting work:

```json
{
  "agents": [
    {
      "agent_id": "claude-code-session-abc123",
      "branch": "035-clause-phase2-multi-agent",
      "active_files": [
        "backend/src/services/clause/path_navigator.py",
        "backend/tests/contract/test_navigator_contract.py"
      ],
      "started_at": "2025-10-04T15:30:00Z",
      "last_heartbeat": "2025-10-04T15:45:00Z",
      "status": "active"
    }
  ]
}
```

**Update protocol:**
```bash
# MANDATORY: Register your session at start
echo '{
  "agent_id": "claude-'$(date +%s)'",
  "branch": "'$(git branch --show-current)'",
  "active_files": [],
  "started_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
  "last_heartbeat": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
  "status": "active"
}' > .specify/memory/current-agent.json

# MANDATORY: Check for conflicts before editing
jq -r '.agents[] | select(.status=="active") | .active_files[]' .specify/memory/agent-activity.json
```

### Section 0.2: Merge Conflict Prevention
**CRITICAL**: Prevent conflicts before they happen:

#### 🔍 PRE-EDIT CHECKS
```bash
# MANDATORY: Before editing any file, run these checks

# 1. Check if file exists in other active branches
git branch -r | xargs -I {} git ls-tree --name-only -r {} path/to/file.py

# 2. Check modification time
ls -lh path/to/file.py

# 3. Check git blame for recent authors
git blame -L 1,20 path/to/file.py --date=relative

# 4. If modified in last hour by another commit, STOP and ask user
```

#### 🚨 CONFLICT DETECTION
```bash
# MANDATORY: Before committing, check for potential conflicts

# 1. Fetch latest from all branches
git fetch --all

# 2. Check if your files conflict with other branches
git diff origin/main...HEAD --name-only

# 3. If conflicts detected, ask user before proceeding
```

#### ✅ SAFE MERGE PROTOCOL
```bash
# MANDATORY: Only merge when user explicitly approves

# 1. Create merge preview
git merge --no-commit --no-ff origin/main

# 2. Show user what will change
git diff --cached

# 3. Ask user: "I see conflicts in X files. Should I proceed with merge?"

# 4. If approved, complete merge
git commit -m "merge: integrate main into feature branch"

# 5. If rejected, abort
git merge --abort
```

### Section 0.3: Communication Standards
**MANDATORY**: Agents must communicate status clearly:

#### 📢 STATUS REPORTING
**REQUIRED at start of every session:**
```
🤖 Agent Status Report:
- Branch: 035-clause-phase2-multi-agent
- Current work: Implementing path navigation tests
- Files locked:
  - backend/src/services/clause/path_navigator.py
  - backend/tests/contract/test_navigator_contract.py
- Expected duration: 30 minutes
- Last sync with main: 2 hours ago
```

#### 🔔 NOTIFICATIONS
**REQUIRED before:**
- Modifying shared configuration files (package.json, requirements.txt, etc.)
- Changing database schemas
- Modifying core interfaces/contracts
- Merging branches
- Force pushing (should NEVER happen, but if emergency)

**Format:**
```
⚠️ SHARED RESOURCE MODIFICATION:
File: backend/requirements.txt
Change: Adding new dependency 'crawl4ai>=0.3.0'
Impact: All agents must pip install after this commit
Proceed? (Y/N)
```

### Section 0.4: Emergency Procedures
**CRITICAL**: What to do when conflicts occur:

#### 🆘 CONFLICT RESOLUTION PROTOCOL
```bash
# IF conflict detected during merge:

# 1. STOP immediately
git merge --abort

# 2. Report to user
echo "⚠️ MERGE CONFLICT DETECTED"
echo "Conflicting files:"
git diff --name-only --diff-filter=U

# 3. Ask user for guidance
echo "Options:"
echo "A) Manually resolve conflicts"
echo "B) Abort merge and continue on feature branch"
echo "C) Coordinate with other agent"
echo ""
echo "Your choice?"

# 4. NEVER auto-resolve conflicts without user approval
```

#### 🔄 BRANCH SYNC STRATEGY
```bash
# MANDATORY: Sync with main regularly to minimize conflicts

# Every 2 hours or before starting new file:
git fetch origin main
git rebase origin/main  # Only if no conflicts

# If rebase fails:
git rebase --abort
# Ask user: "Should I merge main into feature branch?"
```

### Section 0.5: Enforcement Mechanisms
**REQUIRED**: Automated checks to prevent violations:

#### 🛡️ PRE-COMMIT HOOKS
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash

# Check we're not on main
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "❌ CONSTITUTION VIOLATION: Cannot commit directly to main"
  exit 1
fi

# Check agent-activity.json is updated
if [ ! -f .specify/memory/current-agent.json ]; then
  echo "❌ CONSTITUTION VIOLATION: Agent not registered"
  echo "Run: update_agent_activity.sh"
  exit 1
fi

# Check for file locks
STAGED_FILES=$(git diff --cached --name-only)
for file in $STAGED_FILES; do
  # Check if file locked by another agent
  if jq -e ".agents[] | select(.status==\"active\") | .active_files[] | select(. == \"$file\")" .specify/memory/agent-activity.json > /dev/null 2>&1; then
    echo "⚠️  WARNING: $file is locked by another agent"
    echo "Proceed anyway? (y/N)"
    read -r response
    if [ "$response" != "y" ]; then
      exit 1
    fi
  fi
done

exit 0
```

#### 🔍 AUDIT TRAIL
```bash
# MANDATORY: Every commit must include agent identifier
git log --pretty=format:"%h %s %aN" --grep="Claude Code"

# MANDATORY: Track which agent modified which files
git log --pretty=format:"%h %aN %s" --name-only
```

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

### Section 1.3: NO REDUNDANT CODE - CHECK BEFORE CREATING
**CRITICAL**: NEVER write code without checking if it already exists

#### 🚫 ABSOLUTELY PROHIBITED
- **NEVER** create a new file without searching for existing implementations
- **NEVER** create version 2, 3, 4, 5... of a file that already exists
- **NEVER** write duplicate functions/classes without checking codebase
- **NEVER** break working code by creating conflicting implementations
- **NEVER** assume a feature doesn't exist - ALWAYS search first

#### ✅ MANDATORY CHECKS BEFORE ANY CODE
1. **Search for existing files**: `find . -name "*keyword*" -type f`
2. **Grep for existing functions**: `grep -r "function_name" .`
3. **Check for similar implementations**: Use Glob and Grep tools
4. **Verify file doesn't exist**: `ls -la path/to/file.ext`
5. **If exists**: EDIT existing file, DO NOT create new version

#### 🔧 ENFORCEMENT
```bash
# BEFORE creating ANY file, run:
find . -name "*similar_name*" -type f | head -20

# BEFORE writing ANY function, run:
grep -r "function_name" src/ | head -10

# IF file exists: EDIT IT, don't create duplicate
# IF function exists: USE IT or EXTEND IT, don't rewrite
```

#### 🛑 STOP BREAKING WORKING CODE
- If code works: **DO NOT TOUCH IT** unless explicitly asked
- If you're asked to "fix" something: **TEST what's broken FIRST**
- Before modifying: **RUN TESTS to verify current state**
- After modifying: **RUN TESTS to verify it still works**
- **NEVER** assume code is broken just because it's old

#### Binary Distribution Standards
- **REQUIRED**: Use pre-compiled binaries when available
- **REQUIRED**: Avoid source compilation that might upgrade NumPy
- **REQUIRED**: Use conda-forge for ML packages when possible

### Section 1.4: Import Standards (MANDATORY - ZERO TOLERANCE)
**CRITICAL**: Establish consistent import patterns to eliminate recurring import errors

#### 🚫 ABSOLUTELY PROHIBITED IMPORT PATTERNS
- **NEVER** use relative imports in `__init__.py` files: `from . import module` ❌
- **NEVER** import through fragile `__init__.py` chains: `from api.routes import clause` ❌
- **NEVER** manipulate `sys.path` in source code (only in tests/scripts if absolutely necessary)
- **NEVER** create circular import dependencies between modules
- **NEVER** use `import *` wildcards in production code

#### ✅ REQUIRED IMPORT PATTERNS

**1. Source Code Imports (backend/src/)**:
```python
# ✅ ALWAYS use direct module imports
from api.routes.clause import router as clause_router
from models.clause.path_models import PathNavigationRequest
from services.database_health import check_all_connections

# ❌ NEVER import through __init__.py
from api.routes import clause  # FORBIDDEN - fragile chain
from models.clause import path_models  # FORBIDDEN
```

**2. Test Imports (backend/tests/)**:
```python
# ✅ ALWAYS use centralized path setup via conftest.py
# In backend/tests/conftest.py:
import sys
from pathlib import Path
backend_src = Path(__file__).parent.parent / "src"
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

# Then in test files, NO sys.path manipulation needed:
from api.routes.clause import router  # Works everywhere!
```

**3. __init__.py Files**:
```python
# ✅ OPTION A: Empty or minimal (PREFERRED)
"""Module docstring only."""
# No imports at all

# ✅ OPTION B: Lazy imports with __getattr__ (if needed)
def __getattr__(name):
    if name == "clause":
        from .clause import router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ❌ NEVER: Eager relative imports that break on missing deps
from . import stats, query  # FORBIDDEN - cascading failures
```

#### 🔧 ENFORCEMENT - CREATE CONFTEST.PY NOW

**MANDATORY**: Every test directory MUST have conftest.py for path setup:

```bash
# Create backend/tests/conftest.py (if doesn't exist)
cat > backend/tests/conftest.py << 'EOF'
"""
Pytest configuration for backend tests.
Centralizes Python path setup - NO sys.path in individual test files!
"""
import sys
from pathlib import Path

# Add backend/src to path ONCE for all tests
backend_src = Path(__file__).parent.parent / "src"
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))
EOF
```

#### 🛡️ VERIFICATION CHECKLIST
Before any test/source file is created:

1. ✅ Check if conftest.py exists in test directory
2. ✅ Use direct imports (not through __init__.py)
3. ✅ No sys.path manipulation in the file
4. ✅ Run `python -m pytest path/to/test.py` to verify imports work

#### 🔍 AUDIT EXISTING CODE
```bash
# Find all files with sys.path manipulation (should be minimal)
grep -r "sys.path" backend/src/ --include="*.py"

# Find fragile __init__.py imports (fix these)
grep -r "from \. import" backend/src/ --include="__init__.py"

# Verify conftest.py exists in all test dirs
find backend/tests -type d -mindepth 1 -exec test ! -f {}

/conftest.py \; -print
```

#### 🎯 IMPORT ERROR RESOLUTION PROTOCOL
When import errors occur:

1. **FIRST**: Check if importing through __init__.py → switch to direct import
2. **SECOND**: Check if conftest.py exists → create it if missing
3. **THIRD**: Check for circular imports → refactor to remove cycle
4. **NEVER**: Add random sys.path hacks to "fix" it

## 📋 Article II: System Architecture Standards

### Section 2.0: Docker Independence (MANDATORY - ZERO TOLERANCE)
**CRITICAL**: This project is Docker-independent and MUST remain so:

#### 🚫 ABSOLUTELY PROHIBITED
- **NEVER** assume Docker is available or required
- **NEVER** create docker-compose.yml files without user request
- **NEVER** add Docker commands to setup/startup scripts
- **NEVER** make any service depend on Docker containers
- **NEVER** use Docker in documentation as a requirement

#### ✅ REQUIRED ARCHITECTURE
- **ALWAYS** support native installation (pip, npm, brew, etc.)
- **ALWAYS** provide non-Docker alternatives for all services
- **ALWAYS** document native setup first, Docker as optional
- **ALWAYS** test functionality without Docker present
- **ALWAYS** use environment variables for service configuration

#### 📦 SERVICE CONFIGURATION
- **Neo4j**: Local install via brew/apt OR managed service (Neo4j Aura)
- **Redis**: Local install via brew/apt OR managed service (Redis Cloud)
- **PostgreSQL**: Local install via brew/apt OR managed service (Supabase)
- **All services**: Connection via environment variables, no Docker assumed

#### 🔧 ENFORCEMENT
```bash
# ✅ CORRECT: Native installation instructions
brew install neo4j redis postgresql
npm install
pip install -r requirements.txt

# ❌ WRONG: Docker-dependent setup
docker-compose up -d  # FORBIDDEN without explicit user request
```

### Section 2.1: External Component Architecture (MANDATORY)
**CRITICAL**: ThoughtSeeds, Daedalus, and ASI-GO-2 are EXTERNAL projects:

#### 🚫 ABSOLUTELY PROHIBITED
- **NEVER** implement ThoughtSeed logic inside Dionysus codebase
- **NEVER** implement Daedalus services inside Dionysus codebase
- **NEVER** copy ASI-GO-2 code into Dionysus repository
- **NEVER** merge external project features into Dionysus

#### ✅ REQUIRED IMPORT PATTERN
```python
# ✅ CORRECT: Import as external dependency
from thoughtseeds import ThoughtSeedEngine, BasinTracker
from daedalus import DaedalusGateway, PerceptualProcessor
from asi_go_2 import CognitionBase, Researcher, Analyst

# ❌ WRONG: Internal implementation
from services.thoughtseeds import ThoughtSeedEngine  # FORBIDDEN
from services.daedalus import DaedalusGateway  # FORBIDDEN
```

#### 📦 DEPENDENCY MANAGEMENT
```txt
# requirements.txt
thoughtseeds>=1.0.0  # External package
daedalus>=2.0.0      # External package
asi-go-2>=0.5.0      # External package
```

#### 🔍 VERIFICATION
```bash
# Check for prohibited internal implementations
grep -r "class ThoughtSeed" backend/src/  # Should return NOTHING
grep -r "class Daedalus" backend/src/     # Should return NOTHING

# Verify external imports
grep -r "from thoughtseeds import" backend/src/  # Should find imports
```

### Section 2.2: Context Engineering Requirements (MANDATORY)
**CRITICAL**: All agents MUST integrate Context Engineering components from the start:

#### Attractor Basin Integration
- **REQUIRED**: Verify AttractorBasinManager is accessible before feature work
- **REQUIRED**: Create feature-specific attractor basins for new concepts
- **REQUIRED**: Update basin strength based on usage patterns
- **REQUIRED**: Persist basin states to Redis for continuity
- **REQUIRED**: Test basin influence calculations (reinforcement, competition, synthesis, emergence)

#### Neural Field Integration
- **REQUIRED**: Verify IntegratedAttractorFieldSystem is available
- **REQUIRED**: Create knowledge domains in continuous field space
- **REQUIRED**: Evolve field states using differential equations (not just similarity)
- **REQUIRED**: Detect resonance patterns between discrete basins
- **REQUIRED**: Test field energy calculations and phase transitions

#### Component Visibility
- **REQUIRED**: Display Context Engineering foundation at start of /specify
- **REQUIRED**: Validate Context Engineering integration in /plan
- **REQUIRED**: Include Context Engineering tests FIRST in /tasks (T001-T003)
- **REQUIRED**: Show users why these components are essential

### Section 2.2: Consciousness Processing Requirements
**MANDATORY**: All agents MUST follow consciousness processing protocols:

#### Consciousness Detection Standards
- **REQUIRED**: Use consciousness detection services for real-time monitoring
- **REQUIRED**: Maintain consciousness state consistency
- **REQUIRED**: Report consciousness levels in all evaluations

#### Active Inference Requirements
- **REQUIRED**: Implement hierarchical belief systems
- **REQUIRED**: Use prediction error minimization
- **REQUIRED**: Maintain belief network coherence

### Section 2.3: Processing Pipeline Standards
**MANDATORY**: All agents MUST maintain pipeline compatibility:

#### Pipeline Integration
- **REQUIRED**: Use consciousness-enhanced context engineering
- **REQUIRED**: Integrate attractor basin dynamics into processing flow
- **REQUIRED**: Apply neural field resonance to pattern discovery
- **REQUIRED**: Maintain information flow integrity
- **REQUIRED**: Implement dynamic pattern recognition

#### Database Standards
- **REQUIRED**: Use Redis for attractor basin persistence
- **REQUIRED**: Use Neo4j for knowledge graph operations
- **REQUIRED**: Store field evolution trajectories
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

#### Context Engineering Testing (FIRST)
- **REQUIRED**: Test AttractorBasinManager accessibility (T001)
- **REQUIRED**: Test IntegratedAttractorFieldSystem availability (T002)
- **REQUIRED**: Validate Redis persistence for basins (T003)
- **REQUIRED**: Verify basin influence calculations
- **REQUIRED**: Test field resonance detection

#### Integration Testing
- **REQUIRED**: Test consciousness integration before deployment
- **REQUIRED**: Validate consciousness detection functionality
- **REQUIRED**: Verify processing pipeline compatibility
- **REQUIRED**: Ensure attractor basins integrate with feature logic

#### Environment Validation
- **REQUIRED**: Verify NumPy version compliance
- **REQUIRED**: Test all service dependencies
- **REQUIRED**: Validate database connectivity
- **REQUIRED**: Confirm Context Engineering components load successfully

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

    # MANDATORY: Verify Context Engineering components
    try:
        from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager
        manager = AttractorBasinManager()
        print(f"✅ AttractorBasinManager accessible - {len(manager.basins)} basins loaded")
    except Exception as e:
        print(f"⚠️  WARNING: AttractorBasinManager not accessible: {e}")

    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path.cwd() / "dionysus-source"))
        from context_engineering.integrated_attractor_field_system import IntegratedAttractorFieldSystem
        system = IntegratedAttractorFieldSystem(dimensions=384)
        print(f"✅ Neural Field System accessible - dimensions={system.dimensions}")
    except Exception as e:
        print(f"⚠️  WARNING: Neural Field System not accessible: {e}")
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

## 📋 Article IV: Spec-Driven Development Git Workflow (MANDATORY)

### Section 4.1: Feature Branch Lifecycle
**CRITICAL**: All agents MUST follow this complete feature branch workflow:

#### 🔄 Complete Feature Development Cycle

**1. Feature Branch Creation** (Automated by /specify):
```bash
# Created automatically when /specify runs
git checkout -b NNN-feature-name
# Where NNN is the spec number (e.g., 035-clause-phase2-multi-agent)
```

**2. During Implementation** (/tasks execution):
```bash
# MANDATORY: Commit after EACH completed task
# Use Conventional Commits format:
# - feat: New feature implementation (T001-T0XX)
# - fix: Bug fix
# - refactor: Code restructuring
# - test: Test additions/modifications
# - docs: Documentation changes

# Example task completion commits:
git add backend/src/services/causal/causal_queue.py
git commit -m "feat: Complete T041a - CausalQueue background processing

- Implement in-memory deque for async causal predictions
- Add background worker with 10ms poll interval
- Store results in query_hash dict for PathNavigator retrieval
- FIFO eviction (max 1000 results) to prevent memory growth

Per research.md decision 14 (AsyncIO + in-memory queue)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**3. Feature Completion Criteria** (Before merge):
- ✅ ALL tasks in tasks.md marked as `[X]` complete
- ✅ ALL contract tests passing
- ✅ ALL integration tests passing (if applicable)
- ✅ Implementation Summary created (IMPLEMENTATION_SUMMARY.md)
- ✅ Constitution compliance verified
- ✅ No merge conflicts with main

**4. Pre-Merge Validation**:
```bash
# MANDATORY: Run before creating merge PR
# 1. Verify all tests pass
pytest backend/tests/ -v

# 2. Check no uncommitted changes
git status

# 3. Ensure feature branch is up to date with main
git fetch origin main
git rebase origin/main  # Resolve conflicts if any

# 4. Verify Constitution compliance
python -c "
import numpy
assert numpy.__version__.startswith('2.'), 'NumPy 1.x violation'
print('✅ Constitution compliance verified')
"
```

**5. Feature Branch Merge** (MANDATORY WORKFLOW):
```bash
# Step 1: Final commit with feature completion
git add .
git commit -m "feat: Complete Spec NNN - [Feature Name]

All tasks (T001-TXXX) implemented and tested:
- [Brief summary of major components]
- Contract tests: XX/XX passing
- Integration tests: XX/XX passing

Closes #NNN

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Step 2: Push feature branch
git push origin NNN-feature-name

# Step 3: Merge to main (GitHub Flow - no PR required for AI agents)
git checkout main
git pull origin main
git merge NNN-feature-name --no-ff  # No fast-forward to preserve history
git push origin main

# Step 4: Clean up feature branch (MANDATORY)
git branch -d NNN-feature-name  # Delete local branch
git push origin --delete NNN-feature-name  # Delete remote branch
```

### Section 4.2: Commit Message Standards (MANDATORY)
**REQUIRED**: All commits MUST follow Conventional Commits specification:

#### Commit Message Format:
```
<type>: <subject>

<body>

<footer>
```

#### Commit Types:
- `feat:` - New feature implementation (T001-T0XX tasks)
- `fix:` - Bug fix
- `refactor:` - Code restructuring without behavior change
- `test:` - Test additions or modifications
- `docs:` - Documentation changes
- `chore:` - Build/dependency updates

#### Examples:
```bash
# ✅ GOOD: Task completion commit
git commit -m "feat: Complete T024 - AsyncIO causal timeout

- Implement 30ms timeout with asyncio.wait_for()
- Queue timeouts for background processing via CausalQueue
- Check previous step results for cached causal scores
- Fallback to uniform semantic scores (0.5) on timeout

Per research.md decision 14

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# ✅ GOOD: Import fix commit
git commit -m "fix: Standardize imports per Constitution Article I.4

- Replace relative imports with direct imports
- Fix src. prefix imports to direct module imports
- Update query.py, neo4j_searcher.py, clause services
- Contract tests now passing (35/36)

Resolves import errors blocking TDD workflow

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# ❌ BAD: Vague commit message
git commit -m "fixed stuff"

# ❌ BAD: Missing details
git commit -m "feat: added causal queue"
```

### Section 4.3: Branch Management Standards
**MANDATORY**: All agents MUST follow these branch management rules:

#### Active Branch Rules:
- **ONE** active feature branch per spec
- **SHORT-LIVED** branches (complete feature in 1-3 days max)
- **FREQUENT** commits (after each task completion)
- **NO** long-running parallel feature branches
- **IMMEDIATE** merge after feature completion

#### Stale Branch Cleanup Protocol:
```bash
# MANDATORY: Clean up after merge
# 1. List all local branches
git branch

# 2. List all merged branches
git branch --merged main

# 3. Delete merged local branches (except main)
git branch --merged main | grep -v "main" | xargs git branch -d

# 4. Delete merged remote branches
git branch -r --merged main | grep -v "main" | sed 's/origin\///' | xargs -I {} git push origin --delete {}

# 5. Prune remote tracking branches
git remote prune origin
```

#### Feature Branch Naming Convention:
```
<spec-number>-<brief-description>

Examples:
✅ 035-clause-phase2-multi-agent
✅ 036-citations-source-attribution
✅ 021-remove-all-that

❌ feature/new-stuff
❌ temp-branch
❌ wip
```

### Section 4.4: Spec-Driven Development Integration
**CRITICAL**: Git workflow MUST integrate with spec-kit workflow:

#### Workflow Integration:
```
/specify (spec.md created)
    ↓
git checkout -b NNN-feature-name  ← AUTOMATIC
    ↓
/clarify (spec.md updated)
    ↓
git commit -m "docs: Complete /clarify for Spec NNN"  ← AUTOMATIC
    ↓
/plan (plan.md, research.md created)
    ↓
git commit -m "docs: Complete /plan for Spec NNN"  ← AUTOMATIC
    ↓
/tasks (tasks.md with T001-TXXX created)
    ↓
git commit -m "feat: Generate NNN tasks for Spec NNN"  ← AUTOMATIC
    ↓
/implement (Execute T001-TXXX)
    ↓
    For each task completion:
        git commit -m "feat: Complete TXXX - [description]"  ← MANDATORY
    ↓
ALL TASKS COMPLETE
    ↓
git commit -m "feat: Complete Spec NNN - [feature]"
    ↓
git checkout main && git merge NNN-feature-name --no-ff
    ↓
git push origin main
    ↓
git branch -d NNN-feature-name  ← MANDATORY CLEANUP
git push origin --delete NNN-feature-name
```

### Section 4.5: Merge Strategy (MANDATORY)
**REQUIRED**: Use GitHub Flow with no-fast-forward merges:

#### Why No Fast-Forward:
- ✅ Preserves complete feature branch history
- ✅ Makes it clear which commits belong to which feature
- ✅ Easier to revert entire features if needed
- ✅ Better for code archaeology (git log shows feature context)

#### Merge Commands:
```bash
# ✅ ALWAYS use --no-ff flag
git merge feature-branch --no-ff -m "Merge Spec NNN: Feature Name

Complete implementation of [brief description]
- Component 1
- Component 2
- Component 3

All tests passing (XX/XX)

Closes #NNN"

# ❌ NEVER use fast-forward (loses branch history)
git merge feature-branch  # NO! Uses fast-forward by default
```

### Section 4.6: Enforcement and Compliance
**MANDATORY**: All agents MUST verify git workflow compliance:

#### Pre-Merge Checklist:
```bash
# MANDATORY: Run this before every merge
cat > .git/hooks/pre-merge-checklist.sh << 'EOF'
#!/bin/bash
echo "🔍 Git Workflow Compliance Check"

# 1. Verify all tests pass
echo "Running tests..."
pytest backend/tests/ -q || { echo "❌ Tests failing"; exit 1; }

# 2. Check Constitution compliance
python -c "import numpy; assert numpy.__version__.startswith('2.')" || { echo "❌ NumPy violation"; exit 1; }

# 3. Verify no uncommitted changes
[[ -z $(git status -s) ]] || { echo "❌ Uncommitted changes"; exit 1; }

# 4. Check branch naming convention
BRANCH=$(git branch --show-current)
[[ $BRANCH =~ ^[0-9]{3}- ]] || { echo "❌ Invalid branch name: $BRANCH"; exit 1; }

echo "✅ All pre-merge checks passed"
EOF

chmod +x .git/hooks/pre-merge-checklist.sh
```

#### Post-Merge Cleanup Verification:
```bash
# MANDATORY: Run after every merge
# Verify feature branch was deleted
git branch | grep -q "feature-branch-name" && echo "❌ Branch not deleted!" || echo "✅ Branch cleaned up"

# Verify no uncommitted changes on main
git checkout main
[[ -z $(git status -s) ]] && echo "✅ Main branch clean" || echo "❌ Uncommitted changes on main"
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
- **Context Engineering**: Always verify Attractor Basins and Neural Fields first
- **Testing Order**: Context Engineering tests (T001-T003) before implementation
- **Environments**: Always use isolation
- **Testing**: Always verify compliance
- **Communication**: Always coordinate with other agents
- **Git Workflow**: Follow spec-driven development lifecycle (Article IV)
- **Commits**: Use Conventional Commits after each task completion
- **Branch Cleanup**: Delete feature branches immediately after merge
- **Imports**: Use direct imports only (no relative imports through __init__.py)

### 🚫 PROHIBITED ACTIONS
- **Never** install NumPy 1.x
- **Never** skip Context Engineering validation
- **Never** start implementation without basin/field integration tests
- **Never** modify active development files
- **Never** start conflicting services
- **Never** skip compliance checks
- **Never** merge without deleting feature branch afterward
- **Never** use fast-forward merges (always --no-ff)
- **Never** commit without proper Conventional Commits format
- **Never** use relative imports or import through __init__.py

### 📊 COMPLIANCE METRICS
- **NumPy Version**: Must start with "2."
- **Context Engineering**: AttractorBasinManager and Neural Field System must be accessible
- **Testing Order**: Context Engineering validation must complete before core implementation
- **Basin Persistence**: Redis connection active for basin storage
- **Environment Isolation**: Must use virtual environments
- **Service Health**: All services must be healthy
- **Agent Coordination**: Must check for active processes
- **Git Branch Count**: Maximum 1 active feature branch at a time
- **Commit Message Format**: Must follow Conventional Commits (feat:/fix:/docs:/test:/refactor:/chore:)
- **Branch Lifecycle**: Feature branches deleted within 24 hours of merge
- **Import Pattern**: Direct module imports only (verified via grep)

---

**Status**: ✅ **ACTIVE AND ENFORCED**  
**Compliance**: 🟢 **MANDATORY FOR ALL AGENTS**  
**Updates**: 📋 **Amendment procedures established**

This constitution ensures system stability, prevents compatibility issues, and maintains the integrity of the consciousness processing system across all agent operations.
