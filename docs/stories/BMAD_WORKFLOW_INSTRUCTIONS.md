# BMAD v6-alpha Workflow Instructions for Flux UI Hardening

## Story: flux-ui-hardening.md

This document provides the complete BMAD v6-alpha workflow commands to execute the Flux UI hardening story using the BMad Method Module (BMM).

## Prerequisites

1. **Feature Branch**: `feature/flux-ui-hardening-bmad` (already created)
2. **Story File**: `docs/stories/flux-ui-hardening.md` (already exists)
3. **BMAD v6-alpha Installed**: Version 6.0.0-alpha.0
4. **BMM Agents Required**:
   - SM (Scrum Master)
   - DEV (Developer)
   - SR (Senior Reviewer)

## BMAD v6-alpha Workflow

BMAD v6-alpha uses the **4-Phase Scale-Adaptive Workflow**. For this UI hardening task (Level 1 complexity - focused bug fix/cleanup), we'll use the **Implementation Phase** directly.

### Phase 4: Implementation Phase

#### Step 1: SM - Create Story

The Scrum Master creates and prepares the story for development.

**Command:**
```bash
bmad sm create-story
```

**In the SM chat session:**
- Point SM to `docs/stories/flux-ui-hardening.md`
- SM will validate story structure and acceptance criteria
- SM will ensure story is ready for implementation

---

#### Step 2: SM - Generate Story Context

**NEW in v6**: Story-context provides just-in-time expertise injection for the developer.

**Command:**
```bash
bmad sm story-context
```

**Expected Output:**
- SM generates specialized technical context for this story
- Context includes relevant patterns, dependencies, and guidance
- Prepares optimized context for DEV agent

---

#### Step 3: DEV - Implement Story

The Developer implements the story using the generated context.

**Command:**
```bash
bmad dev dev-story
```

**Expected Actions:**
- DEV reads story and injected context
- DEV implements all tasks:
  1. Remove unused variables/components (ESLint)
  2. Replace `any` types with explicit types
  3. Fix hook dependency arrays (`useEffect`, `useCallback`)
  4. Add/repair Jest unit tests
  5. Document follow-up issues
- DEV ensures `npm run lint` passes with no warnings
- DEV updates story with implementation notes

---

#### Step 4: SR - Review Implementation

The Senior Reviewer validates code quality and adherence to standards.

**Command:**
```bash
bmad sr review-story
```

**Expected Actions:**
- SR performs code review against acceptance criteria:
  1. ✅ `npm run lint` passes with no warnings
  2. ✅ `npm test` (Jest) executes successfully
  3. ✅ Key Playwright smoke flows pass
  4. ✅ Debug/Document pages load without console errors
  5. ✅ TypeScript types are properly defined
  6. ✅ Hook dependencies are correct
- SR documents any issues found
- SR approves story or requests corrections

---

#### Step 5: Course Correction (If Needed)

If SR finds issues, use the correct-course workflow.

**Command:**
```bash
bmad dev correct-course
```

**Expected Actions:**
- DEV addresses SR feedback
- DEV re-runs tests and validation
- DEV updates story with corrections

---

## Final Steps

Once SR approval is complete:

1. **Update Project Log**:
   ```bash
   echo "## $(date +%Y-%m-%d) - Flux UI Hardening Complete (BMAD v6-alpha)" >> docs/flux-project-log.md
   echo "- Achieved lint-zero status via BMAD v6-alpha workflow" >> docs/flux-project-log.md
   echo "- Added smoke test coverage" >> docs/flux-project-log.md
   echo "- Fixed ESLint warnings (unused vars, any types, hook deps)" >> docs/flux-project-log.md
   echo "- SR review passed all quality gates" >> docs/flux-project-log.md
   ```

2. **Merge Feature Branch**:
   ```bash
   git add .
   git commit -m "feat: Flux UI hardening - BMAD v6-alpha workflow execution

   - Lint-zero achieved (no ESLint warnings)
   - TypeScript types properly defined
   - Hook dependencies corrected
   - Jest unit tests passing
   - SR review approved"
   git checkout main
   git merge feature/flux-ui-hardening-bmad
   ```

## BMAD v6-alpha Installation

If BMAD v6-alpha is not installed, run:

```bash
cd /path/to/Dionysus-2.0
npm run install:bmad
```

Follow the installer prompts to configure BMad for your project.

## v6-alpha Workflow Reference

- **[BMM Module README](../../BMAD-METHOD/src/modules/bmm/README.md)** - BMad Method overview
- **[v6 Workflows Guide](../../BMAD-METHOD/src/modules/bmm/workflows/README.md)** - Complete workflow documentation
- **[Scale Levels](../../BMAD-METHOD/src/modules/bmm/README.md#scale-levels)** - Understanding project complexity levels

## Notes

- BMAD v6-alpha uses **CLI commands** (`bmad sm create-story`) instead of chat commands
- Each agent maintains state through workflow files in `bmad/` directory
- Story context is dynamically generated for each implementation
- SR replaces the separate QA role from v4

## Alternative: Manual Execution

If you prefer to execute the work manually (without BMAD agents):

1. Follow the task list in `docs/stories/flux-ui-hardening.md`
2. Run `npm run lint` to identify issues
3. Fix each category:
   - Unused variables: Remove or prefix with `_`
   - `any` types: Add proper TypeScript types
   - Hook deps: Add missing dependencies or use ESLint disable with justification
4. Run `npm test` to verify tests pass
5. Document any deferred work as new stories
