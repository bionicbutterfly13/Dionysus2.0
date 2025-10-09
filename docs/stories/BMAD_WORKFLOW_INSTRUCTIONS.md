# BMAD Workflow Instructions for Flux UI Hardening

## Story: flux-ui-hardening.md

This document provides the complete BMAD workflow commands to execute the Flux UI hardening story using the BMAD Method agents.

## Prerequisites

1. **Feature Branch**: `feature/flux-ui-hardening-bmad` (already created)
2. **Story File**: `docs/stories/flux-ui-hardening.md` (already exists)
3. **BMAD Agents Required**:
   - Scrum Master (SM)
   - Developer (Dev)
   - QA/Test Architect (QA)

## BMAD Agent Workflow

### Phase 1: Scrum Master - Story Approval

Load the **Scrum Master agent** from `BMAD-METHOD/bmad-core/agents/sm.md` or `BMAD-METHOD/dist/agents/sm.txt`

**Commands to execute:**

```
*load docs/stories/flux-ui-hardening.md

*approve-story docs/stories/flux-ui-hardening.md
```

**Expected Output:**
- SM will review the story for completeness
- SM will validate acceptance criteria are clear
- SM will update story status to "Approved"
- SM will provide implementation guidance

---

### Phase 2: Developer - Story Implementation

Load the **Developer agent** from `BMAD-METHOD/bmad-core/agents/dev.md` or `BMAD-METHOD/dist/agents/dev.txt`

**Commands to execute:**

```
*develop-story docs/stories/flux-ui-hardening.md
```

**Expected Actions:**
- Dev will read the approved story
- Dev will implement the following tasks:
  1. Remove unused variables/components surfaced by ESLint
  2. Replace `any` types with explicit types or shared interfaces
  3. Audit and fix hook dependency arrays (`useEffect`, `useCallback`)
  4. Add/repair Jest unit coverage for linted components
  5. Document follow-up issues as new stories
- Dev will ensure `npm run lint` passes with no warnings
- Dev will update the story with implementation notes

---

### Phase 3: QA/Test Architect - Review and Gate

Load the **QA agent** from `BMAD-METHOD/bmad-core/agents/qa.md` or `BMAD-METHOD/dist/agents/qa.txt`

**Commands to execute:**

```
*review docs/stories/flux-ui-hardening.md

*gate docs/stories/flux-ui-hardening.md
```

**Expected Actions:**
- QA will verify all acceptance criteria:
  1. ✅ `npm run lint` passes with no warnings
  2. ✅ `npm test` (Jest) executes without failure
  3. ✅ Key Playwright smoke flows pass
  4. ✅ Debug/Document pages load without console errors
- QA will run automated tests
- QA will document any issues found
- QA will approve the story or request fixes

---

## Final Steps

Once QA approval is complete:

1. **Update Project Log**:
   ```bash
   # Add entry to docs/flux-project-log.md
   echo "## $(date +%Y-%m-%d) - Flux UI Hardening Complete" >> docs/flux-project-log.md
   echo "- Achieved lint-zero status" >> docs/flux-project-log.md
   echo "- Added smoke test coverage" >> docs/flux-project-log.md
   echo "- Fixed ESLint warnings (unused vars, any types, hook deps)" >> docs/flux-project-log.md
   ```

2. **Merge Feature Branch**:
   ```bash
   git add .
   git commit -m "feat: Flux UI hardening - lint cleanup and test coverage"
   git checkout main
   git merge feature/flux-ui-hardening-bmad
   ```

## BMAD Agent Loading Instructions

### Option 1: Web UI (ChatGPT/Claude/Gemini)

1. Load the full-stack team file: `BMAD-METHOD/dist/teams/team-fullstack.txt`
2. Or load individual agents from `BMAD-METHOD/dist/agents/`
3. Use commands: `*sm`, `*dev`, `*qa` to switch between agents

### Option 2: IDE Integration

Follow the BMAD user guide for IDE integration:
- [BMAD User Guide](../BMAD-METHOD/docs/user-guide.md)

## Notes

- The BMAD commands (`*approve-story`, `*develop-story`, `*review`, `*gate`) work within AI chat interfaces
- Each agent maintains context through the story file
- Story file acts as the source of truth for all agents
- Agents pass notes to each other through story updates

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
