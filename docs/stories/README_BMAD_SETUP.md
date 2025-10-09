# BMAD Workflow Setup for Flux UI Hardening

## Overview

This setup prepares the Flux UI hardening story for execution using the BMAD Method (Breakthrough Method of Agile AI-Driven Development). BMAD uses specialized AI agents (Scrum Master, Developer, QA) to collaboratively complete development stories.

## What Has Been Set Up

1. **Feature Branch**: `feature/flux-ui-hardening-bmad`
   - Created and checked out
   - Ready for BMAD agent work

2. **Story File**: `docs/stories/flux-ui-hardening.md`
   - Contains the complete story with acceptance criteria
   - Serves as the source of truth for all agents

3. **Workflow Instructions**: `docs/stories/BMAD_WORKFLOW_INSTRUCTIONS.md`
   - Complete command sequence for BMAD agents
   - Phase-by-phase workflow (SM → Dev → QA)
   - Integration instructions for project log

## How to Execute the BMAD Workflow

### Important: BMAD Commands Work in AI Chat Interfaces

The BMAD commands (`*approve-story`, `*develop-story`, `*review`, `*gate`) are designed to be executed within an AI chat interface (ChatGPT, Claude, Gemini) where you've loaded the BMAD agent personas.

### Step-by-Step Execution

#### 1. Load the BMAD Scrum Master Agent

**Option A: Using Web UI**
- Open ChatGPT/Claude/Gemini
- Load the file: `BMAD-METHOD/dist/agents/sm.txt`
- Or load the full team: `BMAD-METHOD/dist/teams/team-fullstack.txt` and type `*sm`

**Option B: Using Source**
- Use the markdown file: `BMAD-METHOD/bmad-core/agents/sm.md`

#### 2. Execute SM Phase
```
*load docs/stories/flux-ui-hardening.md
*approve-story docs/stories/flux-ui-hardening.md
```

#### 3. Switch to Developer Agent

Load `BMAD-METHOD/dist/agents/dev.txt` or type `*dev` if using full team

#### 4. Execute Dev Phase
```
*develop-story docs/stories/flux-ui-hardening.md
```

The Developer agent will:
- Remove unused variables/components
- Replace `any` types with proper TypeScript types
- Fix hook dependency arrays
- Add Jest unit tests
- Ensure `npm run lint` passes

#### 5. Switch to QA Agent

Load `BMAD-METHOD/dist/agents/qa.txt` or type `*qa` if using full team

#### 6. Execute QA Phase
```
*review docs/stories/flux-ui-hardening.md
*gate docs/stories/flux-ui-hardening.md
```

The QA agent will verify:
- ✅ `npm run lint` passes with no warnings
- ✅ `npm test` executes successfully
- ✅ Key Playwright flows work
- ✅ No console errors in dev mode

### 7. Final Steps (After QA Approval)

Update the project log:
```bash
echo "## $(date +%Y-%m-%d) - Flux UI Hardening Complete (BMAD)" >> docs/flux-project-log.md
echo "- Achieved lint-zero status via BMAD Dev agent" >> docs/flux-project-log.md
echo "- Added smoke test coverage" >> docs/flux-project-log.md
echo "- Fixed ESLint warnings (unused vars, any types, hook deps)" >> docs/flux-project-log.md
echo "- QA gate passed all acceptance criteria" >> docs/flux-project-log.md
```

Merge the feature branch:
```bash
git add .
git commit -m "feat: Flux UI hardening - BMAD workflow execution"
git checkout main
git merge feature/flux-ui-hardening-bmad
```

## Alternative: Direct Execution (Without BMAD Agents)

If you prefer to execute manually:

1. Follow task list in `docs/stories/flux-ui-hardening.md`
2. Run `npm run lint` in the frontend directory
3. Fix each category of issues
4. Run tests to verify
5. Update project log manually

## Why BMAD?

BMAD provides:
- **Context Continuity**: Story file maintains full context across agents
- **Specialized Expertise**: Each agent focuses on their domain (planning, dev, QA)
- **Quality Gates**: Built-in validation checkpoints
- **Documentation**: Agents document their work in the story file
- **Collaboration**: Agents pass notes through story updates

## Resources

- **BMAD User Guide**: `BMAD-METHOD/docs/user-guide.md`
- **BMAD Architecture**: `BMAD-METHOD/docs/core-architecture.md`
- **Available Agents**: `BMAD-METHOD/bmad-core/agents/`
- **Story Template**: `docs/stories/flux-ui-hardening.md`

## Status

- ✅ Feature branch created
- ✅ Story file exists with acceptance criteria
- ✅ Workflow instructions documented
- ⏳ Ready for BMAD agent execution
- ⏳ Awaiting SM → Dev → QA workflow completion

---

**Next Action**: Load the Scrum Master agent and execute the approval phase.
