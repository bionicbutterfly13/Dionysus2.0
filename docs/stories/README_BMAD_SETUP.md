# BMAD v6-alpha Setup for Flux UI Hardening

## Overview

This setup prepares the Flux UI hardening story for execution using BMAD v6-alpha (BMad Method Module). BMAD v6 uses CLI-based workflows with specialized agents that maintain state through local workspace files.

## What Has Been Set Up

1. **Feature Branch**: `feature/flux-ui-hardening-bmad`
   - Created and checked out
   - Ready for BMAD v6-alpha workflow execution

2. **Story File**: `docs/stories/flux-ui-hardening.md`
   - Contains complete story with acceptance criteria
   - Will be processed by BMAD agents

3. **BMAD v6-alpha**: Version 6.0.0-alpha.0
   - Installed at project root
   - BMM (BMad Method Module) available

4. **Workflow Instructions**: `docs/stories/BMAD_WORKFLOW_INSTRUCTIONS.md`
   - Complete v6-alpha workflow steps
   - Phase 4 (Implementation) commands
   - Agent roles and responsibilities

## BMAD v6-alpha Key Changes from v4

### Major Differences

| Feature | v4 | v6-alpha |
|---------|-----|----------|
| **Interface** | Web chat commands (`*approve-story`) | CLI commands (`bmad sm create-story`) |
| **Agent Loading** | Load .txt files in ChatGPT/Claude | Run CLI commands in IDE |
| **State Management** | Story files only | Workspace files in `bmad/` directory |
| **QA Role** | Separate QA agent | SR (Senior Reviewer) agent |
| **Context** | Manual context in story | Auto-generated story-context |
| **Workflows** | Simple task-based | 4-Phase Scale-Adaptive system |

### v6-alpha Workflow Phases

1. **Analysis Phase** (Optional) - Brainstorming, research, product brief
2. **Planning Phase** (Required) - Scale-adaptive project planning (PRD/GDD)
3. **Solutioning Phase** (Level 3-4) - Architecture and tech specs
4. **Implementation Phase** (Iterative) - Story creation → dev → review → retrospective

## How to Execute the BMAD v6-alpha Workflow

### Step-by-Step Execution

For this UI hardening task (Level 1 complexity), we use **Phase 4 (Implementation)** directly:

#### 1. SM - Create Story
```bash
bmad sm create-story
```
- Point SM to `docs/stories/flux-ui-hardening.md`
- SM validates story structure and acceptance criteria

#### 2. SM - Generate Story Context (NEW in v6)
```bash
bmad sm story-context
```
- Generates just-in-time expertise injection
- Prepares optimized context for developer

#### 3. DEV - Implement Story
```bash
bmad dev dev-story
```
- Implements all tasks from story
- Ensures `npm run lint` passes
- Updates story with implementation notes

#### 4. SR - Review Implementation
```bash
bmad sr review-story
```
- Validates code quality and acceptance criteria
- Documents any issues
- Approves or requests corrections

#### 5. DEV - Course Correction (if needed)
```bash
bmad dev correct-course
```
- Addresses SR feedback
- Re-runs validation

### Installation (If Not Already Installed)

BMAD v6-alpha is already installed in this project. If you need to install in another project:

```bash
# Clone BMAD v6-alpha
git clone --branch v6-alpha --depth 1 https://github.com/bmad-code-org/BMAD-METHOD.git

# Install node modules
cd BMAD-METHOD
npm install

# Run installer
npm run install:bmad
```

Follow installer prompts to configure for your project.

## v6-alpha Resources

- **[BMM Module README](../../BMAD-METHOD/src/modules/bmm/README.md)** - Overview and quick start
- **[v6 Workflows Guide](../../BMAD-METHOD/src/modules/bmm/workflows/README.md)** - **MUST READ** - Complete workflow documentation
- **[Main README](../../BMAD-METHOD/README.md)** - v6-alpha introduction and philosophy
- **[Discord Community](https://discord.gg/gk8jAdXWmj)** - Get help and share feedback

## Why BMAD v6-alpha?

### Human Amplification Philosophy

BMAD v6 is built on **C.O.R.E.** (Collaboration Optimized Reflection Engine):
- **Collaboration**: Human-AI partnership leveraging unique strengths
- **Optimized**: Refined processes for maximum effectiveness
- **Reflection**: Guided thinking to discover better solutions
- **Engine**: Framework orchestrating specialized agents

### Scale-Adaptive Workflows

v6-alpha automatically adapts to project complexity:
- **Level 0**: Single atomic change
- **Level 1**: 1-10 stories (this task)
- **Level 2**: 5-15 stories, focused PRD
- **Level 3**: 12-40 stories, full architecture
- **Level 4**: 40+ stories, enterprise scale

### Just-In-Time Context

Story-context generates specialized expertise for each task, eliminating generic developer guidance and providing exactly what's needed.

## Status

- ✅ Feature branch created
- ✅ Story file exists with acceptance criteria
- ✅ BMAD v6-alpha installed (v6.0.0-alpha.0)
- ✅ Workflow instructions updated for v6
- ⏳ Ready for Phase 4 workflow execution
- ⏳ Awaiting SM → DEV → SR workflow completion

## Alternative: Manual Execution

If you prefer to execute without BMAD:

1. Follow task list in `docs/stories/flux-ui-hardening.md`
2. Run `npm run lint` in frontend directory
3. Fix each category of issues (unused vars, any types, hook deps)
4. Run tests to verify
5. Update project log manually

---

**Next Action**: Run `bmad sm create-story` to begin Phase 4 workflow.
