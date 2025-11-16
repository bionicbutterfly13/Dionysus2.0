# Implementation Tasks

## Phase 1: Import Command (Core Integration)
- [ ] Create `/openspec:import-to-archon` slash command in `.claude/commands/openspec/`
- [ ] Implement task parser to extract tasks from `tasks.md` markdown
- [ ] Add Archon project creation logic (title from proposal.md)
- [ ] Add Archon task creation loop (one per tasks.md item)
- [ ] Store Archon project_id in `.archon-project-id` file
- [ ] Add error handling for Archon API failures
- [ ] Test import with sample OpenSpec change (3-5 tasks)

## Phase 2: Status Sync (Bidirectional Updates)
- [ ] Design status sync mechanism (polling vs webhook)
- [ ] Implement Archon task status poller (check for "done" tasks)
- [ ] Implement tasks.md checkbox updater (parse and modify markdown)
- [ ] Add git commit for automatic tasks.md updates
- [ ] Handle sync conflicts (manual edits vs automatic updates)
- [ ] Test sync with completing tasks in Archon
- [ ] Add configuration for sync interval (default: 30 seconds)

## Phase 3: Archive Integration
- [ ] Enhance `/openspec:archive` to read `.archon-project-id`
- [ ] Add Archon project validation (all tasks done?)
- [ ] Implement Archon project archival on OpenSpec archive
- [ ] Add error messages for incomplete Archon tasks
- [ ] Test archive workflow end-to-end
- [ ] Document archive validation in error messages

## Phase 4: Documentation & Testing
- [ ] Update `openspec/AGENTS.md` with new workflow
- [ ] Add examples to CLAUDE.md for OpenSpec + Archon integration
- [ ] Create integration test for import → work → archive flow
- [ ] Document `.archon-project-id` file format
- [ ] Add troubleshooting guide for sync issues
- [ ] Create demo video/walkthrough of integrated workflow

## Dependencies
- Archon MCP server must be available and responsive
- OpenSpec CLI v0.14.0+ installed
- Git repository must be initialized (for commit hooks)

## Validation
- Run `/openspec:import-to-archon integrate-openspec-archon-sync` on this change
- Complete 3 tasks in Archon, verify tasks.md updates
- Run `/openspec:archive integrate-openspec-archon-sync`, verify validation
