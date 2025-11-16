---
name: OpenSpec: Archive
description: Archive a deployed OpenSpec change and update specs.
category: OpenSpec
tags: [openspec, archive]
---
<!-- OPENSPEC:START -->
**Guardrails**
- Favor straightforward, minimal implementations first and add complexity only when it is requested or clearly required.
- Keep changes tightly scoped to the requested outcome.
- Refer to `openspec/AGENTS.md` (located inside the `openspec/` directory—run `ls openspec` or `openspec update` if you don't see it) if you need additional OpenSpec conventions or clarifications.

**Steps**
1. Determine the change ID to archive:
   - If this prompt already includes a specific change ID (for example inside a `<ChangeId>` block populated by slash-command arguments), use that value after trimming whitespace.
   - If the conversation references a change loosely (for example by title or summary), run `openspec list` to surface likely IDs, share the relevant candidates, and confirm which one the user intends.
   - Otherwise, review the conversation, run `openspec list`, and ask the user which change to archive; wait for a confirmed change ID before proceeding.
   - If you still cannot identify a single change ID, stop and tell the user you cannot archive anything yet.
2. Validate the change ID by running `openspec list` (or `openspec show <id>`) and stop if the change is missing, already archived, or otherwise not ready to archive.
3. **Archon Integration Check** (if Archon MCP is available):
   - Check if `openspec/changes/<id>/.archon-project-id` exists
   - If exists:
     a. Read the Archon project UUID from the file
     b. Query Archon for project completion: `mcp__archon__find_tasks(project_id="<uuid>", filter_by="status", filter_value="todo")`
     c. Also check for "doing" and "review" status tasks
     d. If any tasks are not "done":
        - Report: "⚠️ Archon project has N incomplete tasks (M todo, K doing, L review)"
        - List top 5 incomplete tasks with titles
        - Prompt user: "Archive anyway? (This will leave incomplete tasks in Archon)"
        - Wait for confirmation before proceeding
     e. If all tasks are "done":
        - Report: "✅ All Archon tasks complete ({total} tasks)"
        - Ask: "Archive Archon project too? (Recommended)"
        - If confirmed, call: `mcp__archon__manage_project("update", project_id="<uuid>", archived=true)` (Note: Check if archive field exists, if not, just report completion)
   - If `.archon-project-id` doesn't exist:
     - Report: "ℹ️ No Archon project linked (proceeding without validation)"
4. Run `openspec archive <id> --yes` so the CLI moves the change and applies spec updates without prompts (use `--skip-specs` only for tooling-only work).
5. Review the command output to confirm the target specs were updated and the change landed in `changes/archive/`.
6. Validate with `openspec validate --strict` and inspect with `openspec show <id>` if anything looks off.

**Archon Validation Error Messages**

When archiving a change with linked Archon tasks, you may see:

**⚠️ Archon project has N incomplete tasks**
```
⚠️ Archon project has 15 incomplete tasks (10 todo, 3 doing, 2 review)

Incomplete tasks:
  1. [todo] Implement feature X
  2. [todo] Add unit tests
  3. [doing] Write documentation
  4. [review] Code review cleanup
  5. [review] Update CHANGELOG

Archive anyway? (This will leave incomplete tasks in Archon)
```

**What this means**: The linked Archon project still has tasks that aren't marked "done".

**How to resolve**:
- **Option 1: Complete tasks** - Finish remaining work, mark tasks as "done" in Archon, then re-run archive
- **Option 2: Sync status** - Run `/openspec:sync-status <change-id>` to update tasks.md with latest Archon status
- **Option 3: Force archive** - Proceed anyway (not recommended - leaves tasks orphaned)

**✅ All Archon tasks complete**
```
✅ All Archon tasks complete (26 tasks)
Archive Archon project too? (Recommended)
```

**What this means**: All tasks in Archon are marked "done". Archive is ready to proceed.

**Recommendation**: Confirm "yes" to archive the Archon project along with the OpenSpec change for clean lifecycle management.

**ℹ️ No Archon project linked**
```
ℹ️ No Archon project linked (proceeding without validation)
```

**What this means**: The change has no `.archon-project-id` file (was never imported to Archon, or was created before integration).

**No action needed**: Archive proceeds normally using only OpenSpec validation.

**Reference**
- Use `openspec list` to confirm change IDs before archiving.
- Inspect refreshed specs with `openspec list --specs` and address any validation issues before handing off.
- For Archon task status: `/openspec:sync-status <change-id>`
- For Archon project details: `mcp__archon__find_tasks(project_id="<uuid>")`
<!-- OPENSPEC:END -->
