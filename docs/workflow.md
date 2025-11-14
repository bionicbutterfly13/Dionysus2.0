# Dionysus 2.0 Development Workflow (OpenSpec + Archon)

1. **Change Proposals First**
   - Use `/openspec:proposal` to create formal change specifications
   - Generate `proposal.md`, `tasks.md`, and `design.md` in `openspec/changes/<id>/`
   - Validate with `openspec validate <id> --strict` before proceeding

2. **Bridge to Task Management**
   - Create Archon project: `manage_project("create", title="...")`
   - Import tasks from OpenSpec `tasks.md` into Archon task tracking
   - Assign priority with `task_order` (higher = more priority)

3. **Implementation Workflow**
   - Get next task: `find_tasks(filter_by="status", filter_value="todo")`
   - Start work: `manage_task("update", task_id="...", status="doing")`
   - Research first: Use `rag_search_knowledge_base()` before implementing
   - Follow TDD: Write test → implement → refactor
   - Mark for review: `manage_task("update", task_id="...", status="review")`

4. **Developer Checklist**
   - Run linter: `ruff check src`
   - Run tests: `pytest` (backend) or `npm test` (frontend)
   - Verify contract tests pass: `pytest -m contract`
   - Update task status to "done" when complete

5. **Pull Request & Archive**
   - Create feature branch from `main` (e.g., `feature/add-vector-search`)
   - Open PR with OpenSpec proposal link
   - After merge, use `/openspec:archive` to move to archive and update main specs
   - Archive Archon project when all tasks complete

6. **Project Memory**
   - Archon maintains task history and project progress
   - OpenSpec archives preserve all change rationale and design decisions
   - Use Archon RAG to search past implementations and patterns

