# Flux Development Workflow (BMAD-Aligned)

1. **Branch First**  
   - Always create a feature branch from `main` before starting work (e.g., `feature/flux-ui-hardening`).
   - No direct commits to `main`.

2. **Story-Driven**  
   - Capture every piece of work as a story under `docs/stories/`.  
   - Update the status blocks (`Draft → Approved → In Progress → Ready for Review → Done`).

3. **Developer Checklist**  
   - Run lint/tests locally (`npm run lint`, `npm test`, backend `pytest` as relevant).  
   - Record results in the story file before opening a PR.

4. **PR Template**  
   - Use the BMAD checklist (Summary, Risk, Tests, QA Gate).  
   - Links to relevant story and QA artifacts.

5. **QA Agent (Future)**  
   - Once Flux stabilizes, re-enable QA tasks (`*risk`, `*design`, `*review`) to populate `docs/qa/`.

6. **Project Log**  
   - Append major decisions and sessions to `docs/flux-project-log.md` for long-term memory.

