## Claude Session Log

Use this file to capture key decisions, next steps, and prompts whenever you pause
or restart a Claude Code session. Keeping a short running log makes it easy to
resume context after a restart.

### 2025-10-07
- Backend citation endpoint implemented (Spec 058). Contract tests now green.
- Next focus: build `CitationPanel` component and wire document detail click events.
- Reminder: run `pytest backend/tests/contract/test_documents_citations_get.py` and
  `npm test -- CitationPanel` before committing related changes.
