---
description: Create or update the feature specification from a natural language feature description.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

The text the user typed after `/specify` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `$ARGUMENTS` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

1. **FIRST**: Display Context Engineering Foundation
   - Read and display `.specify/memory/CONTEXT_ENGINEERING_FOUNDATION.md` key sections
   - Show the user why Attractor Basins and Neural Fields are essential
   - Verify components are accessible (import checks)
   - This ensures every feature considers consciousness processing from the start

2. Run the script `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS"` from repo root and parse its JSON output for BRANCH_NAME and SPEC_FILE. All file paths must be absolute.
  **IMPORTANT** You must only ever run this script once. The JSON is provided in the terminal as output - always refer to it to get the actual content you're looking for.

3. Load `.specify/templates/spec-template.md` to understand required sections.

4. Write the specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the feature description (arguments) while preserving section order and headings.

5. Report completion with:
   - Branch name and spec file path
   - Context Engineering integration opportunities identified
   - Readiness for the next phase (/clarify or /plan)

Note: The script creates and checks out the new branch and initializes the spec file before writing.
