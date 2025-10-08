# Backlog.md Workspace for Dionysus 2.0

This directory follows the [Backlog.md](https://github.com/MrLesk/Backlog.md) format so we can manage tasks and milestones directly inside the repository.  
Use the official CLI (`npm i -g backlog.md` or `brew install backlog-md`) to interact with the board.

## Quick start

```bash
# install once
npm i -g backlog.md  # or bun add -g backlog.md / brew install backlog-md

# view the kanban board (opens TUI)
backlog board view

# export the board to markdown
backlog board export backlog.md --output docs/BACKLOG_EXPORT.md

# create a new task inside this repo
backlog task create "Implement citation scroll behaviour" -l spec-058 -s "To Do"
```

## Structure

- `backlog/config.yml` holds project-level settings (statuses, milestones, editor preference).
- `backlog/tasks/` contains markdown files for each task.
- `backlog/docs/`, `backlog/decisions/`, and related folders are ready for future Backlog.md resources.

## Current milestone

The `spec-058` milestone tracks the Flux Citation Trust UI work. Tasks are linked via `milestone: spec-058` metadata so the CLI and Kanban views can filter easily.

Remember to run `backlog task edit <id> -s "In Progress"` (or mark as done) as you move through the work, and keep acceptance criteria updated.
