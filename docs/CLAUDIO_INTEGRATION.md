# Claudio Integration Guide

Claudio adds gentle audio feedback to Claude Code sessions. Little bursts of sound
make long migrations feel less monotonous—consider it a tiny UX upgrade for the
multi-agent crew.

## Quick install

```bash
./scripts/install_claudio.sh
```

What the script does:

1. Verifies the Go toolchain is available.
2. Installs the `claudio` CLI from [claudio.click](https://claudio.click).
3. Runs `claudio install` so Claude Code hooks call the tool automatically.

If the script cannot update hooks (for example on a locked-down environment), follow
the manual instructions below.

## Manual setup

```bash
go install claudio.click/cmd/claudio@latest

# Ensure GOPATH/bin is on PATH, then wire the hooks:
claudio install
```

If you prefer manual editing, add this snippet to your Claude Code configuration:

```json
{
  "hooks": {
    "PreToolUse": "claudio",
    "PostToolUse": "claudio",
    "UserPromptSubmit": "claudio"
  }
}
```

## Customising sound packs

Claudio ships with sensible defaults (system sounds on macOS, Windows, Linux).
You can override them by creating a directory soundpack:

```
~/.config/claudio/sounds/
 ├── loading/
 ├── success/
 ├── error/
 ├── interactive/
 └── default.wav
```

Alternatively, use a JSON mapping:

```json
{
  "name": "dionysus-pack",
  "mappings": {
    "success/git-commit-success.wav":
      "/System/Library/Sounds/Hero.aiff",
    "error/bash-error.wav":
      "/System/Library/Sounds/Basso.aiff",
    "default.wav":
      "/System/Library/Sounds/Glass.aiff"
  }
}
```

Point Claudio at the file via `~/.config/claudio/config.json`.

## Tips

- Keep the volume modest (`volume: 0.5` is a good starting point).
- Status-specific sounds (e.g., `success/git-commit-success.wav`) fall back to
  more generic categories if a file is missing, so you can start small and grow the pack.
- When pair-programming with agents, the audio cues double as situational awareness—
  you know when a command starts, succeeds, or fails without watching logs.
