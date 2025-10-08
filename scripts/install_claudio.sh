#!/usr/bin/env bash
#
# Claudio audio feedback installer
# ---------------------------------
# Brings the claudio.click tool into the Dionysus workspace so Claude Code
# sessions play a little sonic feedback while agents work.
# This script is idempotent – safe to rerun whenever you update or switch machines.

set -euo pipefail

# Colours for minimal UX feedback
BLUE="\033[34m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

log() {
  printf "%b[Claudio]%b %s\n" "${BLUE}" "${RESET}" "$1"
}

warn() {
  printf "%b[Claudio]%b %s\n" "${YELLOW}" "${RESET}" "$1"
}

success() {
  printf "%b[Claudio]%b %s\n" "${GREEN}" "${RESET}" "$1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "Missing dependency: $1"
    return 1
  fi
}

main() {
  log "Checking prerequisites…"
  if ! require_command go; then
    warn "Go toolchain is required. Install from https://go.dev/dl/ and rerun."
    exit 1
  fi

  # Ensure Go bin path exists in PATH for current shell session.
  GO_BIN="$(go env GOPATH)/bin"
  export PATH="${GO_BIN}:${PATH}"

  log "Installing Claudio CLI from claudio.click…"
  go install claudio.click/cmd/claudio@latest

  if ! command -v claudio >/dev/null 2>&1; then
    warn "Installation succeeded but 'claudio' was not found on PATH."
    warn "Verify GOPATH/bin is exported and rerun this script."
    exit 1
  fi

  log "Wiring Claudio into Claude Code hooks…"
  if claudio install; then
    success "Claudio is now installed and hook configuration updated."
  else
    warn "Automatic hook install failed. Configure manually using README instructions."
    exit 1
  fi

  cat <<'EOF'

Claudio is ready! Enjoy a little audio joy during Claude Code sessions.
To customise sounds, see docs/CLAUDIO_INTEGRATION.md.

EOF
}

main "$@"
