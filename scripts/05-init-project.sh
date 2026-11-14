#!/usr/bin/env bash
# 05-init-project.sh /path/to/project
# Prepares a project for the "no re-teaching" workflow:
#   .agent/HANDOFF.md   long-term memory, auto-loaded on every request (keep it short and stable)
#   .agent/SCRATCH.md   working journal, NOT auto-loaded (append-only)
#   opencode.json       project config that loads .agent/HANDOFF.md via "instructions"
#   AGENTS.md           skeleton (only if the project has none; run /init in OpenCode afterwards)
# Existing files are never overwritten.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$ROOT/config/project-templates"
PROJECT="${1:-}"

if [[ -z "$PROJECT" || ! -d "$PROJECT" ]]; then
  echo "Usage: $0 /path/to/project" >&2
  exit 1
fi
PROJECT="$(cd "$PROJECT" && pwd)"

mkdir -p "$PROJECT/.agent"

if [[ -f "$PROJECT/.agent/HANDOFF.md" ]]; then
  echo "kept existing .agent/HANDOFF.md"
else
  cp "$T/HANDOFF.md" "$PROJECT/.agent/HANDOFF.md"
  echo "created .agent/HANDOFF.md"
fi

if [[ -f "$PROJECT/.agent/SCRATCH.md" ]]; then
  echo "kept existing .agent/SCRATCH.md"
else
  : > "$PROJECT/.agent/SCRATCH.md"
  echo "created .agent/SCRATCH.md"
fi

if [[ -f "$PROJECT/opencode.json" ]]; then
  echo
  echo "NOTE: $PROJECT/opencode.json already exists. Merge this key into it yourself:"
  echo '  "instructions": [".agent/HANDOFF.md"]'
else
  cp "$T/opencode.project.json" "$PROJECT/opencode.json"
  echo "created opencode.json"
fi

if [[ -f "$PROJECT/AGENTS.md" ]]; then
  echo "kept existing AGENTS.md"
else
  cp "$T/AGENTS.project.md" "$PROJECT/AGENTS.md"
  echo "created AGENTS.md skeleton (fill it in, or run /init in OpenCode to improve it)"
fi

# Keep the agent's memory files out of git without touching tracked files (local-only exclude).
if [[ -d "$PROJECT/.git" ]]; then
  EXCL="$PROJECT/.git/info/exclude"
  mkdir -p "$(dirname "$EXCL")"
  touch "$EXCL"
  if ! grep -qx '.agent/' "$EXCL"; then
    echo '.agent/' >> "$EXCL"
    echo "added .agent/ to .git/info/exclude"
  fi
fi

echo
echo "Project ready: $PROJECT"
echo "Start the model server (scripts/<model>/llm-up.sh), then run 'opencode' inside the project."
