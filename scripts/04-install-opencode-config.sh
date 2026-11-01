#!/usr/bin/env bash
# 04-install-opencode-config.sh
# Installs the global OpenCode files into ~/.config/opencode (or $XDG_CONFIG_HOME/opencode):
#   opencode.json                   provider (llama.cpp), compaction, agent sampling
#   AGENTS.md                       global working rules for the local model
#   commands/handoff.md             the /handoff command
#   plugins/better-compaction.ts    custom compaction prompt (experimental hook)
# Existing files that differ are backed up as <file>.bak.<timestamp>.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
STAMP="$(date +%Y%m%d-%H%M%S)"

die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

install_file() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || die "Missing source file: $src"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]] && ! cmp -s "$src" "$dst"; then
    cp -a "$dst" "$dst.bak.$STAMP"
    echo "  backed up existing file -> $dst.bak.$STAMP"
  fi
  install -m 644 "$src" "$dst"
  echo "  installed $dst"
}

command -v python3 >/dev/null 2>&1 || die "python3 is required to validate the JSON."
python3 -m json.tool "$ROOT/config/opencode.json" >/dev/null || die "config/opencode.json is not valid JSON."

echo "Installing OpenCode files into $CFG"
install_file "$ROOT/config/opencode.json"                  "$CFG/opencode.json"
install_file "$ROOT/config/AGENTS.global.md"               "$CFG/AGENTS.md"
install_file "$ROOT/config/commands/handoff.md"            "$CFG/commands/handoff.md"
install_file "$ROOT/config/plugins/better-compaction.ts"   "$CFG/plugins/better-compaction.ts"

echo
echo "Done. Next: start the server (scripts/llm-up.sh), then initialise each project with"
echo "scripts/05-init-project.sh /path/to/project"
