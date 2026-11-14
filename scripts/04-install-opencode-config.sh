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
install_file "$ROOT/config/plugins/deferred-compaction/index.ts" "$CFG/plugins/deferred-compaction/index.ts"
install_file "$ROOT/config/plugins/deferred-compaction/plugin.ts" "$CFG/plugins/deferred-compaction/plugin.ts"
install_file "$ROOT/config/plugins/deferred-compaction/state.ts" "$CFG/plugins/deferred-compaction/state.ts"

# The V1 plugin used the old @opencode-ai/plugin API and cannot load in V2.
# Keep it as a backup instead of leaving a broken file in the plugin directory.
if [[ -f "$CFG/plugins/better-compaction.ts" ]]; then
  mv "$CFG/plugins/better-compaction.ts" "$CFG/plugins/better-compaction.ts.v1.bak.$STAMP"
  echo "  retired V1 plugin -> $CFG/plugins/better-compaction.ts.v1.bak.$STAMP"
fi

# V2 plugins import from @opencode/plugin, not @opencode-ai/plugin.
if [[ ! -d "$CFG/node_modules/@opencode/plugin" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "Installing @opencode/plugin into $CFG ..."
    ( cd "$CFG" && npm install --no-fund --no-audit --save @opencode/plugin@2.3.0 )
  else
    echo "WARN: @opencode/plugin is missing and npm is unavailable." >&2
    echo "      Install it manually: cd $CFG && npm install @opencode/plugin@2.3.0" >&2
  fi
fi

echo
echo "Done. Next: start the server (scripts/<model>/llm-up.sh), then initialise each project with"
echo "scripts/05-init-project.sh /path/to/project"
