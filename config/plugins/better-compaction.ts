// better-compaction.ts
// Replaces OpenCode's default compaction prompt so the summary keeps exact details
// (errors, paths, flags) and debugging state. Uses an experimental hook: after your first
// /compact, read the summary to confirm the custom prompt took effect.
export const BetterCompactionPlugin = async () => {
  return {
    "experimental.session.compacting": async (_input: any, output: any) => {
      output.prompt = [
        "Compact this conversation so the work can resume seamlessly.",
        "Never generalize: keep exact names, paths, values, error messages, flags and versions.",
        "Reproduce key user-pasted content verbatim.",
        "Keep every user constraint and preference.",
        "Keep debugging state: what was tried, what was ruled out, and the evidence.",
        "Use this template: ## Goal / ## Instructions / ## Discoveries / ## Completed / ## Not solved / ## Next steps / ## Files.",
        "End with: Before continuing, run: tail -n 40 .agent/SCRATCH.md (HANDOFF.md is already loaded).",
      ].join("\n");
    },
  };
};
