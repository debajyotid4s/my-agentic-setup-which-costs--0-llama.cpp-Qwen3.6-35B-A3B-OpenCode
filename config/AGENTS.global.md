# Working rules
Local model, one inference slot: every token in context costs real time. Be lean.

## Workflow
- Read only the relevant files first. Then state in 2-3 lines what you will change and why.
- Think briefly: decide, act, verify. Do not re-derive a conclusion you already reached or re-read a file that is already in context.
- For tasks with 3+ steps, keep a short todo list (todowrite) and update it as you go.
- Small, focused edits. Never invent file names, flags, config keys or APIs; inspect the code (grep/read) instead.
- Do not commit, push, or run destructive commands (rm -rf, git reset --hard, force push) unless asked.

## Context budget
- Never cat whole large files or logs. Locate with grep -n / glob first, then read only the needed range (read with offset/limit, or sed -n 'a,bp'). Use head/tail -n 80 on any command that may be noisy.
- Tool output is NOT cleared automatically (pruning is off). Everything you print stays in context until compaction, so keep commands quiet and filtered.
- Issue independent reads/greps together in one turn instead of one per turn.
- Prefer targeted edits over rewriting files. Write files longer than ~150 lines in chunks (one response is capped at ~32K tokens).
- Avoid subagents (task tool) unless the exploration would otherwise dump thousands of lines into context. There is one inference slot: subagents run serially and can evict the main session's cache.
- Keep key facts in .agent/SCRATCH.md so they survive compaction.

## Edit and test loop
- Re-read the exact lines right before editing them, so the match is exact.
- After each edit, run the narrowest relevant test (single file or test name) using the test command from the project AGENTS.md. Pipe output through tail -n 60 or grep for failures.
- Run the full suite once before declaring done. Report real results; never claim a fix without a passing run.
- If the same approach fails twice, stop. Record what you learned in .agent/SCRATCH.md, then change approach or ask the user. No blind retry loops.

## Memory (two files, two jobs)
- At the start of a session, if `.agent/SCRATCH.md` exists, run `tail -n 40 .agent/SCRATCH.md` before doing anything else. (`.agent/HANDOFF.md` is already loaded for you.)
- `.agent/SCRATCH.md`: working journal, NOT auto-loaded. After each milestone (root cause found, hypothesis ruled out, fix applied, test result) append one line: `echo '...' >> .agent/SCRATCH.md`. Keep exact error messages, path:line and commands verbatim. Read it only with `tail -n 40`.
- `.agent/HANDOFF.md`: long-term memory, auto-loaded on every request, so keep it small and stable. Max 60 lines. Rewrite it (do not append) only when the task is finished or blocked, the user says to wrap up, or a major phase ends. Never after every milestone.
  Contents: goal, status, decisions and why, open problems (verbatim errors, path:line), exact next commands, key files. Distill from SCRATCH.md; do not copy it.
- Reason: editing HANDOFF.md changes the start of the prompt, which forces the server to reprocess the whole context (minutes, not seconds).
- If `.agent/` is missing, create it (`mkdir -p .agent`).
