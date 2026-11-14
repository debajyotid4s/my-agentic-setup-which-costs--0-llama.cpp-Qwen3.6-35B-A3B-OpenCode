---
description: Distill SCRATCH.md into .agent/HANDOFF.md (deliberate session reset only)
---
Rewrite .agent/HANDOFF.md (max 60 lines) from .agent/SCRATCH.md and this session:
goal, status, decisions and why, open problems (verbatim errors, path:line),
exact next commands, key files. Do not copy SCRATCH.md. Then reply "saved".

Use this only for a deliberate reset: a new goal, damaged history, or a model switch.
Normal context pressure is handled by compaction, which keeps the current session;
do not run `/new` after a routine compaction.
