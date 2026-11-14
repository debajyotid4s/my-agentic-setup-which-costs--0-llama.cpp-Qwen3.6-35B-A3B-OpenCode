# deferred-compaction

OpenCode V2 plugin that owns context compaction for the local `llama.cpp` provider.

`opencode.json` sets `compaction.auto: false`. This plugin decides when to compact:

1. `PENDING` once a step's context (`input + cache.read + output` tokens) reaches 80% of the model window (209,715 of 262,144).
2. Compact only when **all** hold: execution ended, no tool in flight, last step finished with `stop`, session idle.
3. Hybrid prewarm: below the threshold, compact after 10 minutes idle once usage is at 70%.
4. One compaction at a time across sessions. Child/subagent sessions are ignored.
5. Failed requests retry forever with a capped backoff (1 min × failures, capped at 3×).
6. Watchdogs: a request with no compaction event after 5 minutes is retried; a compaction
   that never reports ended/failed stops blocking after 30 minutes.

The controller manages the local `v100` provider only. `compaction.auto` is global, so other
providers are left alone: their cached-token accounting differs (`input` already includes
`cache.read`), and the 262K fallback limit belongs to llama.cpp. Set
`DEFERRED_COMPACTION_PROVIDER=*` only when every provider reports usage like llama.cpp.

The server does not always emit `session.idle` for API-created sessions, so a finished
`session.execution.succeeded|failed|interrupted` also marks the session safe. Context is
estimated as `input + cache.read + output`, which matches llama.cpp's usage split.

The session is preserved. `keep.tokens: 15000` keeps recent turns; no `/new` is involved.

## Files

- `index.ts` — V2 entrypoint (`Plugin.define`).
- `plugin.ts` — controller: event subscription, model limit resolution, compaction requests.
- `state.ts` — pure state machine, no OpenCode imports.
- `state.test.ts` — unit tests.

## Test

```sh
cd config/plugins/deferred-compaction
npx -y tsx state.test.ts
npx -y tsx plugin.test.ts
```

## Tuning

Environment variables read by the plugin:

| Variable | Default | Meaning |
|---|---|---|
| `DEFERRED_COMPACTION_PROVIDER` | `v100` | provider to manage; `*` = all (llama.cpp-style usage only) |
| `DEFERRED_COMPACTION_THRESHOLD` | `0.8` | compact at this share of the window |
| `DEFERRED_COMPACTION_PREWARM` | `0.7` | proactive compaction share |
| `DEFERRED_COMPACTION_IDLE_MS` | `600000` | idle time before prewarm |
| `DEFERRED_COMPACTION_POLL_MS` | `15000` | timer resolution |

## Why not background compaction

`experimental.background_compaction` only applies to automatic compaction, and the server runs `--parallel 1`.
A background summary is another inference request on the same slot, so it queues behind the agent. Deferred
compaction at an idle safe point is the only variant that does not steal decode capacity.
