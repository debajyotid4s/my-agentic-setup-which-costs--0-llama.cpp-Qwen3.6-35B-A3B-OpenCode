import test from "node:test"
import assert from "node:assert/strict"
import { CompactionScheduler, estimatePromptTokens, type SchedulerOptions } from "./state.ts"

const T0 = 1_000_000

function setup(opts: Partial<SchedulerOptions> = {}) {
  return new CompactionScheduler({
    thresholdRatio: 0.8,
    prewarmRatio: 0.7,
    prewarmIdleMs: 60_000,
    retryMs: 10_000,
    contextLimit: 1000,
    providerID: "v100",
    ...opts,
  })
}

function created(s: CompactionScheduler, id = "ses_1", providerID = "v100") {
  s.handle(
    {
      type: "session.created",
      data: { sessionID: id, model: { providerID, id: "m" } },
    },
    T0,
  )
  s.setLimit(id, 1000)
  return id
}

function step(s: CompactionScheduler, id: string, tokens: number, finish = "stop", now = T0) {
  s.handle(
    {
      type: "session.step.ended",
      data: { sessionID: id, tokens: { input: tokens, output: 0 }, finish },
    },
    now,
  )
}

function runningStep(s: CompactionScheduler, id: string, tokens: number, now = T0) {
  s.handle({ type: "session.execution.started", data: { sessionID: id } }, now)
  step(s, id, tokens, "stop", now)
  s.handle({ type: "session.execution.succeeded", data: { sessionID: id } }, now)
}

function idle(s: CompactionScheduler, id: string, now = T0) {
  s.handle({ type: "session.idle", data: { sessionID: id } }, now)
}

test("estimates context from uncached, cached, and output tokens", () => {
  assert.equal(estimatePromptTokens({ input: 100, output: 20, cache: { read: 90 } }), 210)
  assert.equal(estimatePromptTokens(undefined), 0)
})

test("does not compact below threshold", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 500)
  idle(s, id)
  assert.equal(s.tick(T0), null)
})

test("compacts at threshold once idle and checkpointed", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  assert.equal(s.tick(T0), id)
})

test("compacts when execution ends without a separate idle event", () => {
  const s = setup()
  const id = created(s)
  s.handle({ type: "session.execution.started", data: { sessionID: id } }, T0)
  step(s, id, 900)
  s.handle({ type: "session.execution.succeeded", data: { sessionID: id } }, T0)
  assert.equal(s.tick(T0), id)
})

test("waits while a tool is in flight", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  s.handle({ type: "session.tool.called", data: { sessionID: id, id: "call_1" } }, T0)
  idle(s, id)
  assert.equal(s.tick(T0), null)
  s.handle({ type: "session.tool.success", data: { sessionID: id, id: "call_1" } }, T0)
  assert.equal(s.tick(T0), id)
})

test("waits while the agent is still executing", () => {
  const s = setup()
  const id = created(s)
  s.handle({ type: "session.execution.started", data: { sessionID: id } }, T0)
  step(s, id, 900)
  assert.equal(s.tick(T0), null)
})

test("waits for a stop finish, not a tool-calls step", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  step(s, id, 900, "tool-calls")
  assert.equal(s.tick(T0), null)
})

test("compacts after the turn is interrupted on a tool-calls step", () => {
  const s = setup()
  const id = created(s)
  s.handle({ type: "session.execution.started", data: { sessionID: id } }, T0)
  s.handle({ type: "session.tool.called", data: { sessionID: id, id: "call_1" } }, T0)
  step(s, id, 900, "tool-calls")
  s.handle({ type: "session.execution.interrupted", data: { sessionID: id } }, T0)
  assert.equal(s.tick(T0), id)
})

test("compacts after the turn fails on a tool-calls step", () => {
  const s = setup()
  const id = created(s)
  s.handle({ type: "session.execution.started", data: { sessionID: id } }, T0)
  step(s, id, 900, "tool-calls")
  s.handle({ type: "session.execution.failed", data: { sessionID: id } }, T0)
  assert.equal(s.tick(T0), id)
})

test("prewarms only after the idle window", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 750)
  idle(s, id)
  assert.equal(s.tick(T0 + 30_000), null)
  assert.equal(s.tick(T0 + 61_000), id)
})

test("does not prewarm below the prewarm ratio", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 600)
  idle(s, id)
  assert.equal(s.tick(T0 + 120_000), null)
})

test("one compaction at a time across sessions", () => {
  const s = setup()
  const a = created(s, "ses_a")
  const b = created(s, "ses_b")
  runningStep(s, a, 900)
  runningStep(s, b, 900)
  idle(s, a)
  idle(s, b)
  const first = s.tick(T0)
  assert.ok(first === a || first === b)
  s.markRequested(first, T0)
  s.handle({ type: "session.compaction.started", data: { sessionID: first } }, T0)
  assert.equal(s.tick(T0), null)
})

test("a compaction request is not repeated before it starts", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  assert.equal(s.tick(T0), id)
  s.markRequested(id, T0)
  assert.equal(s.tick(T0), null)
})

test("manual compaction clears pending state", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  s.tick(T0)
  s.handle({ type: "session.compaction.started", data: { sessionID: id } }, T0)
  s.handle({ type: "session.compaction.ended", data: { sessionID: id } }, T0)
  assert.equal(s.get(id)?.pending, false)
  assert.equal(s.get(id)?.promptTokens, 0)
})

test("failed compaction retries with backoff", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  s.markRequested(id, T0)
  s.handle({ type: "session.compaction.failed", data: { sessionID: id } }, T0)
  assert.equal(s.tick(T0 + 1_000), null)
  assert.equal(s.tick(T0 + 11_000), id)
})

test("retries a request that never produces a compaction event", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  assert.equal(s.tick(T0), id)
  s.markRequested(id, T0)
  assert.equal(s.tick(T0 + 60_000), null)
  assert.equal(s.tick(T0 + 5 * 60_000 + 1), id)
})

test("a compaction stuck without ended/failed stops blocking", () => {
  const s = setup()
  const id = created(s, "ses_a")
  const other = created(s, "ses_b")
  runningStep(s, id, 900)
  runningStep(s, other, 900)
  idle(s, id)
  idle(s, other)
  assert.equal(s.tick(T0), id)
  s.markRequested(id, T0)
  s.handle({ type: "session.compaction.started", data: { sessionID: id } }, T0)
  assert.equal(s.tick(T0 + 60_000), null)
  const after = s.tick(T0 + 30 * 60_000 + 1)
  assert.ok(after === id || after === other)
})

test("keeps retrying after repeated failures without giving up", () => {
  const s = setup()
  const id = created(s)
  runningStep(s, id, 900)
  idle(s, id)
  let now = T0
  for (let i = 0; i < 6; i += 1) {
    s.markRequested(id, now)
    s.handle({ type: "session.compaction.failed", data: { sessionID: id } }, now)
    now += 10_000 * Math.min(i + 1, 3)
  }
  assert.equal(s.tick(now), id)
})

test("ignores sessions from other providers and subagents", () => {
  const s = setup()
  const cloud = created(s, "ses_cloud", "deepseek")
  const child = created(s, "ses_child")
  const childState = s.get("ses_child")
  if (childState) childState.parent = true
  runningStep(s, cloud, 900)
  runningStep(s, child, 900)
  idle(s, cloud)
  idle(s, child)
  assert.equal(s.tick(T0), null)
})
