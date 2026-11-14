import test from "node:test"
import assert from "node:assert/strict"
import { definition } from "./plugin.ts"

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function harness(models: Record<string, string> = {}) {
  const queue: any[] = []
  const compactions: string[] = []
  let stopped = false

  const ctx: any = {
    session: {
      get: async ({ sessionID }: { sessionID: string }) => ({
        model: {
          providerID: models[sessionID] ?? "v100",
          id: models[sessionID] === "deepseek" ? "x" : "m",
        },
      }),
      compact: async ({ sessionID }: { sessionID: string }) => {
        compactions.push(sessionID)
        return { sessionID }
      },
    },
    model: {
      list: async () => [
        { providerID: "v100", id: "m", limit: { context: 1000 } },
        { providerID: "deepseek", id: "x", limit: { context: 1000 } },
      ],
    },
    event: {
      subscribe: () => ({
        async *[Symbol.asyncIterator]() {
          while (!stopped) {
            if (queue.length) yield queue.shift()
            else await sleep(5)
          }
        },
      }),
    },
  }

  return {
    ctx,
    compactions,
    send: (type: string, data: any) => queue.push({ type, data }),
    wait: async (ms = 300) => await sleep(ms),
    stop: () => {
      stopped = true
    },
  }
}

async function start(h: ReturnType<typeof harness>, provider?: string) {
  process.env.DEFERRED_COMPACTION_POLL_MS = "50"
  if (provider) process.env.DEFERRED_COMPACTION_PROVIDER = provider
  else delete process.env.DEFERRED_COMPACTION_PROVIDER
  return await definition.setup(h.ctx)
}

test("compacts at a safe point after execution ends", async (t) => {
  const h = harness()
  const cleanup = await start(h)
  t.after(async () => { h.stop(); await cleanup?.() })
  h.send("session.created", { sessionID: "ses_a", model: { providerID: "v100", id: "m" } })
  await h.wait(50)
  h.send("session.execution.started", { sessionID: "ses_a" })
  h.send("session.step.ended", { sessionID: "ses_a", tokens: { input: 17, output: 2, cache: { read: 881 } }, finish: "stop" })
  h.send("session.execution.succeeded", { sessionID: "ses_a" })
  await h.wait()
  assert.deepEqual(h.compactions, ["ses_a"])
})

test("does not compact while a tool is in flight", async (t) => {
  const h = harness()
  const cleanup = await start(h)
  t.after(async () => { h.stop(); await cleanup?.() })
  h.send("session.created", { sessionID: "ses_b", model: { providerID: "v100", id: "m" } })
  await h.wait(50)
  h.send("session.execution.started", { sessionID: "ses_b" })
  h.send("session.tool.called", { sessionID: "ses_b", id: "call_1" })
  h.send("session.step.ended", { sessionID: "ses_b", tokens: { input: 900, output: 0 }, finish: "stop" })
  h.send("session.execution.succeeded", { sessionID: "ses_b" })
  await h.wait()
  assert.deepEqual(h.compactions, [])

  h.send("session.tool.success", { sessionID: "ses_b", id: "call_1" })
  await h.wait()
  assert.deepEqual(h.compactions, ["ses_b"])
})

test("ignores sessions from another provider by default", async (t) => {
  const h = harness({ ses_cloud: "deepseek" })
  const cleanup = await start(h)
  t.after(async () => { h.stop(); await cleanup?.() })
  h.send("session.created", { sessionID: "ses_cloud", model: { providerID: "deepseek", id: "x" } })
  await h.wait(50)
  h.send("session.execution.started", { sessionID: "ses_cloud" })
  h.send("session.step.ended", { sessionID: "ses_cloud", tokens: { input: 999, output: 0 }, finish: "stop" })
  h.send("session.execution.succeeded", { sessionID: "ses_cloud" })
  await h.wait()
  assert.deepEqual(h.compactions, [])
})

test("manages every provider when set to *", async (t) => {
  const h = harness({ ses_cloud: "deepseek" })
  const cleanup = await start(h, "*")
  t.after(async () => { h.stop(); await cleanup?.() })
  h.send("session.created", { sessionID: "ses_cloud", model: { providerID: "deepseek", id: "x" } })
  await h.wait(50)
  h.send("session.execution.started", { sessionID: "ses_cloud" })
  h.send("session.step.ended", { sessionID: "ses_cloud", tokens: { input: 999, output: 0 }, finish: "stop" })
  h.send("session.execution.succeeded", { sessionID: "ses_cloud" })
  await h.wait()
  assert.deepEqual(h.compactions, ["ses_cloud"])
})
