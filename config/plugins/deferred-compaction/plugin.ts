// V2 deferred / proactive compaction controller.
//
// Automatic compaction is disabled in opencode.json. This plugin watches the
// session state and requests compaction only when the agent is finished:
// execution has ended, no tool is running, the last step stopped normally, and
// the session is idle. Below the threshold it can also compact proactively after
// an idle window, so the summary does not land on the critical path.
//
// This file has no @opencode/plugin import so the controller can be exercised
// without the server. index.ts wraps it with Plugin.define.
//
// Environment overrides (mostly for testing):
//   DEFERRED_COMPACTION_PROVIDER   provider to manage (default: v100; "*" = all,
//                                  safe only when every provider reports
//                                  uncached input + cache.read like llama.cpp)
//   DEFERRED_COMPACTION_THRESHOLD  threshold ratio (default: 0.75)
//   DEFERRED_COMPACTION_PREWARM    prewarm ratio (default: 0.65)
//   DEFERRED_COMPACTION_IDLE_MS    idle window before prewarm (default: 600000)
//   DEFERRED_COMPACTION_POLL_MS    timer resolution (default: 15000)
import { CompactionScheduler, type EventLike } from "./state.ts"

interface SessionRef {
  model?: { providerID: string; id: string }
}

interface ModelRef {
  providerID: string
  id: string
  limit?: { context?: number }
}

export interface PluginContext {
  session: {
    get(input: { sessionID: string }): Promise<SessionRef | undefined>
    compact(input: { sessionID: string }): Promise<unknown>
  }
  model: {
    list(): Promise<readonly ModelRef[]>
  }
  event: {
    subscribe(options: { signal: AbortSignal }): AsyncIterable<EventLike>
  }
}

const num = (value: string | undefined, fallback: number): number => {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

export const definition = {
  id: "deferred-compaction",

  async setup(ctx: PluginContext): Promise<() => void> {
    const providerID = process.env.DEFERRED_COMPACTION_PROVIDER || "v100"
    const pollMs = num(process.env.DEFERRED_COMPACTION_POLL_MS, 15_000)
    const manages = (id: string) => providerID === "*" || id === providerID

    const scheduler = new CompactionScheduler({
      providerID,
      thresholdRatio: num(process.env.DEFERRED_COMPACTION_THRESHOLD, 0.75),
      prewarmRatio: num(process.env.DEFERRED_COMPACTION_PREWARM, 0.65),
      prewarmIdleMs: num(process.env.DEFERRED_COMPACTION_IDLE_MS, 10 * 60 * 1000),
    })

    const limits = new Map<string, number>()
    let pumping = false
    let stopped = false

    const limitFor = async (sessionID: string): Promise<void> => {
      const record = scheduler.get(sessionID)
      if (!record || record.limit) return
      try {
        const session = await ctx.session.get({ sessionID })
        const model = session?.model
        if (!model) return
        scheduler.setManaged(sessionID, manages(model.providerID))
        if (!manages(model.providerID)) return
        const key = `${model.providerID}/${model.id}`
        let limit = limits.get(key)
        if (!limit) {
          const models = await ctx.model.list()
          const found = models.find(
            (m) => m.providerID === model.providerID && m.id === model.id,
          )
          const context = found?.limit?.context
          if (context) {
            limit = context
            limits.set(key, context)
          }
        }
        if (limit) scheduler.setLimit(sessionID, limit)
      } catch (error) {
        console.log(`[deferred-compaction] could not resolve model limit: ${String(error)}`)
      }
    }

    const pump = async (): Promise<void> => {
      if (pumping || stopped) return
      pumping = true
      try {
        const sessionID = scheduler.tick()
        if (!sessionID) return
        scheduler.markRequested(sessionID)
        try {
          await ctx.session.compact({ sessionID })
          console.log(
            `[deferred-compaction] requested compaction for ${sessionID} ` +
              `(${scheduler.get(sessionID)?.promptTokens ?? "?"} tokens)`,
          )
        } catch (error) {
          scheduler.markError(sessionID)
          console.log(`[deferred-compaction] compaction request failed for ${sessionID}: ${String(error)}`)
        }
      } finally {
        pumping = false
      }
    }

    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          const type = event.type
          if (type === "model.updated" || type === "provider.updated") {
            limits.clear()
            continue
          }
          const data = event.data
          if (type === "session.created") {
            scheduler.handle(event)
            if (data?.sessionID) void limitFor(data.sessionID)
          } else if (data?.sessionID) {
            scheduler.ensure(data.sessionID)
            scheduler.handle(event)
            if (
              type === "session.step.ended" ||
              type === "session.idle" ||
              type === "session.execution.succeeded" ||
              type === "session.execution.failed" ||
              type === "session.execution.interrupted"
            ) {
              void limitFor(data.sessionID)
              void pump()
            }
          } else {
            scheduler.handle(event)
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          console.log(`[deferred-compaction] event stream ended: ${String(error)}`)
        }
      }
    })()

    const timer = setInterval(() => void pump(), pollMs)
    console.log(`[deferred-compaction] active (provider=${providerID}, poll=${pollMs}ms)`)

    return () => {
      stopped = true
      clearInterval(timer)
      controller.abort()
    }
  },
}
