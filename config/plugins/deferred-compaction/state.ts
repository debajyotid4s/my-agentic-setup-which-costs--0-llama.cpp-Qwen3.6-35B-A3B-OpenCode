// Pure compaction scheduling state. No OpenCode imports, so it can be tested
// directly. The plugin feeds it V2 events and asks `tick()` who to compact.

export interface TokenUsage {
  input?: number
  output?: number
  reasoning?: number
  cache?: { read?: number; write?: number }
}

export interface SchedulerOptions {
  /** Compact when the last request's context reaches this share of the window. */
  thresholdRatio: number
  /** Proactive compaction: below the threshold, compact after an idle window. */
  prewarmRatio: number
  prewarmIdleMs: number
  /** Base retry delay; the multiplier is capped by maxBackoffSteps. */
  retryMs: number
  /** Backoff cap: delay = retryMs * min(failures, maxBackoffSteps). */
  maxBackoffSteps: number
  /** Retry a request that produced no compaction event after this long. */
  requestTimeoutMs: number
  /** Stop treating a compaction as active after this long. */
  compactionTimeoutMs: number
  /** Fallback only when model metadata is unavailable. */
  contextLimit: number
  /** Only sessions on this provider are managed ("*" = all providers). */
  providerID: string
}

export interface SessionState {
  id: string
  managed: boolean
  parent: boolean
  limit: number
  executing: boolean
  idle: boolean
  tools: Set<string>
  lastFinish: string | null
  promptTokens: number
  lastActivityAt: number
  pending: boolean
  requested: boolean
  requestedAt: number
  prewarmRequested: boolean
  compacting: boolean
  failures: number
  retryAt: number
}

export interface EventLike {
  type: string
  data?: {
    sessionID?: string
    parentID?: string
    model?: { providerID?: string; provider?: string; id?: string }
    status?: { type?: string }
    tokens?: TokenUsage
    finish?: string
    id?: string
    reason?: string
    [key: string]: unknown
  }
}

export const DEFAULTS: Omit<SchedulerOptions, "providerID"> = {
  thresholdRatio: 0.8,
  prewarmRatio: 0.7,
  prewarmIdleMs: 10 * 60 * 1000,
  retryMs: 60 * 1000,
  maxBackoffSteps: 3,
  requestTimeoutMs: 5 * 60 * 1000,
  compactionTimeoutMs: 30 * 60 * 1000,
  contextLimit: 262144,
}

// llama.cpp reports uncached prompt tokens in `input` and the reused prefix in
// `cache.read`; the next request carries both plus the new output.
export function estimatePromptTokens(tokens: TokenUsage | undefined | null): number {
  if (!tokens || typeof tokens !== "object") return 0
  const input = Number(tokens.input) || 0
  const cached = Number(tokens.cache?.read) || 0
  const output = Number(tokens.output) || 0
  return input + cached + output
}

function newSession(id: string, now: number): SessionState {
  return {
    id,
    managed: false,
    parent: false,
    limit: 0,
    executing: false,
    idle: true,
    tools: new Set<string>(),
    lastFinish: null,
    promptTokens: 0,
    lastActivityAt: now,
    pending: false,
    requested: false,
    requestedAt: 0,
    prewarmRequested: false,
    compacting: false,
    failures: 0,
    retryAt: 0,
  }
}

function isLocalModel(
  model: { providerID?: string; provider?: string } | undefined,
  providerID: string,
): boolean {
  if (providerID === "*") return true
  if (!model) return false
  const provider = model.providerID || model.provider
  return !providerID || provider === providerID
}

export class CompactionScheduler {
  readonly options: SchedulerOptions
  readonly sessions = new Map<string, SessionState>()
  activeCompaction: string | null = null
  activeSince = 0

  constructor(options: Partial<SchedulerOptions> & { providerID: string }) {
    this.options = { ...DEFAULTS, ...options }
  }

  ensure(sessionID: string, now = Date.now()): SessionState {
    let s = this.sessions.get(sessionID)
    if (!s) {
      s = newSession(sessionID, now)
      this.sessions.set(sessionID, s)
    }
    return s
  }

  setManaged(sessionID: string, managed: boolean): void {
    const s = this.sessions.get(sessionID)
    if (s) s.managed = managed
  }

  setLimit(sessionID: string, limit: number): void {
    const s = this.sessions.get(sessionID)
    if (s && Number(limit) > 0) s.limit = Number(limit)
  }

  /** Called after the scheduler hands a session to the caller. */
  markRequested(sessionID: string, now = Date.now()): void {
    const s = this.sessions.get(sessionID)
    if (!s) return
    s.requested = true
    s.requestedAt = now
    s.prewarmRequested = true
    s.lastActivityAt = now
  }

  markError(sessionID: string, now = Date.now()): void {
    const s = this.sessions.get(sessionID)
    if (!s) return
    s.failures += 1
    const steps = Math.min(s.failures, this.options.maxBackoffSteps)
    s.retryAt = now + this.options.retryMs * steps
    s.pending = true
    s.requested = false
    s.requestedAt = 0
    s.prewarmRequested = false
  }

  get(sessionID: string): SessionState | undefined {
    return this.sessions.get(sessionID)
  }

  private clearRequest(s: SessionState): void {
    s.requested = false
    s.requestedAt = 0
    s.prewarmRequested = false
  }

  private releaseActive(id: string): void {
    if (this.activeCompaction === id) this.activeCompaction = null
    this.activeSince = 0
  }

  handle(event: EventLike, now = Date.now()): void {
    if (!event || typeof event.type !== "string") return
    const data = event.data || {}
    const id = data.sessionID
    const type = event.type

    if (type === "session.created") {
      if (!id) return
      const s = newSession(id, now)
      s.parent = Boolean(data.parentID)
      s.managed = isLocalModel(data.model, this.options.providerID)
      this.sessions.set(s.id, s)
      return
    }

    if (type === "session.deleted") {
      if (!id) return
      this.sessions.delete(id)
      if (this.activeCompaction === id) {
        this.activeCompaction = null
        this.activeSince = 0
      }
      return
    }

    if (type === "session.model.selected") {
      if (!id) return
      const s = this.sessions.get(id)
      if (s) s.managed = isLocalModel(data.model, this.options.providerID)
      return
    }

    if (!id) return
    const s = this.sessions.get(id)
    if (!s) return

    switch (type) {
      case "session.execution.started":
        s.executing = true
        s.idle = false
        s.lastFinish = null
        s.lastActivityAt = now
        break
      case "session.execution.succeeded":
      case "session.execution.failed":
      case "session.execution.interrupted":
        // V2 does not always emit session.idle/session.status for API-created
        // sessions. A finished execution is the authoritative end of the loop.
        s.executing = false
        s.idle = true
        s.lastActivityAt = now
        break
      case "session.status": {
        const status = data.status?.type
        s.idle = status === "idle"
        if (!s.idle) s.lastActivityAt = now
        break
      }
      case "session.idle":
        s.idle = true
        s.lastActivityAt = now
        break
      case "session.tool.called":
        if (data.id) s.tools.add(data.id)
        s.lastActivityAt = now
        break
      case "session.tool.success":
      case "session.tool.failed":
        if (data.id) s.tools.delete(data.id)
        s.lastActivityAt = now
        break
      case "session.step.ended": {
        // A compaction summary is its own request; do not let its usage become
        // the next conversation baseline.
        if (!s.compacting) {
          const tokens = estimatePromptTokens(data.tokens)
          if (tokens > 0) s.promptTokens = tokens
          if (data.finish) s.lastFinish = data.finish
        }
        s.lastActivityAt = now
        break
      }
      case "session.compaction.started":
        s.compacting = true
        s.pending = false
        this.clearRequest(s)
        s.failures = 0
        s.retryAt = 0
        s.tools.clear()
        this.activeCompaction = id
        this.activeSince = now
        s.lastActivityAt = now
        break
      case "session.compaction.ended":
        s.compacting = false
        s.pending = false
        this.clearRequest(s)
        s.failures = 0
        s.lastFinish = null
        s.promptTokens = 0
        s.idle = true
        this.releaseActive(id)
        s.lastActivityAt = now
        break
      case "session.compaction.failed":
        s.compacting = false
        this.releaseActive(id)
        this.markError(id, now)
        break
      default:
        break
    }
  }

  /**
   * Returns the session ID that should be compacted now, or null.
   * A lost compaction event is retried instead of wedging the session.
   */
  tick(now = Date.now()): string | null {
    if (this.activeCompaction) {
      if (
        this.activeSince > 0 &&
        now - this.activeSince > this.options.compactionTimeoutMs
      ) {
        const stuck = this.sessions.get(this.activeCompaction)
        if (stuck) {
          stuck.compacting = false
          this.clearRequest(stuck)
        }
        this.releaseActive(this.activeCompaction)
      } else {
        return null
      }
    }

    for (const s of this.sessions.values()) {
      if (!s.managed || s.parent) continue
      if (s.compacting || s.executing || s.tools.size > 0) continue
      if (!s.idle) continue
      if (s.lastFinish !== "stop") continue

      if (s.requested && s.requestedAt > 0 && now - s.requestedAt > this.options.requestTimeoutMs) {
        this.clearRequest(s)
      }

      const limit = s.limit || this.options.contextLimit
      const threshold = limit * this.options.thresholdRatio
      const tokens = s.promptTokens
      if (tokens <= 0) continue

      if (tokens >= threshold) {
        s.pending = true
        if (s.requested) continue
        if (s.retryAt && now < s.retryAt) continue
        // Failures retry forever with a capped backoff.
        return s.id
      }

      // Hybrid: proactive compaction while the slot is idle.
      const prewarm = limit * this.options.prewarmRatio
      if (!s.requested && !s.prewarmRequested && tokens >= prewarm) {
        if (now - s.lastActivityAt >= this.options.prewarmIdleMs) return s.id
      }
    }
    return null
  }
}
