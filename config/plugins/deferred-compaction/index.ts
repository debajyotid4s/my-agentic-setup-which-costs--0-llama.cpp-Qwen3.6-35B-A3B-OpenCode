// OpenCode V2 entrypoint. The controller lives in plugin.ts so it can be tested
// without importing the plugin SDK.
import { Plugin } from "@opencode/plugin"
import { definition } from "./plugin.ts"

export default Plugin.define(definition)
