import { Plugin } from "@opencode/plugin"
import type { Model, Provider } from "@opencode/plugin"
import type { ProviderEditor } from "@opencode/plugin/promise/provider"
import { appendFile, mkdir, readFile, stat, writeFile } from "node:fs/promises"
import { homedir } from "node:os"
import { dirname, join } from "node:path"

const PROVIDER_ID = "axonhub"
const INTEGRATION_ID = "axonhub"
const CACHE_FILE = join(homedir(), ".cache", "opencode", "axonhub-models.json")
const LOG_FILE = join(homedir(), ".cache", "opencode", "axonhub-plugin.log")
const CACHE_TTL = 24 * 60 * 60 * 1000
const HTTP_TIMEOUT = 15_000

const ENV_KEY = "AXONHUB_API_KEY"
const ENV_BASE_URL = "AXONHUB_BASE_URL"

// OpenCode V2 ships these provider entrypoints in-process, so an AxonHub model
// only has to declare which protocol its AxonHub endpoint speaks.
const PACKAGES = {
  anthropic: "@opencode/ai/providers/anthropic",
  google: "@opencode/ai/providers/google",
  openai: "@opencode/ai/providers/openai",
} as const

type PackageID = (typeof PACKAGES)[keyof typeof PACKAGES]

type ModelInfo = typeof Model.Info.Type
type ProviderInfo = typeof Provider.Info.Type
type ModelCost = ModelInfo["cost"][number]

type PluginOptions = {
  /** AxonHub origin, for example `https://axonhub.example.com`. Falls back to `AXONHUB_BASE_URL`. */
  baseURL?: string
  /** API key. Falls back to `AXONHUB_API_KEY`, then to the credential stored for the `axonhub` integration. */
  apiKey?: string
  /** Copy metadata (family, cost, limits, capabilities, variants) from OpenCode's model catalog. Defaults to true. */
  enrichModels?: boolean
  /** Append discovery diagnostics to `~/.cache/opencode/axonhub-plugin.log`. Defaults to true. */
  log?: boolean
}

type Logger = (message: string, extra?: Record<string, unknown>) => Promise<void>

type AxonHubCapabilities = {
  vision?: boolean
  tool_call?: boolean
  toolCall?: boolean
  reasoning?: boolean
}

type AxonHubPricing = {
  input?: number
  output?: number
  cache_read?: number
  cacheRead?: number
  cache_write?: number
  cacheWrite?: number
}

type AxonHubModel = {
  id?: string
  name?: string
  display_name?: string
  created?: number
  created_at?: string
  owned_by?: string
  context_length?: number
  max_output_tokens?: number
  capabilities?: AxonHubCapabilities
  pricing?: AxonHubPricing
}

type AxonHubResponse = {
  data?: AxonHubModel[]
}

type CatalogMatch = {
  providerID: string
  model: ModelInfo
}

function normalizeBaseURL(baseURL: string) {
  return baseURL.replace(/\/v1\/?$/, "").replace(/\/+$/, "")
}

function packageFor(owner: string, modelID: string): PackageID {
  if (modelID.startsWith("gemini-") || owner === "google" || owner === "gemini") return PACKAGES.google
  if (owner === "openai") return PACKAGES.openai
  return PACKAGES.anthropic
}

function endpointFor(baseURL: string, pkg: PackageID) {
  const clean = normalizeBaseURL(baseURL)
  if (pkg === PACKAGES.google) return `${clean}/gemini/v1beta`
  if (pkg === PACKAGES.openai) return `${clean}/v1`
  return `${clean}/anthropic/v1`
}

function released(item: AxonHubModel, template: ModelInfo | undefined) {
  if (item.created_at) {
    const parsed = Date.parse(item.created_at)
    if (Number.isFinite(parsed)) return parsed
  }
  if (item.created) return item.created * 1000
  return template?.time.released ?? 0
}

function copyCost(cost: ModelInfo["cost"]): ModelInfo["cost"] {
  return cost.map((item) => ({
    ...item,
    ...(item.tier ? { tier: { ...item.tier } } : {}),
    cache: { ...item.cache },
  }))
}

function copyVariants(variants: ModelInfo["variants"]): ModelInfo["variants"] {
  return variants.map((variant) => ({
    id: variant.id,
    ...(variant.settings === undefined ? {} : { settings: { ...variant.settings } }),
    ...(variant.headers === undefined ? {} : { headers: { ...variant.headers } }),
    ...(variant.body === undefined ? {} : { body: { ...variant.body } }),
  }))
}

function catalogIndex(editor: ProviderEditor) {
  const index = new Map<string, CatalogMatch[]>()
  for (const record of editor.list()) {
    for (const model of record.models.values()) {
      const match: CatalogMatch = { providerID: record.provider.id, model }
      for (const id of new Set([model.id, model.modelID])) {
        const existing = index.get(id)
        if (existing) existing.push(match)
        else index.set(id, [match])
      }
    }
  }
  return index
}

function catalogMatch(item: AxonHubModel, index: Map<string, CatalogMatch[]>) {
  if (!item.id) return
  const matches = index.get(item.id)
  if (!matches?.length) return

  const owner = item.owned_by
  return (
    (owner ? matches.find((match) => match.providerID === owner) : undefined) ??
    matches.find((match) => match.providerID === "opencode") ??
    matches.find((match) => match.providerID === "openai") ??
    matches[0]
  )
}

function buildModel(item: AxonHubModel, baseURL: string, match: CatalogMatch | undefined): ModelInfo | undefined {
  if (!item.id) return
  const id = item.id
  const owner = item.owned_by ?? ""
  const template = match?.model
  const pkg = packageFor(owner, id)
  const capabilities = item.capabilities

  const input = (modality: string) => template?.capabilities.input.includes(modality) === true
  const output = (modality: string) => template?.capabilities.output.includes(modality) === true
  const vision = capabilities?.vision ?? (template ? input("image") : true)
  const wantsPDF = template ? input("pdf") : true

  const pricing: ModelCost = {
    input: (item.pricing?.input ?? 0) as ModelCost["input"],
    output: (item.pricing?.output ?? 0) as ModelCost["output"],
    cache: {
      read: (item.pricing?.cache_read ?? item.pricing?.cacheRead ?? 0) as ModelCost["input"],
      write: (item.pricing?.cache_write ?? item.pricing?.cacheWrite ?? 0) as ModelCost["input"],
    },
  }
  const base = template?.cost[0]
  const merged: ModelCost = {
    input: base?.input ?? pricing.input,
    output: base?.output ?? pricing.output,
    cache: {
      read: base?.cache.read ?? pricing.cache.read,
      write: base?.cache.write ?? pricing.cache.write,
    },
  }

  return {
    id: id as ModelInfo["id"],
    modelID: id as ModelInfo["modelID"],
    providerID: PROVIDER_ID as ModelInfo["providerID"],
    name: item.name ?? item.display_name ?? template?.name ?? id,
    ...(template?.family === undefined ? {} : { family: template.family }),
    package: pkg,
    settings: { ...(template?.settings ?? {}), baseURL: endpointFor(baseURL, pkg) },
    ...(template?.headers === undefined ? {} : { headers: { ...template.headers } }),
    ...(template?.body === undefined ? {} : { body: { ...template.body } }),
    ...(template?.compatibility === undefined ? {} : { compatibility: { ...template.compatibility } }),
    capabilities: {
      tools: capabilities?.tool_call ?? capabilities?.toolCall ?? template?.capabilities.tools ?? true,
      input: Array.from(
        new Set([
          "text",
          ...(vision ? ["image"] : []),
          ...(input("audio") ? ["audio"] : []),
          ...(input("video") ? ["video"] : []),
          ...(wantsPDF ? ["pdf"] : []),
        ]),
      ),
      output: Array.from(
        new Set([
          ...(template ? (output("text") ? ["text"] : []) : ["text"]),
          ...(output("image") ? ["image"] : []),
          ...(output("audio") ? ["audio"] : []),
          ...(output("video") ? ["video"] : []),
          ...(output("pdf") ? ["pdf"] : []),
        ]),
      ),
    },
    variants: template ? copyVariants(template.variants) : [],
    time: { released: released(item, template) },
    cost: template?.cost.length ? [merged, ...copyCost(template.cost.slice(1))] : [merged],
    status: template?.status ?? "active",
    enabled: true,
    limit: {
      context: item.context_length ?? template?.limit.context ?? 200_000,
      ...(template?.limit.input === undefined ? {} : { input: template.limit.input }),
      output: item.max_output_tokens ?? template?.limit.output ?? 32_000,
    },
  }
}

function createLogger(enabled: boolean): Logger {
  return async (message, extra) => {
    if (!enabled) return
    try {
      await mkdir(dirname(LOG_FILE), { recursive: true })
      await appendFile(LOG_FILE, `${new Date().toISOString()} ${message}${extra ? ` ${JSON.stringify(extra)}` : ""}\n`)
    } catch {}
  }
}

async function readCache(freshOnly: boolean) {
  try {
    const info = await stat(CACHE_FILE)
    if (freshOnly && Date.now() - info.mtimeMs > CACHE_TTL) return
    return JSON.parse(await readFile(CACHE_FILE, "utf8")) as AxonHubResponse
  } catch {
    return
  }
}

async function writeCache(payload: AxonHubResponse) {
  await mkdir(dirname(CACHE_FILE), { recursive: true })
  await writeFile(CACHE_FILE, JSON.stringify(payload, null, 2))
}

async function fetchModels(baseURL: string, key: string, log: Logger) {
  const clean = normalizeBaseURL(baseURL)
  const headers = { Authorization: `Bearer ${key}` }
  await log("fetching AxonHub models", { baseURL: clean })
  const responses = await Promise.all(
    [`${clean}/v1/models`, `${clean}/v1/models?include=all`].map((url) =>
      fetch(url, { headers, signal: AbortSignal.timeout(HTTP_TIMEOUT) }),
    ),
  )

  const payloads: AxonHubResponse[] = []
  for (const response of responses) {
    if (!response.ok) {
      await log("AxonHub model endpoint failed", { status: response.status, url: response.url })
      continue
    }
    const payload = (await response.json()) as AxonHubResponse
    if (Array.isArray(payload.data)) payloads.push(payload)
  }
  if (payloads.length === 0) throw new Error("no AxonHub model payload was returned")

  const byID = new Map<string, AxonHubModel>()
  for (const payload of payloads) {
    for (const model of payload.data ?? []) {
      if (!model.id) continue
      byID.set(model.id, { ...byID.get(model.id), ...model })
    }
  }
  return { data: [...byID.values()] } satisfies AxonHubResponse
}

export default Plugin.define({
  id: "opencode-axonhub",
  async setup(ctx) {
    const options = (ctx.options ?? {}) as PluginOptions
    const baseURL = normalizeBaseURL(options.baseURL ?? process.env[ENV_BASE_URL] ?? "")
    const enrich = options.enrichModels ?? true
    const log = createLogger(options.log ?? true)

    let payload: AxonHubResponse | undefined
    let key: string | undefined

    const storedKey = async () => {
      // Credentials resolve through the integration, so `opencode auth login axonhub` and
      // `AXONHUB_API_KEY` both feed discovery without the plugin reading auth files.
      const connection = await ctx.integration.connection.active(INTEGRATION_ID)
      if (!connection) return
      const value = await ctx.integration.connection.resolve(connection)
      return value?.type === "key" ? value.key : undefined
    }

    // Returns whether the discovered catalog changed, so an unchanged credential
    // does not refetch or reload the provider.
    const discover = async () => {
      if (!baseURL) {
        await log("no AxonHub base URL configured; skipping model discovery")
        return false
      }
      const next = options.apiKey ?? process.env[ENV_KEY] ?? (await storedKey())
      if (!next) {
        const changed = key !== undefined || payload !== undefined
        key = undefined
        payload = undefined
        await log("no AxonHub API key available; skipping model discovery")
        return changed
      }
      if (next === key && payload) return false
      key = next
      const cached = await readCache(true)
      if (cached) {
        payload = cached
        await log("loaded AxonHub models from cache", { models: cached.data?.length ?? 0 })
        return true
      }
      try {
        payload = await fetchModels(baseURL, next, log)
        await writeCache(payload)
      } catch (error) {
        await log("AxonHub model discovery failed", { error: `${error}` })
        const stale = await readCache(false)
        if (!stale) return false
        payload = stale
      }
      await log("discovered AxonHub models", { models: payload.data?.length ?? 0 })
      return true
    }

    await ctx.integration.transform((editor) => {
      editor.update(INTEGRATION_ID, (integration) => {
        integration.name = "AxonHub"
      })
      editor.method.update({ integrationID: INTEGRATION_ID, method: { type: "key", label: "API Key" } })
      editor.method.update({ integrationID: INTEGRATION_ID, method: { type: "env", names: [ENV_KEY] } })
    })

    await discover()

    await ctx.provider.transform((editor) => {
      if (!payload?.data?.length || !baseURL) return
      const index = enrich ? catalogIndex(editor) : undefined
      const models = payload.data.flatMap((item) => {
        const model = buildModel(item, baseURL, index ? catalogMatch(item, index) : undefined)
        return model ? [model] : []
      })
      if (models.length === 0) return
      const info: ProviderInfo = {
        id: PROVIDER_ID as ProviderInfo["id"],
        name: "AxonHub",
        activation: "auto",
        integrationID: INTEGRATION_ID as ProviderInfo["integrationID"],
        package: PACKAGES.anthropic,
        settings: {},
      }
      editor.add({ info, models })
    })

    void (async () => {
      for await (const event of ctx.event.subscribe()) {
        if (
          event.type !== "credential.updated" &&
          event.type !== "credential.switched" &&
          event.type !== "integration.updated"
        )
          continue
        try {
          if (!(await discover())) continue
          await ctx.provider.reload()
          await log("reloaded AxonHub provider after credential change")
        } catch (error) {
          await log("AxonHub provider reload failed", { error: `${error}` })
        }
      }
    })().catch(() => {})
  },
})