# OpenClaw integration — validated configuration

This guide assumes you've followed [`dgx-spark-setup.md`](dgx-spark-setup.md) and have the vLLM server up at `http://<spark-ip>:8000/v1` with the 3 served-model-names visible.

[OpenClaw](https://github.com/openclaw/openclaw) by [@steipete](https://github.com/steipete) is an embedded agent runtime (not an HTTP proxy). It calls upstream OpenAI-compatible APIs and surfaces them to messaging channels, CLI, TUI, etc. LLM I/O is delegated to [`@mariozechner/pi-ai`](https://github.com/badlogic/pi-mono).

The config below was **strict-validated** against OpenClaw's actual zod schemas (every object is `.strict()` so unknown keys fail loudly).

---

## Why two model entries (qwen36-fast + qwen36-deep)?

DFlash speculative decoding's acceptance rate depends on the target's sampling distribution matching the drafter's:

- **Greedy (T=0)**: target = drafter argmax → ~80% first-position acceptance → ~3× speedup → ~91 tok/s single-stream
- **Sampled (T=0.7)**: target picks random token from distribution → drafter rarely guesses → ~5% acceptance → speedup collapses → ~38 tok/s single-stream

Solution: register **two model entries** pointing to the same backend, each with a different default `params.temperature`. Route per workload:

| Workload | Mode |
|---|---|
| Tool calls / agent loops | **fast** (greedy) |
| Code generation | **fast** |
| Math / structured reasoning | **fast** |
| JSON / schema output | **fast** |
| Creative writing | **deep** (sampled) |
| Brainstorming / ideation | **deep** |
| Open-ended Q&A | **deep** |

---

## Config — `~/.openclaw/openclaw.json`

OpenClaw config file location: `~/.openclaw/openclaw.json` (override via `OPENCLAW_CONFIG_PATH`). Format is **JSON5** (comments and unquoted keys allowed).

```json5
{
  models: {
    providers: {
      vllm: {
        baseUrl: "http://192.168.x.x:8000/v1",   // ← your Spark IP
        apiKey: "${VLLM_API_KEY}",                 // any non-empty string for vLLM
        api: "openai-completions",                 // vLLM uses /v1/chat/completions

        // OpenClaw forces these for openai-completions backends automatically:
        //   compat.supportsDeveloperRole = false
        //   strips service_tier, store, prompt-cache hints, attribution headers

        models: [
          {
            id: "qwen36-fast",
            name: "Qwen3.6-35B-A3B Heretic - Fast (greedy + DFlash)",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },  // self-hosted, free
            contextWindow: 262144,                 // model native max (256K)
            contextTokens: 245760,                 // effective cap (240K, leave 16K for output)
            maxTokens: 32768,                      // 32K max output per request
            compat: {
              thinkingFormat: "qwen-chat-template",  // sends chat_template_kwargs.enable_thinking
              supportsReasoningEffort: false,        // qwen3 chat template uses on/off
              maxTokensField: "max_tokens",
              requiresStringContent: false,
              supportsStrictMode: false
            }
          },
          {
            id: "qwen36-deep",
            name: "Qwen3.6-35B-A3B Heretic - Deep (sampled, creative)",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 262144,
            contextTokens: 245760,
            maxTokens: 32768,
            compat: {
              thinkingFormat: "qwen-chat-template",
              supportsReasoningEffort: false,
              maxTokensField: "max_tokens",
              requiresStringContent: false,
              supportsStrictMode: false
            }
          }
        ]
      }
    }
  },

  agents: {
    defaults: {
      // Per-model param defaults. Merge order:
      //   agents.defaults.params (global) →
      //     agents.defaults.models.<provider/id>.params (per-model) →
      //       agents.list[].params (per-agent)
      models: {
        "vllm/qwen36-fast": {
          alias: "fast",
          params: {
            temperature: 0,                      // greedy → ~80% DFlash acceptance, ~91 tok/s
            top_p: 1.0
          }
        },
        "vllm/qwen36-deep": {
          alias: "deep",
          params: {
            temperature: 0.7,                    // canonical Qwen3.6 sampling
            top_p: 0.95,
            top_k: 64,
            repetition_penalty: 1.05
          }
        }
      },

      thinkingDefault: "low",                    // valid enum: off|minimal|low|medium|high|xhigh|adaptive
      model: { primary: "vllm/qwen36-fast" }    // default if no agent-specific binding
      // NOTE: reasoningDefault and fastModeDefault belong on agents.list[] entries, not defaults
    },

    list: [
      {
        id: "agent-fast",
        model: "vllm/qwen36-fast",
        params: { temperature: 0 },
        reasoningDefault: "stream",              // stream <think> blocks live to client
        fastModeDefault: false                   // pi-ai's "fast mode" is a different concept
      },
      {
        id: "agent-deep",
        model: "vllm/qwen36-deep",
        params: { temperature: 0.7, top_p: 0.95, top_k: 64, repetition_penalty: 1.05 },
        reasoningDefault: "stream",
        fastModeDefault: false
      }
    ]
  }

  // bindings: [...]   // optional — see "Routing" below
}
```

---

## Validation

OpenClaw ships a CLI subcommand for this:

```bash
openclaw config validate --json
```

It runs `OpenClawSchema.safeParse` (`src/config/validation.ts:599`). Failure aborts startup with a list of issues — no silent key-drop. Print the JSON Schema with:

```bash
openclaw config schema
```

---

## Routing — picking which mode per request

### A. Client picks model name

Most reliable. Just have the client send `"model": "vllm/qwen36-fast"` or `"vllm/qwen36-deep"` per request.

### B. `/model` slash command in chat

Built in. User types `/model fast` to swap during a session.

### C. Channel/peer bindings

Add a `bindings` array at the top level:

```json5
bindings: [
  // Agentic Telegram bot → fast mode
  { channel: "telegram", peer: { kind: "chat", id: "<chatId>" }, agent: "agent-fast" },

  // Creative WhatsApp group → deep mode
  { channel: "whatsapp", peer: { kind: "group", id: "<groupId>" }, agent: "agent-deep" },

  // Slack workspace default → fast
  { channel: "slack", teamId: "T123", agent: "agent-fast" },

  // Global default
  { channel: "*", accountId: "*", agent: "agent-fast" }
]
```

Match priority (per `docs/gateway/configuration-reference.md:1740+`):
peer → guild → team → exact accountId → wildcard accountId → default agent

---

## How reasoning content is surfaced

OpenClaw's pi-ai dependency reads streaming response deltas and checks fields **in this exact order**:
1. `reasoning_content`
2. `reasoning`
3. `reasoning_text`

The first non-empty field is used. **There is no config knob to override the field order** — handled in [`packages/ai/src/providers/openai-completions.ts:186-222`](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-completions.ts).

To control whether reasoning is **requested**, set `compat.thinkingFormat` per model:

| `thinkingFormat` | What gets sent |
|---|---|
| `openai` | `reasoning_effort: "..."` |
| `openrouter` | `reasoning: { effort: "..." }` |
| `zai` | top-level `enable_thinking: true` |
| `qwen` | top-level `enable_thinking: true` |
| `qwen-chat-template` | `chat_template_kwargs: { enable_thinking: true }` ← **what we use** |

Reasoning emerges in pi-ai's stream as `thinking_start` / `thinking_delta` / `thinking_end` events.

---

## Auto-discovery shortcut (alternative)

If you set `VLLM_API_KEY` env var and **don't** define `models.providers.vllm`, OpenClaw queries `GET /v1/models` and auto-creates entries. Default base URL is `http://127.0.0.1:8000/v1`. With our 3 served-model-names, it'd auto-register all three.

But you'd lose:
- Per-model `compat.thinkingFormat` (so reasoning won't be requested)
- Per-model default `params` (so fast/deep can't be set as a default per model)

So the explicit config above is preferred for our two-mode setup.

---

## Quick test once OpenClaw is running

```bash
# OpenClaw should be running and connected to vLLM. From a terminal:
openclaw chat --model vllm/qwen36-fast "Compute 17 × 23. Show your work."
# Expect a streamed reasoning trace + final answer

openclaw chat --model vllm/qwen36-deep "Write a haiku about distributed systems."
# Expect creative output, may differ between runs
```

---

## Sources / source-of-truth links

- OpenClaw repo: https://github.com/openclaw/openclaw
- vLLM provider docs: [`docs/providers/vllm.md`](https://github.com/openclaw/openclaw/blob/main/docs/providers/vllm.md)
- Configuration reference (provider schema, bindings): [`docs/gateway/configuration-reference.md`](https://github.com/openclaw/openclaw/blob/main/docs/gateway/configuration-reference.md)
- Pi integration architecture: [`docs/pi.md`](https://github.com/openclaw/openclaw/blob/main/docs/pi.md)
- Authoritative zod schemas: `src/config/zod-core.ts`, `src/config/zod-schema.agent-runtime.ts`, `src/config/zod-schema.agent-defaults.ts` in the OpenClaw repo
- pi-ai openai-completions provider (reasoning auto-detection): [`packages/ai/src/providers/openai-completions.ts:186-222`](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-completions.ts)
