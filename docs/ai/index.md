# AI in Synapse

Synapse integrates with large language models in two complementary ways. Both let an LLM read, modify, and run the workflow on your canvas — but they target different working styles, cost models, and use cases.

## Two AI surfaces, one graph

| | **In-app AI Chat** | **MCP server** |
|---|---|---|
| **Where the chat lives** | A dock panel inside Synapse | Your existing chat client (Claude Code, Claude Desktop, Antigravity, Gemini CLI) |
| **Cost model** | Pay-as-you-go via your own API key | Your existing chat subscription (Claude Pro / Max, ChatGPT Plus, etc.) |
| **Local LLM support** | Yes — Ollama, llama.cpp | Depends on the client (most popular ones are cloud-only) |
| **Best for** | Quick one-off edits, fully-local workflows, no chat subscription | Long iterative tasks, multi-step debugging, agentic workflows |
| **Setup** | Pick provider + paste API key | One-click auto-setup per client |
| **Tool count** | 6 | 17 |

Both surfaces drive the same underlying NodeGraph — you can mix them freely in the same session.

---

## Quick start

### Option A — In-app AI Chat (works offline with Ollama)

1. **View → AI Chat** to open the dock panel.
2. Pick a **Provider** from the dropdown. For zero setup with no API key, use **Ollama** — install it from [ollama.com](https://ollama.com), then `ollama pull gemma3:12b` once in a terminal.
3. Pick a **Model** from the auto-loaded list.
4. Type a request in the chat box. Example: *"Load the CSV in `~/data/cells.csv`, filter rows with area > 100, plot a histogram of area."*
5. Watch nodes appear on the canvas as the assistant builds the workflow. Hit **Run** when ready.

For cloud providers (Claude / OpenAI / Gemini / Groq / OpenRouter / RunPod), paste your API key into the same panel and pick a model. Keys are stored locally via your OS keyring and never sent anywhere except the selected provider.

### Option B — MCP from your existing chat client

1. Run Synapse normally. It starts an MCP server on `127.0.0.1:51780` by default.
2. **Help → AI Connection (MCP)...** opens the connection dialog.
3. Click the setup button for your chat client (**Claude Code**, **Claude Desktop**, **Antigravity**, or **Gemini CLI**). The dialog writes the right config for that client automatically.
4. Open your chat client and start a conversation. The chat client now sees Synapse and can call its tools.
5. Example prompt to try in Claude Code: *"What nodes are in the current Synapse graph? Add a Gaussian blur before the threshold and re-run."*

The MCP setup is a one-time action per client. After that, every Synapse restart re-uses the same port and the chat client reconnects automatically.

---

## Surface 1 — In-app AI Chat

A dock panel inside Synapse. The conversation history is **session-only** — it clears when you close Synapse.

### Providers

| Provider | API key | Notes |
|----------|---------|-------|
| **Ollama** | not needed | Local. `ollama pull <model>` once, then it's available offline. |
| **llama.cpp** | not needed | Local, point at a GGUF file. Lightest dependency footprint. |
| **Ollama Cloud** | needed | Same models as local Ollama but hosted. Get a key at [ollama.com/settings/keys](https://ollama.com/settings/keys). |
| **Claude** | needed | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | needed | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Gemini** | needed | Free tier at [aistudio.google.com](https://aistudio.google.com) |
| **Groq** | needed | Free tier at [console.groq.com](https://console.groq.com) |
| **OpenRouter** | needed | Multi-provider gateway. One key, many models. |
| **RunPod** | needed | Serverless vLLM endpoint. Enter your Endpoint ID. |

API keys are stored locally via the OS keyring (macOS Keychain / Windows Credential Manager / Linux Secret Service). They never leave your machine except in requests to the provider you selected.

### What the in-app chat can do

The assistant has six tools for working with your graph:

| Tool | What it does |
|------|--------------|
| `generate_workflow` | Build a fresh workflow from a description (best on an empty canvas). |
| `modify_workflow` | Surgically edit the existing graph: add nodes, change wires, set properties, delete branches. Use this when you already have a pipeline you want to extend. |
| `write_python_script` | When no dedicated node exists, write Python for the `Python Script` node. |
| `inspect_canvas` | Read what's currently on the canvas without changing it. |
| `explain_node` | Describe what a specific node does. |
| `read_node_output` | Fetch the output of a node that's already been run. |

The assistant decides which tool to call based on your request. You don't have to think about tools — just say what you want.

### Budget controls

To prevent runaway token usage, each turn is capped at **6 tool calls**. The assistant will give a concise summary instead of running endless tool loops. This is by design — the in-app chat is tuned for cost-conscious users with metered API keys.

For longer iterative tasks (deep debugging, multi-step refactors), use the MCP surface instead — your chat client's subscription absorbs the cost.

---

## Surface 2 — MCP

MCP (Model Context Protocol) is Anthropic's open standard for letting LLMs call tools running outside the chat. Synapse runs an MCP server on `127.0.0.1` that your chat client connects to.

### Supported clients

| Client | Transport | Setup |
|--------|-----------|-------|
| **Claude Code** | HTTP | One-click in the AI Connection dialog |
| **Claude Desktop** | stdio bridge | One-click — writes `claude_desktop_config.json` |
| **Antigravity** (Google) | HTTP (`serverUrl`) | One-click — writes `~/.gemini/antigravity/mcp_config.json` |
| **Gemini CLI** | HTTP (`httpUrl`) | One-click — writes `~/.gemini/settings.json` |

If your client isn't listed, you can connect manually — the dialog also shows the raw MCP URL (`http://127.0.0.1:51780/mcp`) and the `claude mcp add` command for copy-paste.

### What MCP can do

The MCP server exposes 17 tools, more than the in-app chat:

**Discovery:**
- `list_nodes` — what node types are registered
- `describe_node` — full spec for a single type (ports, properties, options, range)
- `search_nodes` — find nodes by keyword

**Inspection:**
- `describe_graph` — current canvas state
- `get_node_status` — has this node been run?
- `get_node_output` — fetch a node's output (preview / describe / range / filter modes)
- `get_node_image` — fetch a rendered figure or mask as an image the LLM can see

**Mutation:**
- `add_node`, `delete_node`, `replace_node`
- `set_property`
- `connect`, `disconnect`
- `create_workflow` — one-shot bulk build of nodes + wires

**Execution:**
- `run_node` — run a node and everything upstream of it

**Workflow management:**
- `new_workflow` — clear the canvas
- `save_workflow`, `load_workflow` — JSON files

### Configuring the port

The default port is **51780**, but you can change it in **Help → AI Connection (MCP)...**. Picking a port writes a preference to `~/.synapse/mcp-port-preference`. The chosen port is then printed at startup and saved to `~/.synapse/mcp-port` for tooling.

If the preferred port is busy (e.g., a second Synapse instance), Synapse falls back to a random port and prints the new URL.

### Full setup details

See [Connecting Synapse to LLM chat clients](../mcp/README.md) for client-specific configuration and troubleshooting.

---

## Privacy & cost

### What gets sent

When you use the **in-app chat**, the assistant sends to your selected provider:

- Your prompt
- A condensed catalog of available node types (around 6,000 tokens by default)
- For tool calls: the relevant graph state, plus images if you ask the assistant to look at a node's output

The condensed catalog is built locally from Synapse's registered nodes. With **Verbose node descriptions** enabled, full docstrings are sent instead — larger prompt, better accuracy.

When you use **MCP**, your chat client (Claude Code, Claude Desktop, etc.) decides what to send. Synapse only responds to the chat client's tool calls — it doesn't push data unsolicited.

In both modes, your raw scientific data (images, CSVs, tables) is only sent when you explicitly ask the assistant to look at it — for example, calling `get_node_image` or `read_node_output`. The data never leaves your machine unless a tool call references it.

### Cost models compared

- **In-app chat** is paid per token via your own API key. Cheap for short tasks; expensive for long iterative loops. Free with local providers (Ollama, llama.cpp). The 6-tool-call cap exists to prevent runaway charges.
- **MCP** uses your chat client's subscription (Claude Pro / Max, ChatGPT Plus, etc.). Unlimited iteration within the subscription's fair-use limits. No per-token billing on Synapse's side.

If you have a Claude Pro / Max subscription, MCP via Claude Code or Claude Desktop is usually the cheaper option for long sessions. If you don't have a subscription, the in-app chat with Ollama or llama.cpp keeps everything local and free.

---

## Tips for working with both surfaces

- **Be specific about files and paths.** *"Load `~/data/cells.csv`"* is easier than *"load that CSV I have."*
- **Reference node names verbatim.** If the assistant talks about a "Binary Threshold" node, the canvas should show one called *Binary Threshold*. The assistant calls `describe_node` to verify before wiring.
- **Iterate, don't restart.** Once you have a partial workflow, ask for incremental changes (*"add a Gaussian blur before the threshold"*) rather than re-generating from scratch. The assistant has a `modify_workflow` tool for this — using it preserves your tweaks.
- **Inspect before re-running.** If a node produces an unexpected result, ask the assistant to show you its output (`get_node_image` / `read_node_output`). The visual feedback loop is what makes the canvas-driven workflow valuable.
- **Mix the surfaces.** Build the initial workflow with the in-app chat (cheap on a fresh canvas), then switch to MCP via Claude Code for the long iterative debugging session. They drive the same graph.

## Limitations

- The assistant cannot click in 3D viewers, draw ROIs on images, or interact with custom widgets. For those, you still drive the GUI — then ask the assistant to handle the analysis downstream.
- Smaller local models (under 7B parameters) often struggle with multi-step tool use. If Ollama or llama.cpp gives flaky results, try a larger model or switch to a cloud provider.
- The in-app chat's 6-tool-call cap means complex multi-step debugging can time out before completing. Switch to MCP for those sessions.
- Workflows generated entirely by the assistant should still be reviewed before running on irreplaceable data. The LLM occasionally picks the wrong threshold direction or forgets a node — the canvas makes these easy to spot, but only if you look.

---

## See also

- [Connecting Synapse to LLM chat clients (MCP setup)](../mcp/README.md) — detailed per-client configuration
- [Creating Plugins](../developing/creating-plugins.md) — adding new nodes so the assistant has more tools to call
