# Connecting Synapse to LLM chat clients

Synapse exposes its running workflow over MCP (Model Context Protocol).
This means chat clients like **Claude Code** can compose and run workflows
in Synapse without you needing an API key — your existing chat subscription
is all that's required.

## Step 1 — Launch Synapse

Just run Synapse normally. On startup it prints a line like:

```
[mcp] server listening on 127.0.0.1:51780
```

The default port is **51780**. The current port is also written to
`~/.synapse/mcp-port` (`{"port": N}`) for tooling.

> If 51780 is already in use (e.g. a second Synapse instance), a random
> fallback port is chosen and printed instead — re-run `claude mcp add`
> with the new URL in that case.

## Step 2 — Add Synapse to Claude Code (one-time)

From a terminal where the `claude` CLI is installed:

```bash
claude mcp add synapse --transport http "http://127.0.0.1:51780/mcp"
```

You only need to run this once. Subsequent Synapse restarts re-use the
same port.

Verify with:

```bash
claude mcp list
```

You should see `synapse` listed.

## Step 3 — Use it

Start a Claude Code session and ask things like:

- "What clustering nodes are available?"
- "Build a fingerprint clustering pipeline for the CSV at /Users/me/molecules.csv"
- "Add a Murcko Scaffold node after the MolTable Reader."

Claude has access to 13 tools:

| Group | Tool | Purpose |
|---|---|---|
| Discovery | `list_nodes` | Catalog of every registered node (name + summary + category) |
| Discovery | `describe_node(node_type)` | Full details on a node type — ports + properties |
| Discovery | `search_nodes(query)` | Keyword search; fallback when the catalog is too large |
| Compose | `create_workflow(definition, run=False)` | Atomic one-shot — build a whole pipeline, optionally execute it |
| Inspect | `describe_graph()` | Snapshot of the current workflow |
| Modify | `add_node(node_type, properties?)` | Create a node in the current graph |
| Modify | `delete_node(node_id)` | Remove a node + any edges that touch it |
| Modify | `set_property(node_id, prop, value)` | Update a node's property |
| Modify | `connect(src_node_id, src_port, dst_node_id, dst_port)` | Wire two nodes |
| Modify | `disconnect(src_node_id, src_port, dst_node_id, dst_port)` | Remove a wire |
| Execute | `run_node(node_id)` | Evaluate a node (re-runs dirty upstream) |
| Execute | `get_node_status(node_id)` | Last known status of a node, no re-run |
| Execute | `get_node_output(node_id, port_name?)` | Read the data on a node's output port (table preview, image stats, etc.) |

## Limitations in v0

- **No Claude Desktop support yet** — Claude Desktop only speaks stdio
  to MCP servers; a small bridge script lands in v1.
- **No output filtering** — `get_node_output` is available in v0 as a
  preview (head 10 rows, image stats, etc.).  Advanced modes (range,
  filter, pandas-query) land in v1.
- **No workflow save/load via MCP** — you can still save manually in
  the GUI; tool support lands in v1.
- **No confirmation dialog** — every tool call is auto-allowed in v0.
  v1 adds an "Ask / Auto" setting that can prompt for destructive ops.

## Security

- The server binds only to `127.0.0.1`. It is not reachable from other
  machines on your network.
- No authentication in v0 — anyone running code on your machine could
  send requests to the port. If that's a concern, kill Synapse when not
  in use.
- Tool calls run with the same privileges as Synapse itself. They can
  read files Synapse's nodes can read; they cannot install packages or
  shell out unless a node already does so.
