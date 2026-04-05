# Report Generation

### Report

Generates an HTML scientific report from upstream tables and figures.

??? note "Details"
    **Workflow:**
    
    - Connect table and/or figure inputs, then run the graph. The node collects all upstream data and prepares the report prompt.
    - Click **Generate with API** to use the configured LLM, or click **Copy for Web AI** to paste into ChatGPT / Claude.ai / Gemini.
    - If using web AI, paste the response into the **AI Response** box, then click **Build Report** to render the HTML.
    
    - **title** — report title (default: "Analysis Report").
    
    - **context** — optional text giving the LLM additional context (e.g. "This is a cell viability assay comparing drug A vs control").

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Input** | `figure` | figure |
| **Output** | `report` | html |

---

### Save HTML

Saves HtmlData to an HTML file on disk.

??? note "Details"
    Connect the **html** output from a Report node. Choose a save path
    and the node writes the self-contained HTML file.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `html` | html |

---

### Display HTML

Displays HtmlData content directly on the node surface.

??? note "Details"
    Connect the **html** output from a Report node to preview the
    generated report inline without opening a browser.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `html` | html |

---
