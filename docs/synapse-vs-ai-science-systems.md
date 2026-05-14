# Synapse vs. Recent AI-for-Science Systems

A comparison of Synapse against four recent AI-for-science systems plus the broader 2024–2026 landscape, framed for an academic lab-group audience. Axes: **Architecture & UX**, **Domain coverage**, **Reproducibility & provenance**.

---

## 1. The systems at a glance

| System | Released | Built by | Form factor | Open / local? |
|---|---|---|---|---|
| **Synapse** | 2024– | (your lab) | PySide6 + NodeGraphQt visual node graph with MCP server | Open, fully local |
| **Biomni** | 2025-05 (bioRxiv) | SNAP lab, Stanford (Leskovec / Zou) | Agentic Python + Gradio + hosted web UI | Apache-2.0; local install heavy; hosted at biomni.stanford.edu |
| **scientific-agent-skills** | 2025-10 (repo) | K-Dense Inc. (Biostate AI spinout) | Library of ~135 file-based skills loaded into any Agent-Skills-compatible chat client | MIT; runs wherever the host agent runs |
| **AI Co-scientist** | 2025-02 | Google Research + DeepMind | Six-agent Gemini 2.0 system, hypothesis generator | Closed, cloud-only, Trusted Tester (applications closed) |

Plus the wider landscape (see §5): napari-MCP, napari-chatgpt/Omega, KNIME + AI Extension, Galaxy-MCP, BioImage.IO Chatbot, ChemCrow, FutureHouse Platform, Snakemaker, Code Interpreter, BioinfoMCP / MCPmed.

---

## 2. Axis 1 — Architecture & UX

The clearest split across the field is **who is in the driver's seat**: the human or the LLM.

### Human-in-the-driver-seat, LLM assists (Synapse's regime)

- **Synapse** — scientist authors a node graph in a visual canvas (NodeGraphQt). ROI drawing, mask overlays, click-to-segment (SAM2/Cellpose), inline figure previews are all done in the GUI. An MCP server exposes 17 tools (list/describe/add/connect/run nodes, fetch images, save workflows) so a chat client (Claude Code / Desktop / Antigravity / Gemini CLI) can read the graph, propose edits, and trigger execution — but the user owns the canvas.
- **napari-MCP** — same architectural pattern, applied to napari's layer-based viewer instead of a node graph.
- **KNIME + AI Extension** — closest commercial analogue: visual pipeline with AI nodes (LLM Prompter, Agent, RAG, plus governance nodes). Workflow Summarizer can introspect the graph itself.
- **napari-chatgpt / Omega** — chat agent embedded *inside* the GUI; can author new napari widgets on the fly.

### LLM in the driver's seat, no domain GUI

- **Biomni** — `agent.go("predict the role of TP53 in lung adenocarcinoma")`. LLM plans, writes Python, calls 300+ pre-curated bio tools, returns. Three entry points (Python API, Gradio demo, hosted web), but no graph, no spatial layout — the trace is the chat.
- **AI Co-scientist** — six-agent pipeline (Generation → Reflection → Ranking → Evolution → Proximity → Meta-review) running an Elo tournament over hypotheses. Scientist supplies a research goal and waits asynchronously. **It proposes; it does not run** — no execution, no images, no stats.
- **scientific-agent-skills** — not even an app: 135 skill directories that any Agent-Skills client loads on demand. Compute is the host agent's shell.
- **ChemCrow, FutureHouse Crow/Falcon/Phoenix/Robin, Code Interpreter** — all chat-driven agentic systems.

### Implication for academic audiences

For a microscopy researcher tweaking ROIs, sliders, and mask thresholds in real time, the GUI-in-the-loop regime is qualitatively different — and currently underserved. Most of the headline 2025 systems (Biomni, Co-scientist, K-Dense, ChemCrow, FutureHouse) abandon the GUI entirely and bet that an LLM transcript is sufficient. Synapse takes the opposite bet: **the LLM is the assistant, the canvas is the artifact.**

---

## 3. Axis 2 — Domain coverage

This is the axis where Synapse is most clearly behind — and where the answer should be "we know, and here's the roadmap."

### What the others claim

| System | Reported coverage |
|---|---|
| **Biomni** | Paper: 150 specialised tools + 105 biomedical software packages + 59 databases across 25 biomedical domains. GitHub tree shows 18 domain modules. Built-in connectors: UniProt, Ensembl, ClinVar, dbSNP, GWAS Catalog, GEO, gnomAD, InterPro, cBioPortal, EMDB, GtoPdb. Demonstrated: causal gene prioritisation, drug repurposing, rare-disease diagnosis, microbiome, scRNA-seq, CRISPR screen design, ADMET, wet-lab protocols. |
| **scientific-agent-skills** | 135 skills across 16+ domains: bioinformatics/genomics (~21 incl. Scanpy, BioPython, RNA velocity), cheminformatics (RDKit, DeepChem, DiffDock), ML/AI (PyTorch Lightning, scikit-learn, TimesFM), pathology, proteomics, MD (OpenMM), materials, engineering, data viz, 78–100+ scientific databases (PubMed, ChEMBL, UniProt…), lab automation (LabArchives), research methodology. |
| **Co-scientist** | Validated case studies in AML drug repurposing, liver fibrosis (Vorinostat, ~91% TGFβ chromatin reduction in liver microHOs), antimicrobial resistance / capsid-mediated horizontal gene transfer. Coverage is *reasoning* over biomedicine, not execution. |
| **Synapse (today)** | Confocal microscopy (3D segmentation, IMS reader, stitching, bleach correction, regionprops), image processing (ROI/crop/zoom/rotate/distance ring/rolling ball), statistics (linear/nonlinear regression, two-way ANOVA, contingency, survival, PCA, pairwise comparisons), ML (UMAP, clustering, train/test split, XGBoost, SHAP), cheminformatics (mol table, scaffolds, fingerprints incl. ECFP/MACCS/SECFP, batch docking, IUPAC↔SMILES via PubChem), plotting (XY line, heatmap, volcano, regression, survival), SAM2 + Cellpose plugins. |

### Honest read

- **Breadth gap is real.** Biomni's 314 actions and K-Dense's 135 skills span domains Synapse simply doesn't address: scRNA-seq atlases, MD trajectories, CRISPR screens, mass-spec proteomics, MaCS materials data, GWAS, ChEMBL queries, literature retrieval.
- **Depth in imaging is Synapse's lead.** None of Biomni, Co-scientist, or scientific-agent-skills has first-class confocal-microscopy support with interactive ROI drawing, click-to-segment SAM2/Cellpose, mask provenance through downstream stats, or per-step bleach correction. Image analysis appears only as thin "pathology" / "DeepChem" entries in K-Dense; Biomni's imaging coverage is shallow; Co-scientist doesn't process images at all.
- **The competitors are libraries; Synapse is a workbench.** A useful framing: their breadth measures "how many APIs can the LLM call?" Synapse's depth measures "how many things can a microscopist accomplish without writing code?"

### Roadmap framing (for the slide)

The breadth gap motivates concrete future work:
- **Database connectors** — wrap UniProt / Ensembl / ChEMBL / PubMed / GEO as MCP-callable nodes. Mostly straightforward; closes a visible gap.
- **scRNA-seq / Scanpy nodes** — the most-cited capability of both Biomni and K-Dense that Synapse lacks.
- **Literature retrieval node** — even a thin PaperQA2 / FutureHouse Owl wrapper would be visible value.
- **Protocol export** — emit a Synapse workflow → wet-lab protocol prose, mirroring Co-scientist's strength.
- **Skill bridge** — load K-Dense skills inside Synapse's MCP server (rather than competing, federate); a node like `RunSkill(name, args)` would absorb most of the K-Dense library essentially for free.

---

## 4. Axis 3 — Reproducibility & provenance

This is Synapse's clearest qualitative win, and the easiest to communicate to an academic audience.

### How each system handles "can a colleague rerun this?"

| System | Persisted workflow artifact | Deterministic re-run | Batch over folder/cohort | Audit trail |
|---|---|---|---|---|
| **Synapse** | **JSON graph** saved/loaded as a file | **Yes** — re-execute the saved graph; node properties are explicit; per-node `mark_dirty` propagation makes re-runs incremental | **Yes** — first-class `FolderIteratorNode` + `BatchAccumulatorNode` lifecycle | Node graph *is* the audit trail; each node's properties are visible and editable |
| **Biomni** | `agent.save_conversation_history()` → PDF transcript; emitted Python files | **No** — non-deterministic LLM planning; same prompt may yield a different plan | Not first-class; must ask the agent to loop | Chat log + emitted code |
| **scientific-agent-skills** | None at the library level | Host-dependent | Not first-class | Host agent's transcript |
| **Co-scientist** | "Context memory" persists hypotheses & tournament state across sessions | **No** — closed system, no replay API; outputs are natural-language protocols | N/A — proposes, doesn't run | Cited literature + meta-review summaries |
| **napari-MCP, ChemCrow, FutureHouse, Code Interpreter** | Transcript-bound | No | No | Chat log |
| **Galaxy-MCP** | **Yes** — Galaxy histories are citable artifacts | Yes — re-run a history | Yes | Galaxy's native provenance (one of the strongest in the field, predates the LLM era) |
| **KNIME + AI Extension** | Yes — `.knwf` workflow file | Yes | Yes | Workflow file |

### Why this matters in a lab-group talk

- **Determinism.** A non-deterministic LLM plan is unacceptable for figures going into a paper. Synapse's "open the graph, see exactly what was done, re-run, get the same numbers" is the property a PI cares about most. None of Biomni / Co-scientist / scientific-agent-skills provide this without external scaffolding.
- **Inspection.** A reviewer can open a Synapse `.workflow.json` and read the analysis. A Biomni session is a chat transcript; a Co-scientist run is closed cloud state.
- **Batch.** Lab pipelines run over cohorts. Synapse has `FolderIteratorNode` + `BatchAccumulatorNode` as a first-class abstraction with `on_batch_start` / `on_batch_end` lifecycle. The agentic systems require the LLM to remember to loop, which it does inconsistently.
- **Companion observation.** Galaxy and KNIME are the existing tools that nail reproducibility. Synapse is positioned next to them but with a *modern* LLM interface (MCP) and a *desktop-imaging* focus that neither targets.

---

## 5. Wider landscape (one-liners for context slide)

In rough order of architectural similarity to Synapse:

- **napari-MCP** (Royer lab) — same MCP-bridges-a-local-GUI pattern, applied to napari's layer viewer.
- **napari-chatgpt / Omega** (Royer lab) — embedded chat agent that can author napari widgets at runtime.
- **KNIME + AI Extension** — mature commercial visual node graph with AI nodes + governance.
- **Galaxy-MCP** — bioinformatics pipeline tool + 21-tool MCP server; gold standard for provenance.
- **BioImage.IO Chatbot** (*Nat. Methods*, 2024) — federated bioimaging assistant across Model Zoo, DeepImageJ, image.sc, napari docs.
- **ChemCrow** (*Nat. Mach. Intell.*, 2024) — 18 chemistry tools wrapped behind an LLM; planned & ran real syntheses.
- **FutureHouse Platform** (Crow / Falcon / Owl / Phoenix / Finch / Robin) — hosted multi-agent science suite; aviary training gym is open source, agents are closed.
- **Snakemaker** — turns ad-hoc notebooks/shell commands *into* Snakemake pipelines; complementary, not competing.
- **Claude / ChatGPT Code Interpreter** — the zero-setup baseline; no persistent workflow, no domain tools.
- **BioinfoMCP / MCPmed** (arXiv 2510.02139, PMC 2025) — proposals to standardise MCP as the bioinformatics interface layer; direct context for *why* Synapse's MCP server matters.

---

## 6. Synapse's positioning statement (one slide)

> **Most 2025 AI-for-science systems give the LLM the steering wheel and the scientist a chat box.** Biomni, scientific-agent-skills, Co-scientist, ChemCrow, and FutureHouse all bet on the agent — broad pre-built tool coverage, no GUI, transcript-as-artifact.
>
> **Synapse takes the opposite bet:** the scientist owns a visual node graph, the LLM is a collaborator that reads, edits, and runs the graph through MCP. The workflow is a file. The execution is deterministic. The audit trail *is* the canvas.
>
> **Where the others lead** — breadth of pre-built scientific tools — is exactly where Synapse's roadmap goes next: database-connector nodes, scRNA-seq, literature retrieval, and a thin bridge so K-Dense's 135 skills become callable inside a Synapse graph.

---

## 7. Suggested slide structure (10 slides)

1. **Title** — Synapse: an MCP-native visual workflow editor for science.
2. **Why now** — LLM-for-science exploded in 2025 (one figure: timeline of Biomni / Co-scientist / scientific-agent-skills / ChemCrow / FutureHouse).
3. **Two architectural bets** — agent-as-driver vs. scientist-as-driver-with-LLM-co-pilot. Where Synapse sits.
4. **The Synapse demo** — show a confocal pipeline: ROI → SAM2 → mask props → stats → volcano plot. Then a Claude Code session that adds a survival-analysis branch via MCP.
5. **Axis 1 — Architecture & UX** — the table from §2.
6. **Axis 2 — Domain coverage** — the table from §3, with the honest gap acknowledgement.
7. **Axis 3 — Reproducibility & provenance** — the table from §4. Emphasise determinism + batch + inspectable artifact.
8. **Where Synapse leads** — interactive imaging (ROI drawing, click-to-segment), deterministic re-execution, fully-local data residency, MCP that drives an actual GUI.
9. **Where the field leads, and our roadmap** — db connectors, scRNA-seq, literature retrieval, skill-bridge to K-Dense.
10. **Closing** — Synapse is to KNIME + Galaxy what Claude Code is to the terminal: the same proven workflow primitive, now LLM-native.

---

## Source list

**Biomni** — bioRxiv 10.1101/2025.05.30.656746 (May 30 2025); github.com/snap-stanford/Biomni; biomni.stanford.edu

**scientific-agent-skills** — github.com/K-Dense-AI/scientific-agent-skills (created Oct 19 2025; v2.38.0 May 2026); k-dense.ai; Biostate AI K-Dense Beta launch Sept 17 2025

**AI Co-scientist** — Google Research blog Feb 19 2025; arXiv:2502.18864 *Towards an AI co-scientist* (Gottweis et al., Feb 26 2025); Guan et al., bioRxiv 2025.04.29.651320 / *Advanced Science* 2025

**Others** — napari-mcp (royerlab); napari-chatgpt (royerlab); KNIME AI Extension docs; galaxyproject/galaxy-mcp; BioImage.IO Chatbot *Nature Methods* 2024; ChemCrow *Nat. Mach. Intell.* 2024; futurehouse.org platform announcements; BioinfoMCP arXiv:2510.02139; MCPmed PMC 12927880.
