# LLM Wiki

> AI research knowledge base — papers, concepts & intuitions
> Curated by Muhammad Saqlain Aslam

**🌐 Live Web Demo:** https://MuhammadSaqlainAslam.github.io/my-llm-wiki

---

## Overview

This is a personal AI/ML research knowledge base built from academic papers, concept glossaries, and cross-linked notes. Every paper is distilled into a structured Obsidian note with YAML frontmatter (title, authors, year, arXiv ID, citation count, tags, TL;DR), a full intuition-first explanation, and `[[wikilinks]]` connecting it to every related concept. The result is a single navigable graph where clicking any node — whether a paper or a sub-concept — reveals exactly how it fits into the broader ecosystem.

The writing philosophy is borrowed from Andrej Karpathy: lead with the concrete problem and why it matters, introduce the math only after the intuition is clear, and always give real numbers. Mamba doesn't just "improve inference throughput" — it delivers 5× throughput over a Transformer at the same quality. FlashAttention doesn't just "reduce memory" — it tiles attention into SRAM so the T×T matrix never touches HBM, saving 2–4× wall-clock time at zero approximation cost. Every note is written to that standard.

What makes this wiki distinct is a full citation intelligence layer on top of the content. Every core paper carries its real citation count from Semantic Scholar, a `cited_by_details` list of the ten most-cited downstream works, and arXiv IDs throughout. The live web demo exposes all of this through a Citation Explorer, an interactive D3 knowledge graph sized by citation count, and a full-text search interface — all from a single static site deployed automatically via GitHub Actions to GitHub Pages.

---

## 🌐 Live Web Demo Features

### Browse & Search

- Full-text search across all 138 notes by title, tag, or keyword
- Multi-tag AND filtering — click multiple tags to narrow results simultaneously
- Note cards showing title, year badge, citation count badge (🔖 N citations), tags, and 3-line TL;DR
- Sort by title A–Z, year (newest first), or citation count descending
- Side panel opens the full note — rendered Markdown with KaTeX math, live `[[wikilink]]` resolution, and clickable cross-references

### Citation Explorer

A master-detail two-panel layout — click the **Citation Explorer** tab to open it.

**Left panel** — ranked list of every paper with `citation_count > 0`, sorted descending:
- Each row: `#rank · title · year · proportional citation bar · count`
- Bar width proportional to citations (`citation_count / 100,000`), minimum 4 px so even small counts are visible
- Bar color by impact tier:
  - 🟣 Purple → 10,000+ citations
  - 🩵 Teal → 1,000–9,999 citations
  - 🔵 Blue → 100–999 citations
  - ⬜ Gray → under 100 citations
- Filter box at top to search the ranked list in real time

**Right panel — default state** (nothing clicked): summary dashboard showing:
- Total papers in wiki
- Total citations tracked
- Most cited paper
- Fastest growing paper (Mamba: 4,841 citations in 18 months)
- Citation cluster pills: Foundations · Efficiency · Inference · Hardware

**Right panel — paper selected**:
- Title, authors, year, and arXiv badge with external link
- Large citation count number + full-width color bar
- **What this paper introduced** — the note's TL;DR
- **Top 10 papers that cite this work** — ranked table with: title (clickable), year, ~citations, theme badge, arXiv ↗ link
  - Clicking a citing paper's title: loads its detail view if it exists in the wiki, otherwise opens `arxiv.org/abs/[id]` in a new tab
- **Related papers in this wiki** — all resolved `[[wikilinks]]` from the note shown as clickable pills; clicking one switches to the All tab and opens that note's detail panel
- **← All Papers** back button returns to the summary dashboard

### Knowledge Graph

- Interactive D3.js v7 force-directed graph (`graph.html`)
- Every node = one wiki note, sized logarithmically by `citation_count`; nodes without citation data sized by incoming link count
- Every edge = a `[[wikilink]]` connection between notes
- Color-coded by theme tag (foundations, efficiency, hardware, inference-optimization, scaling, synthesis, meta)
- Click any node → opens full note detail in the side panel
- Hover a node → highlights direct neighbors, dims everything else; tooltip shows title, year, citations, and TL;DR
- Filter by tag — dims non-matching nodes
- Legend: color = theme, node size = citation count (where available, else link count)
- Zoom, pan, drag nodes freely

### Theme Navigation

Notes grouped by theme in the **By Theme** tab:

| Theme | Contents |
|---|---|
| Foundations | Self-attention, RLHF, open models |
| State Space Models | SSMs, linear RNNs, LSTM variants |
| Mixture-of-Experts | Sparse routing, load balancing |
| Inference Optimization | Speculative decoding, KV cache, draft models |
| Hardware & Systems | GPUs, TPUs, FPGAs, ASICs, LPUs, IO-aware kernels |
| Attention & KV | Attention variants, KV cache techniques |
| Training & RL | RLHF, RLVR, GRPO, optimizers |
| Benchmarks | Frontier model comparisons |
| Concepts & Glossary | 100+ sub-concept stub notes |

### Dark / Light Mode

- Toggle between dark (default) and light themes via the ☾ button
- Preference persisted in `localStorage`
- All components respond: cards, graph, citation bars, side panel

### Math Rendering

- KaTeX renders LaTeX inline (`$O(n^2)$`) and display (`$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$`)
- Math is protected during Markdown preprocessing so the renderer never mangles LaTeX delimiters
- Supports `$...$`, `$$...$$`, `\(...\)`, `\[...\]` delimiters

---

## 📚 Wiki Contents

138 notes total — 20 full paper articles + 118 concept and meta notes.

### Papers

#### Foundations

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Attention Is All You Need | 2017 | ~100,000 | 1706.03762 |
| LLaMA 2 | 2023 | ~10,000 | 2307.09288 |

#### Efficient Sequence Modeling

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| S4: Structured State Spaces | 2022 | ~3,000 | 2111.00396 |
| RWKV | 2023 | ~2,000 | 2305.13048 |
| RetNet | 2023 | ~1,000 | 2307.08621 |
| Mamba | 2023 | ~4,841 | 2312.00752 |
| Griffin | 2024 | ~1,000 | 2402.19427 |
| Transformers are SSMs (Mamba-2) | 2024 | ~1,200 | 2405.21060 |
| xLSTM | 2024 | ~800 | 2405.04517 |

#### Scaling / Parameter Efficiency

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Switch Transformers / Mixtral | 2022 | — | — |

#### Hardware & Systems

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| FlashAttention | 2022 | ~8,000 | 2205.14135 |
| FlashAttention-2 | 2023 | ~4,000 | 2307.08691 |
| Hardware Acceleration for Neural Networks | 2025 | ~120 | 2512.23914 |

#### Inference Optimization

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Medusa | 2024 | ~1,500 | 2401.10774 |
| EAGLE | 2024 | ~1,200 | 2401.15077 |
| Speculative Decoding | 2025 | ~600 | 2601.11580 |
| KV Cache Optimization | 2026 | ~200 | 2603.20397 |

#### Modern Synthesis

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Nemotron-3 | 2025 | — | — |
| DeepSeek-V4 | 2026 | — | — |
| LLM Benchmarks | 2025 | — | — |

### Concept Glossary

Over 100 concept stub notes cover every sub-component referenced across the paper articles, organised into six clusters: **Attention & Transformers** (multi-head attention, GQA, MQA, RoPE, XPOS, online softmax, tiling, recomputation, IO-awareness, warp, occupancy, thread block, linear attention, local attention, sliding window attention, attention sinks, CSA, HCA); **State Space Models** (State Space Model, HiPPO, HiPPO matrix, Cauchy kernel, discretization, convolutional view, Selective SSM, semiseparable matrix, multi-head SSM, retention mechanism, chunkwise recurrent, RG-LRU, LRU, exponential gating, matrix memory, covariance update rule, sLSTM, mLSTM, Long Range Arena); **Inference Optimization** (KV Cache, paged attention, KV offloading, prefill, decode phase, cache eviction, cache compression, hybrid memory, eviction policy, H2O eviction, draft model, verification step, lookahead decoding, tree attention, Medusa heads, Medusa-1, Medusa-2, lossless speedup, feature-level drafting, one-step token advance, multi-token prediction); **Hardware & Systems** (GPU, TPU, NPU, FPGA, ASIC, LPU, systolic array, tensor cores, HBM, SRAM, memory hierarchy, kernel fusion, operator fusion, hardware-aware scan, NVFP4, pallas kernel, in-memory computing, near-memory computing, neuromorphic computing); **Training & Optimization** (RLHF, RLVR, GRPO, ghost attention, load balancing loss, on-policy distillation, Muon optimizer, tensor parallelism, sequence parallelism); **Architecture Components** (LatentMoE, Manifold-Constrained Hyper-Connections, selective state spaces, LLM evaluation, LLM benchmarks). Every concept links back to the paper where it originates, and every `[[wikilink]]` in a paper note resolves to a concept page — 0 orphan links.

### Citation Intelligence

The wiki includes a full citation intelligence layer: `wiki/Citation Map.md` tracks total Semantic Scholar citation counts for the six core papers plus their top-10 highest-impact downstream works. Every core paper's YAML frontmatter carries `citation_count` (integer), `arxiv` (arXiv ID string), `cited_by_top` (name list), and `cited_by_details` (list of dicts: title, year, citations, theme, arxiv) — all emitted by `build.py` into `notes.json` and consumed by the Citation Explorer in the web demo. The `wiki/🗺️ Dashboard.md` Obsidian dashboard includes a live **Citation Leaderboard** Dataview query. Papers with full `cited_by_details`: Attention Is All You Need, Mamba, xLSTM, Speculative Decoding, KV Cache Optimization, Hardware Acceleration for Neural Networks.

---

## Repository Structure

```
my-wiki/
├── wiki/                          # 138 Markdown notes (Obsidian vault)
│   ├── Home.md                    # Entry point with concept map + reading paths
│   ├── 000 Index.md               # Full index: themes, glossary, narrative, citation table
│   ├── Citation Map.md            # Citation scores + top-10 citing papers per core paper
│   ├── 🗺️ Dashboard.md           # Live Dataview queries (Obsidian only)
│   ├── Mamba.md                   # Paper notes (20 total)
│   ├── Transformer.md
│   ├── ...
│   ├── KV Cache.md                # Concept stub notes (100+ total)
│   ├── HBM.md
│   └── ...
├── docs/                          # Static web demo (auto-deployed)
│   ├── index.html                 # Browse, search, Citation Explorer
│   ├── graph.html                 # D3 knowledge graph
│   └── notes.json                 # All notes as structured JSON
├── raw/                           # Source PDF papers
├── build.py                       # wiki/*.md → docs/notes.json + HTML
├── build_wiki.py                  # PDF → wiki note (Claude API)
├── .github/workflows/deploy.yml   # Auto-deploy to GitHub Pages on push
└── README.md
```

---

## Workflow

```
PDF papers in raw/
      ↓
build_wiki.py  —  Claude API extracts & summarises
      ↓
wiki/*.md  —  structured notes with YAML frontmatter
      ↓
build.py  —  converts Markdown + math + wikilinks → JSON + HTML
      ↓
docs/notes.json  +  docs/index.html  +  docs/graph.html
      ↓
GitHub Actions  —  auto-deploys on every push to main
      ↓
https://MuhammadSaqlainAslam.github.io/my-llm-wiki
```

`build.py` handles: YAML frontmatter parsing, math protection (swap LaTeX for placeholders before Markdown rendering → restore raw LaTeX for KaTeX), Obsidian callout → HTML div conversion, `![[embed]]` → clickable link conversion, Dataview/Mermaid block stubbing, and emission of `id`, `title`, `authors`, `tags`, `year`, `tldr`, `aliases`, `links`, `citation_count`, `arxiv`, `cited_by_details`, and rendered `html` per note.

---

## How to Add a New Paper

1. Drop the PDF into `raw/`
2. `python build_wiki.py` — Claude API extracts and writes `wiki/<Title>.md`
3. `python build.py` — regenerates `docs/notes.json`
4. `git add . && git commit -m "Add <Paper>" && git push`
5. GitHub Actions deploys to GitHub Pages in ~2 minutes

---

## How to Run Locally

```bash
pip install anthropic pymupdf markdown pyyaml
export ANTHROPIC_API_KEY="sk-ant-..."
python build_wiki.py   # process new PDFs → wiki/*.md
python build.py        # regenerate docs/ from wiki/
open docs/index.html   # preview in browser
```

For the knowledge graph, serve locally to avoid CORS issues:

```bash
python -m http.server 8000
# then open http://localhost:8000/docs/graph.html
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Note generation | Claude API (`claude-sonnet-4-6`) |
| PDF extraction | PyMuPDF |
| Markdown → HTML | Python `markdown` library |
| Math rendering | KaTeX (client-side) |
| Knowledge graph | D3.js v7 (force simulation) |
| Static site | Vanilla HTML / CSS / JS — zero build step, zero dependencies |
| Hosting | GitHub Pages |
| Local knowledge base | Obsidian (Dataview plugin) |
| Version control | Git + GitHub |
| Auto-deploy | GitHub Actions |

---

## Author

**Muhammad Saqlain Aslam**
GitHub: https://github.com/MuhammadSaqlainAslam
Repository: https://github.com/MuhammadSaqlainAslam/my-llm-wiki
Web Demo: https://MuhammadSaqlainAslam.github.io/my-llm-wiki
