---
title: "LLM Benchmarks"
tags: [benchmarks, evaluation, llm-comparison, deepseek, claude, gpt, gemini]
year: "2025"
tldr: "Benchmark comparison across DS-V4-Pro, DS-V4-Flash, K2.6, GLM-5.1, Opus-4.6, GPT-5.4, Gemini-3.1-Pro on knowledge, long context, and agentic tasks"
---

# LLM Benchmarks

> A side-by-side comparison of seven frontier models on 22 benchmarks spanning knowledge, long context, and agentic capability. Scores collected at each model's highest-compute serving tier.

---

## Models

| Short name | Full name | Provider |
|---|---|---|
| DS-V4-Pro | DeepSeek-V4-Pro Max | DeepSeek-AI |
| DS-V4-Flash | DeepSeek-V4-Flash Max | DeepSeek-AI |
| K2.6 | K2.6 Thinking | Unknown / Kimi |
| GLM-5.1 | GLM-5.1 Thinking | Zhipu AI |
| Opus-4.6 | Claude Opus 4.6 Max | Anthropic |
| GPT-5.4 | GPT-5.4 xHigh | OpenAI |
| Gem-3.1-Pro | Gemini-3.1-Pro High | Google DeepMind |

---

## Benchmark Scores

### Knowledge & Reasoning

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| MMLU-Pro (EM) | 87.5 | 86.2 | 87.1 | 86.0 | 89.1 | 87.5 | **91.0** |
| SimpleQA-Verified (Pass@1) | 57.9 | 34.1 | 36.9 | 38.1 | 46.2 | 45.3 | **75.6** |
| Chinese-SimpleQA (Pass@1) | 84.4 | 78.9 | 75.9 | 75.0 | 76.2 | 76.8 | **85.9** |
| GPQA Diamond (Pass@1) | 90.1 | 88.1 | 90.5 | 86.2 | 91.3 | 93.0 | **94.3** |
| HLE (Pass@1) | 37.7 | 34.8 | 36.4 | 34.7 | 40.0 | 39.8 | **44.4** |
| LiveCodeBench (Pass@1) | **93.5** | 91.6 | 89.6 | — | 88.8 | — | 91.7 |
| Codeforces (Rating) | **3206** | 3052 | — | — | — | 3168 | 3052 |
| HMMT 2026 Feb (Pass@1) | 95.2 | 94.8 | 92.7 | 89.4 | 96.2 | **97.7** | 94.7 |
| IMOAnswerBench (Pass@1) | 89.8 | 88.4 | 86.0 | 83.8 | 75.3 | **91.4** | 81.0 |
| Apex (Pass@1) | 38.3 | 33.0 | 24.0 | 11.5 | 34.5 | 54.1 | **60.9** |
| Apex Shortlist (Pass@1) | **90.2** | 85.7 | 75.5 | 72.4 | 85.9 | 78.1 | 89.1 |

### Long Context

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| MRCR 1M (MMR) | 83.5 | 78.7 | — | — | **92.9** | — | 76.3 |
| CorpusQA 1M (ACC) | 62.0 | 60.5 | — | — | **71.7** | — | 53.8 |

### Agentic

| Benchmark | DS-V4-Pro | DS-V4-Flash | K2.6 | GLM-5.1 | Opus-4.6 | GPT-5.4 | Gem-3.1-Pro |
|---|---|---|---|---|---|---|---|
| Terminal Bench 2.0 (Acc) | 67.9 | 56.9 | 66.7 | 63.5 | 65.4 | **75.1** | 68.5 |
| SWE Verified (Resolved) | 80.6 | 79.0 | 80.2 | — | **80.8** | — | 80.6 |
| SWE Pro (Resolved) | 55.4 | 52.6 | **58.6** | 58.4 | 57.3 | 57.7 | 54.2 |
| SWE Multilingual (Resolved) | 76.2 | 73.3 | 76.7 | 73.3 | **77.5** | — | — |
| BrowseComp (Pass@1) | 83.4 | 73.2 | 83.2 | 79.3 | 83.7 | 82.7 | **85.9** |
| HLE w/tools (Pass@1) | 48.2 | 45.1 | **54.0** | 50.4 | 53.1 | 52.0 | 51.6 |
| GDPval-AA (Elo) | 1554 | 1395 | 1482 | 1535 | 1619 | **1674** | 1314 |
| MCPAtlas Public (Pass@1) | 73.6 | 69.0 | 66.6 | 71.6 | **73.8** | 67.2 | 69.2 |
| Toolathlon (Pass@1) | 51.8 | 47.8 | 50.0 | 40.7 | 47.2 | **54.6** | 48.8 |

*Bold = best score in row. — = not reported.*

---

## TL;DR

No single model wins everywhere. Gemini-3.1-Pro High dominates Knowledge & Reasoning — it leads on MMLU-Pro, SimpleQA, Chinese-SimpleQA, GPQA Diamond, HLE, and the punishing Apex benchmark by a wide margin, suggesting Google's data mixture and post-training have a structural edge on factual recall and breadth. Opus-4.6 Max is the clear long-context champion, beating every other model by 9–18 points on MRCR and CorpusQA at 1M tokens — a direct consequence of [[KV cache]] and attention architecture choices. Agentic tasks are the most fragmented category: GPT-5.4 xHigh leads on GDPval-AA Elo and Terminal Bench; Gemini-3.1-Pro edges BrowseComp; K2.6 Thinking takes SWE Pro and HLE-with-tools; Opus-4.6 wins SWE Verified, SWE Multilingual, and MCPAtlas. DS-V4-Pro is the strongest open model, especially in code (Codeforces 3206, LiveCodeBench 93.5), and competitive across nearly every benchmark despite being open-weight. The biggest surprise is GLM-5.1's near-collapse on Apex (11.5 vs Gemini's 60.9) — a 5× gap that doesn't appear elsewhere, hinting at a sharp reasoning ceiling for that model family on frontier difficulty.

---

## Key Takeaways

- **Knowledge & Reasoning leader:** Gemini-3.1-Pro High — top score on 6 of 11 benchmarks including the hardest (Apex, GPQA Diamond, HLE).
- **Long Context leader:** Opus-4.6 Max — 92.9 MRCR and 71.7 CorpusQA at 1M tokens, beating second-place by 9+ points.
- **Agentic leader:** No clear single winner; GPT-5.4 leads GDPval-AA Elo and Terminal Bench, Opus-4.6 leads SWE Verified/Multilingual/MCPAtlas, K2.6 leads SWE Pro and HLE-with-tools.
- **Biggest gap:** Apex (Pass@1) — Gemini-3.1-Pro at 60.9 vs GLM-5.1 at 11.5, a 49.4-point spread on the same benchmark.
- **Most surprising result:** DS-V4-Flash Max (the lighter/cheaper DeepSeek variant) scores within 1–3 points of DS-V4-Pro on most benchmarks, and achieves 3052 Codeforces rating — competitive with GPT-5.4's 3168. The cost-to-performance ratio for Flash looks exceptional.

---

## Model Notes

- **DS-V4-Pro Max:** Strongest open-weight model. Best-in-class on LiveCodeBench (93.5) and Codeforces (3206); competitive everywhere. Weakness: factual recall (SimpleQA 57.9 vs Gemini's 75.6) and Apex reasoning ceiling.
- **DS-V4-Flash Max:** Remarkably close to Pro across the board at presumably much lower cost. Weakness: long context not reported; BrowseComp and agentic benchmarks noticeably trail Pro.
- **K2.6 Thinking:** Surprise leader on SWE Pro (58.6) and HLE-with-tools (54.0); strong BrowseComp (83.2). Weakness: Apex collapses (24.0); long context not evaluated.
- **GLM-5.1 Thinking:** Solid mid-tier on most benchmarks. Severe weakness on Apex (11.5) — by far the lowest score of any model — suggesting hard limits on frontier reasoning difficulty. Long context not evaluated.
- **Opus-4.6 Max:** Dominant on long context; top-tier agentic (SWE Verified, SWE Multilingual, MCPAtlas). Weakness: math olympiad ceiling (IMOAnswerBench 75.3 vs GPT-5.4's 91.4); Apex not at the top.
- **GPT-5.4 xHigh:** Best on HMMT (97.7), IMOAnswerBench (91.4), Terminal Bench (75.1), GDPval-AA Elo (1674), Toolathlon (54.6). Weakness: long context not evaluated; Apex Shortlist trails DS-V4-Pro by 12 points.
- **Gemini-3.1-Pro High:** Broadest knowledge lead — top on MMLU-Pro, SimpleQA, GPQA Diamond, HLE, Apex (60.9). Weakness: GDPval-AA Elo (1314, lowest of all); MRCR long context (76.3 vs Opus's 92.9); SWE Pro (54.2, weakest evaluated model).

---

## Related Concepts

[[DeepSeek_V4]] · [[KV cache]] · [[Speculative Decoding]] · [[GQA]] · [[RLVR]] · [[LLM evaluation]]
