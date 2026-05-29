---
title: "Jamba: A Hybrid Transformer-Mamba Language Model"
authors:
  - Opher Lieber
  - Barak Lenz
  - Hofit Bata
  - Gal Cohen
  - Jhonathan Osin
  - Itay Dalmedigos
  - Erez Safahi
  - Shaked Meirom
  - Yonatan Berant
  - Shai Shalev-Shwartz
  - Omri Abend
  - Raz Alon
  - Tomer Asida
  - Amir Bergman
  - Roman Glozman
  - Michael Gokhman
  - Avashalom Manevich
  - Nir Ratner
  - Noam Rozen
  - Erez Shwartz
  - Mor Zusman
  - Yoav Shoham
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - transformer
  - mixture of experts
  - MoE
  - hybrid model
  - language model
  - deep learning
  - inference optimization
tldr: >
  Jamba is a production-scale hybrid architecture interleaving Transformer attention layers
  with Mamba SSM layers inside a Mixture-of-Experts (MoE) framework. At 52B total / 12B
  active parameters, it fits on a single 80GB GPU while achieving throughput 3× higher than
  Mixtral-8x7B, with strong benchmark performance across long-context tasks up to 256K tokens.
wikilinks:
  - "[[Mamba]]"
  - "[[Mixtral]]"
  - "[[Mixture of experts]]"
  - "[[Multi-Query Attention]]"
  - "[[KV cache]]"
  - "[[Inference optimization]]"
  - "[[Sequence parallelism]]"
  - "[[Memory hierarchy]]"
  - "[[HBM]]"
  - "[[RoPE]]"
---

# Jamba: A Hybrid Transformer-Mamba Language Model

**Paper:** [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)  
**Authors:** AI21 Labs team (Lieber et al.)  
**Year:** 2024  
**PDF:** `Jamba.pdf`

---

## TL;DR

Jamba is the **first production-scale hybrid SSM-Transformer model**, combining:
- **Mamba** selective SSM layers (efficient long-range memory, no [[KV cache]])  
- **Transformer attention** layers (precise short-range token interaction)  
- **Mixture of Experts (MoE)** for parameter efficiency  

Result: 52B total parameters, 12B active, fits on **one 80GB A100**, with **3× throughput** vs. Mixtral-8x7B and competitive quality.

---

## Architecture

### Block Structure

Jamba uses a repeating **Jamba block** consisting of:

```
[ Mamba layer ] × ratio
[ Attention layer ] × 1
[ MoE FFN ] per block
```

The **attention-to-Mamba ratio** is a key hyperparameter (default: 1 attention per 8 Mamba layers in the 7B variant).

### Key Design Choices

| Component | Choice | Rationale |
|---|---|---|
| SSM | [[Mamba]] (selective SSM) | Long-range context without KV cache |
| Attention | Multi-head + [[RoPE]] | Precise local attention |
| FFN | [[Mixture of experts\|MoE]] | Scale parameters without proportional compute |
| Position | [[RoPE]] for attention layers only | SSM layers are position-free |
| Normalization | RMSNorm | Standard for modern LLMs |

---

## Memory Efficiency

The core motivation is that **pure Transformer models** require a [[KV cache]] that grows linearly with sequence length — prohibitive for 256K context. Jamba's Mamba layers maintain a **fixed-size recurrent state** regardless of sequence length.

For a sequence of length $T$:
- Transformer KV cache: $O(T)$ per layer  
- Mamba SSM state: $O(N)$ constant per layer  

By having most layers be Mamba (e.g., 7:1 ratio), Jamba dramatically reduces [[HBM]] usage for long contexts.

---

## Performance

### Throughput (vs. Mixtral-8x7B)
- **3× higher throughput** at long contexts  
- Fits in 80GB GPU (Mixtral requires ≥160GB for equivalent context)

### Quality (7B variant, trained on 1T tokens)
Competitive with Llama-2-7B and Mistral-7B on:
- Hellaswag, ARC, PIQA, WinoGrande  
- Long-context tasks (SCROLLS, RULER up to 256K)

### Long Context
Jamba handles **256K token contexts** — a key advantage over Transformers at this scale.

---

## MoE Integration

The [[Mixture of experts|MoE]] layers follow the [[Mixtral]] design:
- Top-$k$ expert routing (typically $k=2$)  
- Each Jamba block has a shared expert + routed experts  
- Allows 52B total parameters with only 12B active

---

## Relationship to Other Models

| Model | Attention | SSM | MoE | Context |
|---|---|---|---|---|
| Llama-2 | ✓ | ✗ | ✗ | 4K |
| Mistral | ✓ | ✗ | ✗ | 32K |
| [[Mixtral]] | ✓ | ✗ | ✓ | 32K |
| [[Mamba]]-2.8B | ✗ | ✓ | ✗ | ∞ (recurrent) |
| **Jamba** | ✓ (1 in 8) | ✓ | ✓ | 256K |

---

## Key Insights

1. **Complementary strengths:** Attention excels at precise token recall; Mamba excels at compressing long-range context efficiently.  
2. **Ratio matters:** Too much attention → memory pressure; too much Mamba → reduced recall on associative tasks.  
3. **MoE amplifies:** Experts allow specialization without full-parameter compute cost.  
4. **Systems-friendly:** Single-GPU deployment at 52B scale is a practical breakthrough.

---

## Related Wiki Notes

- [[Mamba]] — the SSM backbone used in Jamba  
- [[Mixtral]] — MoE design inspiration  
- [[KV cache]] — the bottleneck Mamba layers eliminate  
- [[Inference optimization]] — Jamba's throughput advantages  
- [[Multi-Query Attention]] — related attention efficiency technique  
- [[RoPE]] — positional encoding used in attention layers  
- [[Memory hierarchy]] / [[HBM]] — GPU memory constraints motivating the design  
- [[Sequence parallelism]] — applicable to Jamba's attention layers  
