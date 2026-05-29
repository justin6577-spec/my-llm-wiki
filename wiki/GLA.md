```markdown
---
title: "GLA (Gated Linear Attention)"
tags: [glossary, attention, linear-attention, ssm, efficient-transformers]
tldr: "A recurrent sequence model that adds data-dependent gating to linear attention, enabling selective memory like SSMs while retaining the parallelizable matrix form of attention."
---

## TL;DR
GLA fuses the best of linear attention and gated SSMs: it uses a 2D hidden state (key-value matrix) updated with a learnable, input-dependent decay gate — giving ~O(1) inference cost and competitive quality with softmax Transformers.

## Intuition
Standard linear attention maintains a hidden state `S ∈ R^{d×d}` updated as `S_t = S_{t-1} + kᵀv`. The problem: every past token contributes equally forever — no forgetting. GLA fixes this by multiplying the accumulated state by a **gating matrix** `G_t = sigmoid(W_g · xₜ)` before adding the new association: `S_t = G_t ⊙ S_{t-1} + kᵀv`. These gates are *data-dependent*, so the model learns *what* to forget at each step, much like Mamba's selective scan.

The key engineering win: despite the per-token gating breaking naive parallelism, GLA can still be trained efficiently using a **chunkwise parallel** form. You split the sequence into chunks of size ~64, run intra-chunk attention in parallel, and propagate inter-chunk hidden states recurrently. This gives roughly 2–3× throughput vs. naive full attention at sequence length 2K, while scaling to 1B+ parameters on language modeling without significant quality loss vs. Transformers.

## Why It Matters
- **Inference efficiency**: Decoding is truly O(1) per token (fixed hidden state size, e.g. d=256 key dim × 256 value dim), making it viable for edge/streaming LLM deployment.
- **Competitive perplexity**: GLA at 1.3B params matches or beats Mamba and RetNet at equivalent token budgets (e.g., ~300B tokens on The Pile), closing much of the gap with softmax attention.
- **Hardware-friendly**: The chunkwise CUDA kernel avoids materializing the full N×N attention matrix; memory is O(N·d) not O(N²), enabling 8K+ context without flash-attention tricks.

## Key Formula or Mechanism
```
# GLA recurrent update (per token):
S_t = diag(g_t) @ S_{t-1} + k_t^T @ v_t   # S ∈ R^{dk × dv}
o_t = q_t @ S_t

# g_t ∈ R^{dk} — data-dependent gate:
g_t = sigmoid(Linear(x_t))   # elementwise decay per key dimension

# Chunkwise parallel form:
for chunk c of size L:
    # intra-chunk: standard linear attention (parallel)
    # inter-chunk: pass S_c recurrently with accumulated gates
```

## Where It Appears
- **GLA paper**: *"Gated Linear Attention Transformers with Hardware-Efficient Training"* — Yang et al., 2024 (arXiv 2312.06635)
- **HGRN2**: related gated recurrent design from same lineage
- **RetNet** (Microsoft, 2023): predecessor using fixed exponential decay instead of learned gates
- **Mamba** (Gu & Dao, 2023): parallel selective SSM that inspired GLA's data-dependent gating

## Related Concepts
[[Linear Attention]]
[[Mamba / S6]]
[[RetNet]]
[[State Space Models (SSM)]]
[[Chunkwise Parallel Scan]]
```