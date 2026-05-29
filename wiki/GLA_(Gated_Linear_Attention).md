```markdown
---
title: "GLA (Gated Linear Attention)"
tags: [glossary, attention, linear-attention, ssm, efficient-transformers]
tldr: "A linear attention variant that adds a data-dependent decay gate, closing the gap with softmax attention while keeping O(1) recurrent inference cost."
---

## TL;DR
GLA replaces the fixed exponential decay of linear RNNs with a **learned, input-dependent gate** applied to the hidden state matrix, giving the model dynamic control over memory retention at each step.

## Intuition
Standard linear attention rewrites `softmax(QK^T)V` as `Q(K^TV)`, collapsing the sequence into a `d×d` hidden state. This is fast but the hidden state accumulates everything with equal weight — old irrelevant tokens pollute new predictions. GLA fixes this by multiplying the hidden state by a gate `G_t ∈ (0,1)^{d×d}` at each step, letting the model *forget* selectively. Think of it as a 2D version of the forget gate in an LSTM, but operating on a full feature matrix rather than a vector.

The gate is computed from the current input (`G_t = sigmoid(Linear(x_t))`), so it's **data-dependent** — attending to a new topic can flush stale state, while a stable context keeps decay near 1.0. In practice GLA uses a low-rank factored gate (rank ~16) to keep parameter count manageable; full `d×d` gating on d=2048 would cost 4M extra parameters per layer just for the gate projection.

## Why It Matters
- **Recurrent inference at O(1) memory**: at generation time GLA runs as a pure RNN — no KV cache growing with context length, critical for long-document or streaming LLM deployment.
- **Hardware-efficient training via chunkwise form**: sequences are split into chunks (~256 tokens); within a chunk you do a small softmax-like materialization, across chunks you do recurrent state passing — achieving ~80% of FlashAttention throughput on A100s.
- **Beats Mamba/RetNet on many benchmarks** at matched parameter count, suggesting data-dependent decay is the key missing ingredient in prior linear attention designs.

## Key Formula or Mechanism
Recurrent update at step t:

```
S_t = G_t ⊙ S_{t-1} + k_t^T v_t      # S ∈ R^{d_k × d_v}
o_t = q_t S_t

where G_t = sigmoid(W_g x_t)  ∈ (0,1)^{d_k}   # broadcast over d_v
```

`⊙` is elementwise multiply (broadcast). The chunkwise form batches C steps:  
within-chunk: `O_chunk = tril(QK^T ⊙ Γ) V`  
cross-chunk: recurrent carry of `S` with accumulated gate product `∏ G`.

## Where It Appears
- **GLA paper**: *"Gated Linear Attention Transformers with Hardware-Efficient Training"* — Yang et al., 2023 (arXiv 2312.06635)
- **HGRN2**: uses similar gated hidden-state ideas
- **Mamba2**: converges on a related formulation (state space duality), validating the design space
- **RetNet**: predecessor without data-dependent gate (fixed decay γ per head)

## Related Concepts
[[Linear Attention]]
[[Mamba (SSM)]]
[[RetNet]]
[[RWKV]]
[[Flash Attention]]
```