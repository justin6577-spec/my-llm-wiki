---
title: "State Space Duality (Mamba-2) Part I – The Model"
authors:
  - Tri Dao
  - Albert Gu
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - attention
  - linear attention
  - sequence modeling
  - deep learning
  - language model
tldr: >
  Tri Dao and Albert Gu's blog post introducing Mamba-2 and the SSD (Structured State Space
  Duality) layer. Explains how a minor structural restriction on Mamba-1's diagonal-A SSM
  yields a scalar-times-identity form that is simultaneously a recurrent SSM and a form of
  linear attention, enabling tensor-core-friendly matrix-multiplication-based training.
wikilinks:
  - "[[Mamba]]"
  - "[[Diagonal recurrence]]"
  - "[[Chunkwise recurrent]]"
  - "[[Exponential decay]]"
  - "[[Channel-mixing]]"
  - "[[Hardware-Aware Scan]]"
  - "[[FlashAttention-2]]"
  - "[[Sequence parallelism]]"
  - "[[RWKV]]"
---

# State Space Duality (Mamba-2) Part I – The Model

**Source:** [https://tridao.me/blog/2024/mamba2-part1-model/](https://tridao.me/blog/2024/mamba2-part1-model/)  
**Authors:** Tri Dao (Princeton), Albert Gu (CMU)  
**Published:** May 31, 2024  
**Paper:** [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)  

---

## Motivation

Despite Mamba-1's strong results, two key problems remained:

1. **Understanding:** SSMs felt disconnected from the dominant paradigm of **attention**. How are they related conceptually?  
2. **Efficiency:** Mamba-1's [[Hardware-Aware Scan]] did not use **tensor cores** (matrix-multiplication units). On an H100 GPU, BF16 matmul delivers 989 TFLOPS vs. 67 TFLOPS for FP32 scalar ops — a ~15× gap.

---

## The SSD Layer

### Core SSM equations (same as Mamba-1)

$$h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t$$

- $x_t, y_t \in \mathbb{R}$ (scalars)  
- $h_t \in \mathbb{R}^N$ (hidden state, size $N$)  
- $(A, B, C)$ are **input-dependent** (selective) parameters

### The SSD Restriction

Mamba-1 uses **diagonal** $A_t \in \mathbb{R}^{N \times N}$. The SSD layer restricts this further:

> **$A_t$ is a scalar × identity**: all diagonal elements are equal.

This means $A_t$ has shape $(T,)$ — a single scalar per timestep — giving a [[Diagonal recurrence]] of the form $h_t = a_t \cdot h_{t-1} + B_t x_t$.

This tiny change has outsized consequences: the SSD layer can be expressed **both** as:
- A **linear recurrence** (SSM mode) — efficient for [[Inference optimization|inference]]
- A **quadratic attention-like form** (attention mode) — efficient for training via matrix multiplications

This is the core of **State Space Duality (SSD)**.

---

## Multi-Head Structure

To handle multi-dimensional inputs, Mamba-2 introduces a **multi-head** design analogous to multi-head attention:

- Inputs $X, Y$ have shape $(T, H \cdot P)$ where $H$ = heads, $P$ = head dim  
- Each head independently runs an SSD layer with its own $(A, B, C)$  
- Larger state sizes (e.g., $N=64$ or $N=128$) are now tractable — vs. $N=16$ in Mamba-1  

This is related to [[Multi-Query Attention]] in its grouped/shared-parameter philosophy.

---

## Mamba-2 Architecture

The overall Mamba-2 block differs from Mamba-1 in:

| Feature | Mamba-1 | Mamba-2 |
|---|---|---|
| A structure | Diagonal (N×N) | Scalar × Identity |
| State size N | Typically 16 | Typically 64–128 |
| Training algorithm | Parallel scan (no tensor cores) | SSD (tensor-core matmuls) |
| Head structure | No explicit heads | Multi-head |
| Normalization | — | Group norm per head |

The block still uses a [[Channel-mixing]] gating mechanism and a local [[Diagonal recurrence]] for the conv layer.

---

## SSD vs. Attention

The SSD layer's quadratic form looks like:

$$Y = \text{masked\_softmax-free attention}(Q=C, K=B, V=X, \text{mask}=L)$$

where $L$ is a lower-triangular mask derived from the [[Exponential decay]] $a_t$. This is a form of **linear attention** with a structured decay mask, connecting to models like RetNet and GLA.

---

## Key Insight

> The SSD model sits at the intersection of SSMs and linear attention — enabling both efficient recurrent inference and GPU-optimized training via tensor cores.

---

## Related Wiki Notes

- [[Mamba]] — predecessor selective SSM  
- [[Chunkwise recurrent]] — the SSD algorithm processes sequences in chunks  
- [[Hardware-Aware Scan]] — Mamba-1's approach replaced by SSD matmuls  
- [[Diagonal recurrence]] — the recurrence type used in SSD  
- [[Exponential decay]] — the scalar $a_t$ acts as a positional decay  
- [[FlashAttention-2]] — systems inspiration for SSD  
- [[Sequence parallelism]] — SSD unlocks transformer-style parallelism for SSMs  
- [[RWKV]] — another recurrent model with attention-like duality  
