---
title: "Mamba-3: Improved Sequence Modeling using State Space Principles"
authors:
  - Aakash Lahoti
  - Kevin Y. Li
  - Berlin Chen
  - Caitlin Wang
  - Aviv Bick
  - J. Zico Kolter
  - Tri Dao
  - Albert Gu
year: 2025
tags:
  - state space model
  - SSM
  - mamba
  - MIMO
  - language model
  - deep learning
  - inference optimization
  - sequence modeling
tldr: >
  Mamba-3 extends the Mamba-2 SSD framework with a MIMO (multiple-input multiple-output)
  SSM formulation and an inference-first design philosophy. It introduces structured
  multi-rank state transitions for improved long-range recall while maintaining linear-time
  recurrence for inference. Achieves better quality/efficiency trade-offs than Mamba-2
  at equivalent parameter counts.
wikilinks:
  - "[[Mamba]]"
  - "[[Chunkwise recurrent]]"
  - "[[Diagonal recurrence]]"
  - "[[Inference optimization]]"
  - "[[Hardware-Aware Scan]]"
  - "[[Sequence parallelism]]"
  - "[[Memory hierarchy]]"
  - "[[Covariance update rule]]"
---

# Mamba-3: Improved Sequence Modeling using State Space Principles

**Paper:** [arXiv 2603.15569](https://arxiv.org/abs/2603.15569)  
**Authors:** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu  
**Year:** 2025  
**PDF:** `Mamba3.pdf`  
**Code:** [state-spaces/mamba](https://github.com/state-spaces/mamba) (`mamba_ssm/modules/mamba3.py`)

---

## TL;DR

Mamba-3 is the third generation in the Mamba family, built on the [[Chunkwise recurrent|SSD framework]] of Mamba-2 but with two key innovations:

1. **MIMO SSM:** A multiple-input multiple-output formulation where multiple input channels jointly update a shared state, enabling richer state representations.  
2. **Inference-first design:** Architecture choices optimized for low-latency autoregressive inference while retaining training efficiency.

---

## Background: Evolution of the Mamba Family

| Generation | Key Innovation | State Size | Algorithm |
|---|---|---|---|
| [[Mamba]]-1 | Selective SSM (input-dependent A,B,C) | N=16 | Hardware-aware scan |
| Mamba-2 (SSD) | Scalar A, multi-head, tensor cores | N=64–128 | Chunkwise matmul |
| **Mamba-3** | MIMO SSM, inference-first | N=128+ | MIMO chunkwise |

---

## MIMO SSM Formulation

Standard Mamba-2 (SISO — single input, single output per head):

$$h_t = a_t \cdot h_{t-1} + b_t x_t, \qquad y_t = c_t^\top h_t$$

Mamba-3 MIMO (shared state across multiple channels):

$$h_t = A_t h_{t-1} + B_t X_t, \qquad Y_t = C_t^\top h_t$$

where $X_t \in \mathbb{R}^{r \times d}$ and $B_t \in \mathbb{R}^{N \times r}$ — the state is updated from **rank-$r$ inputs** simultaneously. This relates to the [[Covariance update rule]] in the sense of low-rank structured updates to a shared state.

**MIMO rank** (`mimo_rank`) controls the number of input channels jointly processed — analogous to the rank in low-rank matrix approximations.

---

## Architecture

```python
from mamba_ssm import Mamba3

model = Mamba3(
    d_model=768,        # model dimension
    d_state=128,        # SSM state size
    headdim=64,         # per-head dimension
    is_mimo=True,       # enable MIMO mode
    mimo_rank=4,        # MIMO rank r
    chunk_size=16,      # SSD chunk size (64/mimo_rank for bf16)
    is_outproj_norm=False,  # optional post-SSM normalization
    dtype=torch.bfloat16,
)
```

### Key Differences from Mamba-2

| Feature | Mamba-2 | Mamba-3 |
|---|---|---|
| Input structure | SISO per head | MIMO (rank-r joint update) |
| State transitions | Scalar × identity $A$ | Structured low-rank $A$ |
| Chunk size | 64 (bf16) | 64/mimo_rank (bf16) |
| Design focus | Training efficiency | Inference + training balance |
| Recall ability | Good | Improved |

---

## Inference-First Design

The "inference-first" philosophy means:
- **Fixed-size recurrent state** at inference time: $O(N \cdot d_\text{state})$ memory regardless of sequence length  
- No [[KV cache]] needed — unlike Transformer attention  
- Autoregressive generation is $O(1)$ per step vs. $O(T)$ for attention  

This makes Mamba-3 attractive for [[Inference optimization]] in production deployment.

---

## Training Algorithm

Mamba-3 extends the [[Chunkwise recurrent|SSD chunkwise algorithm]]:
- MIMO adds a low-rank projection at the state update step  
- Still decomposes into intra-chunk matmuls + inter-chunk short scan  
- Leverages tensor cores for the dominant FLOP budget  
- [[Sequence parallelism]] applies at the chunk boundary

The chunk size formula `chunk_size = 64 / mimo_rank` (bf16) ensures the inner matmuls remain tile-efficient on GPU [[Memory hierarchy]].

---

## Pretrained Models

Available on HuggingFace under `state-spaces` (see repo README). Mamba-3 models show improved perplexity and downstream task performance vs. Mamba-2 at equivalent parameter counts, particularly on:
- Long-range recall benchmarks  
- In-context learning tasks  
- Language modeling (Pile, SlimPajama)

---

## Relationship to Other Work

- **[[Mamba]]:** Direct ancestor; Mamba-3 retains backward compatibility with the `mamba_ssm` package  
- **[[Diagonal recurrence]]:** Mamba-3 relaxes the scalar-$A$ constraint of Mamba-2 while preserving tractability  
- **[[Chunkwise recurrent]]:** Core training algorithm shared with Mamba-2  
- **S4/S4D:** Earlier structured SSMs that motivated the state-space approach  

---

## Related Wiki Notes

- [[Mamba]] — the foundational model  
- [[Chunkwise recurrent]] — shared training algorithm  
- [[Diagonal recurrence]] — extended by MIMO formulation  
- [[Inference optimization]] — key design goal  
- [[Hardware-Aware Scan]] — Mamba-1's approach, evolved through SSD to Mamba-3  
- [[Sequence parallelism]] — applicable at chunk boundaries  
- [[Memory hierarchy]] — drives chunk size selection  
- [[Covariance update rule]] — conceptual parallel for joint state updates  
