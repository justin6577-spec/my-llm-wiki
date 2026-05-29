---
title: "State Space Duality (Mamba-2) Part III – The Algorithm"
authors:
  - Tri Dao
  - Albert Gu
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - algorithm
  - hardware acceleration
  - tensor cores
  - chunkwise recurrent
  - deep learning
  - language model
tldr: >
  Deep dive into the SSD algorithm used in Mamba-2: a block-matrix decomposition of the
  SSM semiseparable matrix that splits computation into intra-chunk (matmul, tensor-core
  friendly) and inter-chunk (short scan) phases. Achieves up to 6× speedup over Mamba-1
  during training while supporting larger state sizes (N=64/128 vs. N=16).
wikilinks:
  - "[[Mamba]]"
  - "[[Chunkwise recurrent]]"
  - "[[Hardware-Aware Scan]]"
  - "[[FlashAttention-2]]"
  - "[[Sequence parallelism]]"
  - "[[HBM]]"
  - "[[Memory hierarchy]]"
  - "[[Systolic array]]"
  - "[[Thread block]]"
  - "[[Diagonal recurrence]]"
  - "[[Exponential decay]]"
---

# State Space Duality (Mamba-2) Part III – The Algorithm

**Source:** [https://tridao.me/blog/2024/mamba2-part3-algorithm/](https://tridao.me/blog/2024/mamba2-part3-algorithm/)  
**Authors:** Tri Dao (Princeton), Albert Gu (CMU)  
**Published:** May 31, 2024  
**Paper:** [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)  

---

## Motivation: Tensor Cores vs. Scans

Mamba-1 used a parallel associative scan ([[Hardware-Aware Scan]]) for training, which avoids $O(T^2)$ complexity but **cannot use tensor cores** (specialized matrix-multiply units). On modern GPUs:

| GPU | BF16 matmul TFLOPS | FP32 scalar TFLOPS | Ratio |
|---|---|---|---|
| A100 | 312 | 19 | 16× |
| H100 | 989 | 67 | ~15× |

Mamba-1 was limited to state size $N=16$ because larger states made the scan prohibitively slow. **The SSD algorithm fixes this** by recasting the SSM computation primarily as matrix multiplications.

---

## The SSD Algorithm: Block Matrix Decomposition

The SSD layer's sequence-mixing operation can be written as multiplication by an **$N$-semiseparable matrix** $M$ of shape $(T, T)$. The SSD algorithm decomposes this matrix into blocks of size $Q \times Q$:

### Four Steps

```
Step 1 (Intra-chunk, Orange):  Diagonal blocks
   Each diagonal block is a smaller semiseparable matrix.
   Compute using the quadratic (attention-like) form → pure matmul.

Step 2 (Green):  Chunk final states
   Compute the final SSM state at end of each chunk.
   Only T/Q distinct blocks → batched matmul.

Step 3 (Yellow):  State passing (short scan)
   The chunk final states form a short 1-semiseparable recurrence.
   Run a parallel or sequential scan on T/Q states.
   ← Only this step requires a scan; runs on a much shorter sequence.

Step 4 (Blue):  Contribution from initial states
   For each chunk, add the contribution from the true initial state
   computed in Step 3 → batched matmul.
```

### Chunking Interpretation

Equivalently: split the input sequence into **chunks of size Q**:
1. **Intra-chunk outputs** — local output assuming zero initial state (matmul)
2. **Chunk states** — final state of each chunk assuming zero initial (matmul)
3. **Pass states** — recurrence over all chunk final states (short scan)
4. **Output states** — contribution from true initial state per chunk (matmul)

Steps 1, 2, 4 use matmuls and run fully in parallel. Step 3 is a scan on a sequence of length $T/Q$ — fast in practice.

This is related to [[Chunkwise recurrent]] processing patterns.

---

## Advantages Over Mamba-1

| Property | Mamba-1 | Mamba-2 (SSD) |
|---|---|---|
| State size N | 16 (limited by scan) | 64–128 |
| Tensor core usage | None | ~90% of FLOPs |
| Training throughput | Baseline | **~2–6× faster** |
| Sequence parallelism | No | Yes |
| Tensor parallelism | No | Yes |

---

## Connection to Prior Work

- **RetNet** "chunkwise" mode: a special case of SSD where the decay mask $L$ is constant  
- **Gated Linear Attention (GLA)**: another chunkwise recurrence, but derived differently  
- SSD's derivation from block matrix decomposition makes parallelization more explicit  

---

## Implementation Details

### HBM Efficiency
Like [[FlashAttention-2]], the SSD algorithm is designed to minimize reads/writes to [[HBM]]:
- Intra-chunk computation fits in [[Thread block|SRAM (shared memory)]]
- Cross-chunk state passing is small (size $N$ per chunk boundary)
- Leverages the [[Memory hierarchy]] of modern GPUs

### Numerical Stability
- The [[Exponential decay]] scalar $a_t$ can underflow for long sequences → log-space computation  
- Discretization of continuous-time SSM parameters follows standard ZOH (zero-order hold)

---

## Code

A minimal Python implementation of SSD (Listing 1 from the paper):

```python
# From ssd_minimal.py in state-spaces/mamba
def segsum(x):
    """Stable segment sum for computing cumulative decays."""
    ...

def ssd_minimal_discrete(X, A, B, C, block_len):
    # X: (batch, seqlen, n_heads, d_head)
    # A: (batch, seqlen, n_heads)       — scalar decays
    # B: (batch, seqlen, n_heads, d_state)
    # C: (batch, seqlen, n_heads, d_state)
    # block_len: chunk size Q
    ...
    # Steps 1-4: chunked matmuls + short scan
```

---

## Related Wiki Notes

- [[Mamba]] — the predecessor model  
- [[Chunkwise recurrent]] — the key computational pattern  
- [[Hardware-Aware Scan]] — Mamba-1's approach, replaced here  
- [[FlashAttention-2]] — systems design inspiration  
- [[HBM]] / [[Memory hierarchy]] — hardware considerations  
- [[Systolic array]] / [[Thread block]] — GPU primitives leveraged  
- [[Sequence parallelism]] — now applicable to SSMs via SSD  
- [[Diagonal recurrence]] — the SSM recurrence form  
- [[Exponential decay]] — role of scalar $a_t$ as decay  
