---
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
authors: "Dao & Gu"
year: 2024
tags: [ssm, mamba, attention, ssd, efficiency, mamba-2, linear-attention]
tldr: "Attention and selective SSMs are two faces of the same object — a structured semiseparable matrix. The Structured State Space Duality (SSD) framework makes the equivalence precise and unlocks Mamba-2: a refined selective SSM whose core layer runs 2–8× faster than Mamba while matching or beating Transformers on language modeling."
aliases: [Mamba-2, SSD, Structured State Space Duality]
theme: efficiency
citation_count: 1526
arxiv: "2405.21060"
cited_by_top: ["Mamba-3", "Falcon Mamba", "RWKV-7", "Samba", "Hymba", "DenseMamba", "PlainMamba", "MambaND", "Mamba Drafters", "LocalMamba"]
---

# Transformers are SSMs

> Tri Dao & Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", 2024 (arXiv:2405.21060)

## TL;DR

For a year, the field treated [[Mamba]] and [[Transformer]] as rival architectures. Dao & Gu show they are the same object — a special class of structured matrix — viewed from two angles. Once you see the equivalence, you can borrow the best of each: Mamba's linear-time recurrence at inference, the Transformer's matmul-heavy parallelism at training. The new layer that drops out of the math is **[[Mamba-2]]**, whose core is 2–8× faster than the original [[Selective State Space Model|selective SSM]] and competitive with [[Transformer]] at language modeling up to 2.7B parameters trained on 300B tokens.

---

## The Core Idea — One Object, Two Views

Both an attention layer and a selective SSM compute the same kind of sequence-to-sequence map:

$$
y = M \cdot x
$$

where $M \in \mathbb{R}^{T \times T}$ is the **mixing matrix** that says how much of input position $j$ shows up in output position $i$. The two architectures differ only in how they parameterize $M$:

- **Attention** writes $M$ explicitly as $\text{softmax}(QK^\top) \odot \text{causal mask}$. You materialize it.
- **Selective SSM** writes $M$ implicitly as a **[[Semiseparable matrix]]** — every off-diagonal block has rank $\le N$ (the SSM state dimension). You never materialize it; you walk the recurrence.

The key theorem of the paper: *the class of matrices computable by an SSM with state size $N$ is exactly the class of $N$-semiseparable causal matrices.* Attention is the "no structure" case. Mamba is the "low rank everywhere off-diagonal" case. Linear attention is the rank-1 case.

This is the **Structured State Space Duality (SSD)**. It tells you that any algorithmic trick that works for one ought to have a counterpart for the other.

---

## Key Concepts

- **[[Semiseparable matrix]]** — a matrix whose every off-diagonal sub-block has rank at most $N$. SSMs of state size $N$ produce exactly these matrices.
- **[[Structured State Space Duality]] (SSD)** — the framework saying selective SSMs ↔ structured masked attention via semiseparable decompositions.
- **[[Mamba-2]]** — the new layer that drops out of SSD: a selective SSM that is also a structured form of attention, parallelizable as one big matmul.
- **[[Multi-head SSM]]** — heads in attention have a direct counterpart in SSMs once you see them as matrix mixers.
- **[[Tensor parallelism]]** and **[[Sequence parallelism]]** — model-parallelism techniques designed for [[Transformer]]; SSD shows how to apply them to SSMs.

---

## Architecture / Method

The original [[Mamba]] block computes the SSM recurrence sequentially with the [[Hardware-Aware Scan]]. SSD replaces the inner SSM with a different decomposition:

1. Treat the $T \times T$ mixing matrix $M$ as a block matrix with block size $Q \approx 64$.
2. **Diagonal blocks** ($Q \times Q$) — compute densely as a small attention-like matmul.
3. **Off-diagonal blocks** — these are low-rank by the semiseparable property; compute via $O(N)$-rank factorizations (a small matmul, then a small recurrent reduction along block boundaries).

Concretely:
- The diagonal contribution becomes a single big GEMM (the same primitive that makes [[Transformer]] training so GPU-friendly).
- The off-diagonal contribution becomes a chunked scan with chunk size $Q$ — much shorter scans than the per-token recurrence Mamba uses.

Result: most of the work is now matmul, which Tensor Cores eat for breakfast. The pure scan part is short and cheap.

Mamba-2 also tweaks the parameterization: $\mathbf{A}$ is restricted to a scalar times identity ($\mathbf{A}_t = a_t \mathbf{I}$), which is what enables the SSD decomposition. The state size grows from $N=16$ in Mamba to $N=64$ or $N=128$ in Mamba-2 — possible because the new algorithm makes large $N$ cheap.

---

## Key Results

| Model            | Params | Train tokens | Pile PPL / Avg zero-shot     |
|------------------|--------|--------------|------------------------------|
| Pythia           | 2.8 B  | 300 B        | baseline                     |
| Pythia           | 6.9 B  | 300 B        | better but 2.5× the params   |
| Mamba            | 2.8 B  | 300 B        | beats Pythia-2.8B            |
| **Mamba-2**      | **2.7 B** | **300 B** | **beats Pythia-6.9B**        |

- **Speed:** the SSD layer is **2–8× faster** than Mamba's selective scan at the same model size, depending on sequence length.
- **MQAR (Multi-Query Associative Recall):** the hard recall task where pure SSMs traditionally lose to attention — Mamba-2 closes most of the gap by virtue of its larger usable state dimension.
- **Tensor parallelism** designed-in: Mamba-2 needs **half** the synchronization points per block compared to Mamba, making it as TP-friendly as a Transformer.
- **No padding for variable-length sequences** — the recurrent state lets the model handle ragged batches without the masking gymnastics Transformers need.

---

## Comparison to Prior Work

- vs. **[[Transformer]]** — same expressive class once you allow arbitrary semiseparable structure; Mamba-2 trades the $T \times T$ unstructured mask for an $N$-rank structured one. Same matmul-heavy training, but $O(T)$ instead of $O(T^2)$ at long context.
- vs. **[[Mamba]]** — same selectivity story, same hardware-awareness, but a fundamentally different computation: chunked matmul + short scan instead of a long fused scan. Larger state dimension is now affordable.
- vs. **[[Linear attention]]** — linear attention is the special case where every off-diagonal block has rank 1. Mamba-2 gives you rank up to $N$, which is why it actually works at scale.
- vs. **RetNet / RWKV** — these are also linear recurrences, but with fixed (input-independent) dynamics. SSD shows they sit inside a strictly weaker class than selective SSMs.

---

## Limitations

- **Scalar $\mathbf{A}$ is a real restriction.** Original Mamba allowed a diagonal $\mathbf{A}_t$; Mamba-2 needs the scalar form for SSD to give clean matmuls. Some long-range tasks where Mamba's diagonal $\mathbf{A}$ helps may regress slightly.
- **Recall still trails attention at the limit.** Bigger $N$ helps, but for tasks that need *exact* lookup over millions of tokens, attention with a [[KV Cache]] still wins. Hybrid architectures like [[Nemotron-3]] keep a few attention layers for exactly this reason.
- **The SSD framework is dense at first read.** Most readers will find the diagonal-vs-off-diagonal decomposition intuitive only after they have stepped through the small example in the paper.

---

## Why It Matters

- **It ends the "SSM vs. Transformer" framing.** Both are mixing matrices. Choose the structure you want; you are not picking sides anymore.
- **It makes SSMs first-class on modern accelerators.** GPUs and TPUs are matmul engines. Mamba's biggest practical wart was the mostly-scan workload; SSD turns most of it back into matmul.
- **It gives a roadmap for hybrid models.** [[Nemotron-3]] and other hybrid stacks can now port [[Tensor parallelism]], [[Sequence parallelism]], FlashAttention-style IO awareness, and other Transformer-era systems work directly to their SSM layers.

---

## Related Notes

[[Mamba]] · [[Transformer]] · [[State Space Model]] · [[Hardware-Aware Scan]] · [[Nemotron-3]] · [[KV Cache]] · [[Mamba-2]] · [[Selective State Space Model]]
