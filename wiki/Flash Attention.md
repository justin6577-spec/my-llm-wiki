---
title: "Flash Attention"
tags: [attention, hardware, kernel-fusion, sram, gpu]
tldr: "An IO-aware exact attention implementation that keeps the softmax computation entirely in [[SRAM]], never materializing the $T \times T$ attention matrix in [[HBM]]. 2–4× wall-clock speedup with zero accuracy change; the canonical example of a [[Hardware-aware algorithms|hardware-aware algorithm]]."
---

# Flash Attention

Standard attention computes $S = QK^\top$ in HBM, then $P = \text{softmax}(S)$, then $O = PV$. The intermediate $T \times T$ matrix is huge — for $T = 8192$ it's 64 MB per head per layer — and it's read and written multiple times. **FlashAttention** (Dao 2022) blocks the computation into SRAM-sized tiles, computes a numerically stable online softmax across blocks, and never writes the full $S$ or $P$ to HBM. The math is identical (no approximation), but the wall-clock speedup is 2–4× on Ampere and 4–8× on Hopper, simply because HBM round-trips are eliminated. FlashAttention v2 added better warp-level scheduling; v3 specialized for Hopper's TMA + warp-specialization. The template — blocked, online softmax, IO-aware — is now the standard for any serious attention kernel.

## Where it appears

- [[Mamba]]
- [[Hardware Acceleration for Neural Networks]]
- [[KV Cache Optimization]]

---

*Related: [[Hardware-Aware Scan]] · [[Kernel fusion]] · [[SRAM]] · [[HBM]] · [[Transformer]]*
