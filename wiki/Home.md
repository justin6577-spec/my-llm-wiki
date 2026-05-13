# LLM Wiki

AI research knowledge base — papers, concepts & intuitions. Curated by Saqlain.

---

## Foundations

| Article | One Line |
|---|---|
| [[Transformer]] | Self-attention replaces recurrence. Any token attends to any other in O(1). |
| [[Mamba]] | Selective state spaces. Linear-time sequence modeling that matches Transformer quality. |

## Efficient Sequence Modeling

| Article | One Line |
|---|---|
| [[Transformers Are SSMs]] | Attention and selective SSMs are the same structured-matrix object. SSD framework gives Mamba-2: 2–8× faster, matmul-friendly. |
| [[xLSTM]] | LSTM rehab — exponential gating + matrix memory. Competitive with Transformer at 1B+ scale, O(1) inference. |

## Scaling

| Article | One Line |
|---|---|
| [[Mixture-of-Experts]] | Decouple parameters from compute. Route tokens to specialized FFN subnetworks. |

## Hardware & Systems

| Article | One Line |
|---|---|
| [[Hardware Acceleration for Neural Networks]] | Survey of GPUs, TPUs, NPUs, FPGAs, ASICs, LPUs, in/near-memory. The bottleneck is bandwidth, not FLOPS. |

## Inference Optimization

| Article | One Line |
|---|---|
| [[KV Cache Optimization]] | Five families: eviction, compression, hybrid memory, novel attention, combination. No universal winner — pick by workload. |
| [[Speculative Decoding]] | Fast draft proposes K tokens; large model verifies all K in one pass. ~2× throughput on long generations. |

## Modern Systems

| Article | One Line |
|---|---|
| [[Nemotron-3]] | Hybrid Mamba-Transformer-MoE. 3x throughput over pure Transformer MoE at same quality. |
| [[DeepSeek_V4]] | CSA + HCA compressed attention cuts KV cache 10× at 1M tokens. MoE + Muon optimizer. SOTA open model. |

---

## Concept Map

```
Transformer (2017)
│
├── Problem: O(n²) attention cost at long sequences
│   ├──► Mamba (2024): replace attention with selective SSM
│   ├──► Transformers Are SSMs (2024): unify attention + SSM via SSD; Mamba-2 is 2–8× faster
│   ├──► xLSTM (2024): modernize LSTM with exponential gating + matrix memory
│   └──► DeepSeek-V4 (2026): CSA + HCA compress KV cache 10× at 1M tokens
│
├── Problem: Parameters tied to compute
│   └──► Mixture-of-Experts: sparse routing, constant FLOPs/token
│
├── Problem: MoE communication + latency bottlenecks
│   └──► Nemotron-3 LatentMoE: route in latent space
│
├── Problem: Inference is memory-bandwidth bound
│   ├──► Speculative Decoding: amortize verifier passes across K draft tokens (~2×)
│   └──► KV Cache Optimization (2026): 5 families — eviction, compression,
│        hybrid memory, novel attention, combination
│
└── Problem: Knowing which silicon to run on
    └──► Hardware Acceleration Survey (2025): GPU vs TPU vs NPU vs FPGA vs ASIC vs LPU vs
         in-memory; bandwidth/KV-cache I/O dominates, not FLOPS
```

```
Mamba + MoE + few attention layers
= Nemotron-3 hybrid architecture
= best throughput-to-accuracy frontier (2025)

Standard attention + DeepSeekMoE + CSA/HCA + mHC + Muon
= DeepSeek-V4 (2026)
= 1M-token context at 27% of V3 FLOPs, SOTA open model

DeepSeek-V4 sub-concepts:
  [[Compressed Sparse Attention]]          compress + sparse select KV
  [[Heavily Compressed Attention]]         compress hard, dense attention
  [[Manifold-Constrained Hyper-Connections]]  stable residual highway
  [[Muon Optimizer]]                       orthogonal gradient updates
  [[On-Policy Distillation]]               unify specialists post-RL
  [[GRPO]]                                 group-relative RL, no critic
```

---

## Reading Order

If you're new: **Transformer → Mixture-of-Experts → Mamba → Nemotron-3**

If you care about efficiency: **Mamba → Transformers Are SSMs → xLSTM → Nemotron-3**

If you're deploying: **KV Cache Optimization → Speculative Decoding → Hardware Acceleration → Nemotron-3**

If you care about silicon: **Hardware Acceleration → KV Cache Optimization → Hardware-Aware Scan → Flash Attention**
