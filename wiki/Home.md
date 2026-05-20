# LLM Wiki

AI research knowledge base — papers, concepts & intuitions. Curated by Saqlain.

---

## Foundations

| Article | One Line |
|---|---|
| [[Transformer]] | Self-attention replaces recurrence. Any token attends to any other in O(1). |
| [[LLaMA 2]] | Meta's open RLHF-aligned 7B–70B models. GQA at scale. The base for most community finetunes. |
| [[Mamba]] | Selective state spaces. Linear-time sequence modeling that matches Transformer quality. |

## Efficient Sequence Modeling

| Article | One Line |
|---|---|
| [[S4]] | First SSM to solve 16K-step tasks. HiPPO matrix + Cauchy kernel. The theoretical root of Mamba. |
| [[RWKV]] | Linear attention as an RNN. Trains like a Transformer, decodes with O(1) memory. Scaled to 14B. |
| [[RetNet]] | Three computation modes (parallel/recurrent/chunkwise). 15.6× throughput over Transformer at inference. |
| [[Transformers Are SSMs]] | Attention and selective SSMs are the same structured-matrix object. SSD framework gives Mamba-2: 2–8× faster, matmul-friendly. |
| [[xLSTM]] | LSTM rehab — exponential gating + matrix memory. Competitive with Transformer at 1B+ scale, O(1) inference. |
| [[Griffin]] | Google DeepMind hybrid: RG-LRU + local attention. Matches LLaMA-2 at 7× fewer training tokens. |

## Scaling

| Article | One Line |
|---|---|
| [[Mixture-of-Experts]] | Decouple parameters from compute. Route tokens to specialized FFN subnetworks. |

## Hardware & Systems

| Article | One Line |
|---|---|
| [[Flash Attention]] | IO-aware exact attention: tiles into SRAM, never writes T×T matrix to HBM. 2–4× wall-clock speedup. |
| [[FlashAttention-2]] | Better warp scheduling pushes GPU to 50–73% peak FLOPS. 225 TFLOPS/s per A100 training GPT models. |
| [[Hardware Acceleration for Neural Networks]] | Survey of GPUs, TPUs, NPUs, FPGAs, ASICs, LPUs, in/near-memory. The bottleneck is bandwidth, not FLOPS. |

## Inference Optimization

| Article | One Line |
|---|---|
| [[Medusa]] | K extra decoding heads + tree attention. 2.2× lossless speedup. No separate draft model needed. |
| [[EAGLE]] | Feature-level draft model. 3–3.5× lossless speedup on LLaMA-2-Chat 70B — best in the speculative family. |
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
Transformer (2017, ~100K citations)
│
├── Problem: O(n²) attention cost at long sequences
│   ├──► S4 (2022, ~3K): SSM + HiPPO matrix solves Path-X (16K steps)
│   ├──► RWKV (2023, ~2K): linear attention = O(1) RNN inference, 14B scale
│   ├──► RetNet (2023, ~1K): parallel/recurrent/chunkwise triple mode; 15.6× throughput
│   ├──► Mamba (2024): selective SSM; 5× inference throughput
│   ├──► Transformers Are SSMs (2024): unify attention + SSM via SSD; Mamba-2 2–8× faster
│   ├──► xLSTM (2024): exponential gating + matrix memory; O(1) inference
│   ├──► Griffin (2024, ~1K): RG-LRU + local attention; matches LLaMA-2 at 7× fewer tokens
│   └──► DeepSeek-V4 (2026): CSA + HCA compress KV cache 10× at 1M tokens
│
├── Problem: Parameters tied to compute
│   └──► Mixture-of-Experts: sparse routing, constant FLOPs/token
│
├── Problem: MoE communication + latency bottlenecks
│   └──► Nemotron-3 LatentMoE: route in latent space
│
├── Problem: Attention kernel too slow / memory-intensive
│   ├──► FlashAttention (2022, ~8K): IO-aware tiling; 2–4× speedup, zero approximation
│   └──► FlashAttention-2 (2023, ~4K): better warp scheduling; 50–73% GPU utilization
│
├── Problem: Inference is memory-bandwidth bound
│   ├──► Medusa (2024, ~1.5K): K extra heads + tree attention; 2.2× lossless speedup
│   ├──► EAGLE (2024, ~1.2K): feature-level drafting; 3–3.5× lossless speedup
│   ├──► Speculative Decoding: amortize verifier passes across K draft tokens
│   └──► KV Cache Optimization (2026): 5 families — eviction, compression,
│        hybrid memory, novel attention, combination
│
├── Problem: Open RLHF-aligned models
│   └──► LLaMA 2 (2023, ~10K): 7B–70B + GQA + detailed RLHF recipe; democratized finetuning
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

If you're new: **Transformer → LLaMA 2 → Mixture-of-Experts → Mamba → Nemotron-3**

If you care about efficiency: **S4 → Mamba → Transformers Are SSMs → xLSTM → RWKV → RetNet → Griffin**

If you're deploying: **Flash Attention → FlashAttention-2 → KV Cache Optimization → Medusa → EAGLE → Speculative Decoding**

If you care about silicon: **Hardware Acceleration → Flash Attention → FlashAttention-2 → Hardware-Aware Scan → KV Cache Optimization**

If you want the linear-RNN lineage: **S4 → RWKV → RetNet → Griffin → Mamba → Transformers Are SSMs**
