---
title: "LLM Wiki Index"
tags: [index, meta]
tldr: "All papers, grouped by theme, with a concept glossary and a paragraph on how everything connects."
---

# LLM Wiki Index

> Every note in one place. Start here if you want the big picture.

---

## Theme I — Foundations

The paper that started everything. Everything else in this wiki is either built on it or reacting to it.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Transformer]] | Attention Is All You Need | 2017 | Self-attention replaces recurrence; any two tokens connect in one step, enabling parallel training and O(1) path length between positions. |

**Tags:** `foundational` `attention` `architecture` `parallelism`

---

## Theme II — Efficient Sequence Modeling

The O(n²) attention cost is fine for short sequences. For long ones it breaks. These papers replace attention with something cheaper, or generalize it.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Mamba]] | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2024 | Make SSM parameters input-dependent so the model selectively compresses context. Hardware-aware kernel fusion makes it practical. 5× inference throughput over Transformers. |
| [[Transformers Are SSMs]] | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | 2024 | Attention and selective SSMs are two faces of the same structured semiseparable matrix. The SSD framework unlocks Mamba-2: 2–8× faster than Mamba, matmul-friendly, TP-friendly. |
| [[xLSTM]] | xLSTM: Extended Long Short-Term Memory | 2024 | Modernize the 1997 LSTM with exponential gating + matrix memory + no hidden-to-hidden recurrence. Matches Transformer/Mamba quality at 1B+ scale, keeps O(1)-per-step inference. |

**Tags:** `ssm` `efficiency` `linear-time` `recurrence` `selectivity` `lstm` `gating` `mamba-2` `ssd`

---

## Theme III — Scaling / Parameter Efficiency

Parameters and compute don't have to scale together. These papers decouple them.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Mixture-of-Experts]] | Switch Transformers / Mixtral of Experts | 2022 | Route each token to a sparse subset of expert FFNs; parameters scale cheaply while per-token compute stays constant. More parameters, same FLOPs. |

**Tags:** `scaling` `moe` `routing` `efficiency` `sparse`

---

## Theme IV — Modern Synthesis

What happens when you combine the three themes above into a single production system.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Nemotron-3]] | Nemotron 3: Efficient and Open Intelligence | 2025 | Hybrid Mamba-2 + sparse attention + LatentMoE, trained at NVFP4. 120B/12B active. 7.5× throughput over Qwen3.5-122B at 1M context with competitive accuracy. |
| [[DeepSeek_V4]] | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 2026 | CSA + HCA compressed attention yields 10× KV cache reduction at 1M tokens. mHC residual connections + Muon optimizer. 1.6T/49B active. SOTA open model. |

**Tags:** `hybrid` `moe` `mamba` `production` `inference` `quantization` `rl` `agentic`

---

## Hardware & Systems

The silicon and memory systems the algorithms have to run on. Knowing the hardware constraints explains *why* every other paper in this wiki looks the way it does.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Hardware Acceleration for Neural Networks]] | Xu, Banerjee & Gupta — A Comprehensive Survey | 2025 | Survey of every chip family running NNs (GPU, TPU, NPU, FPGA, ASIC, LPU, in/near-memory, neuromorphic). The recurring theme: peak FLOPS doesn't matter — memory bandwidth and KV-cache I/O dominate end-to-end performance. |

**Tags:** `hardware` `gpu` `tpu` `npu` `fpga` `asic` `lpu` `in-memory` `kv-cache`

---

## Inference Optimization

Once a model is trained, the cost is moving bytes. These papers attack inference latency, throughput, and KV-cache memory directly.

| Note | Paper | Year | TL;DR |
|---|---|---|---|
| [[Speculative Decoding]] | (Concept note) | — | Fast draft model proposes K tokens; large model verifies all K in one parallel pass. Expected accepted tokens per call = K × acceptance-rate. With MTP heads as drafts, ~2× throughput. |
| [[KV Cache Optimization]] | Xu, Khaira & Singh — KV Cache Optimization Strategies for Scalable and Efficient LLM Inference | 2026 | Five families of KV-cache optimization (eviction, compression, hybrid memory, novel attention, combination), mapped to seven production scenarios. No single technique dominates — choose by workload. |

**Tags:** `inference` `kv-cache` `throughput` `eviction` `compression` `paged-attention` `speculative-decoding`

---

## Concept Glossary

Every `[[wikilink]]` that appears anywhere in this wiki, with a one-line definition.

| Concept | What it is | First introduced |
|---|---|---|
| [[Transformer]] | Self-attention-based encoder-decoder; the universal backbone of modern NLP | Vaswani et al. 2017 |
| [[Mamba]] | Selective SSM: input-dependent B, C, Δ parameters enable content-aware compression | Gu & Dao 2024 |
| [[Transformers Are SSMs]] | Structured State Space Duality — attention and selective SSMs are the same structured-matrix object; gives Mamba-2 | Dao & Gu 2024 |
| [[xLSTM]] | LSTM rehab: exponential gating + matrix memory + parallelizable mLSTM cell; competitive with Transformer at 1B+ | Beck, Hochreiter et al. 2024 |
| [[Mixture-of-Experts]] | Sparse routing: each token activates only K of N expert FFNs; decouples params from FLOPs | Fedus et al. 2022 |
| [[Nemotron-3]] | Hybrid Mamba-Transformer-LatentMoE; synthesizes Themes I–III into a production system | NVIDIA 2025 |
| [[DeepSeek_V4]] | MoE LLM with CSA+HCA compressed attention, mHC residual connections, Muon optimizer; 1M-token context at 27% of V3's FLOPs | DeepSeek-AI 2026 |

**DeepSeek-V4 concept pages** — new mechanisms introduced in [[DeepSeek_V4]]:

| Concept | What it is | First introduced |
|---|---|---|
| [[Compressed Sparse Attention]] | Compress $m$ tokens → 1 KV entry via learned weighted sum, then sparse top-k selection via Lightning Indexer; reduces KV cache to $n/m \cdot k$ entries | DeepSeek-AI 2026 |
| [[Heavily Compressed Attention]] | Compress $m' \gg m$ tokens → 1 entry, then full dense attention over the tiny sequence; coarser than CSA but cheaper | DeepSeek-AI 2026 |
| [[Manifold-Constrained Hyper-Connections]] | Expand residual stream to $n_\text{hc} \times d$; constrain mixing matrix to Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp so spectral norm ≤ 1 | DeepSeek-AI 2026 |
| [[Muon Optimizer]] | Orthogonalize gradient matrix via Newton-Schulz iterations before applying; isotropic update, faster convergence than AdamW on matrix weights | Jordan et al. 2024; used at scale in DeepSeek-V4 |
| [[On-Policy Distillation]] | Student generates own rollouts and minimizes reverse KL vs. ensemble of specialist teachers; no distribution shift, mode-seeking | Lu & Lab 2025; DeepSeek-AI 2026 |
| [[GRPO]] | PPO variant without a value function; normalizes advantages within a group of $G$ rollouts per prompt; cheaper and more stable than PPO for LLMs | DeepSeek-AI 2024–2026 |

**Stub pages** — key sub-concepts with their own notes:

| Term | Brief definition | Appears in |
|---|---|---|
| [[KV Cache]] | Stored key-value pairs from attention; grows linearly with sequence length | [[Transformer]], [[Mamba]], [[Nemotron-3]] |
| [[State Space Model]] | Continuous-time linear system discretized for sequences: h_t = Āh_{t-1} + B̄x_t | [[Mamba]] |
| [[Hardware-Aware Scan]] | Kernel fusion keeping SSM recurrence in SRAM; avoids HBM reads for intermediate states | [[Mamba]] |
| [[Load Balancing Loss]] | Auxiliary loss encouraging uniform token distribution across experts | [[Mixture-of-Experts]], [[Nemotron-3]] |
| [[LatentMoE]] | Project d → ℓ before routing; cuts all-to-all communication by d/ℓ, allows more experts | [[Nemotron-3]], [[Mixture-of-Experts]] |
| [[Multi-Token Prediction]] | Auxiliary heads predict 2, 3… tokens ahead; richer training signal + free speculative decoding | [[Nemotron-3]] |
| [[NVFP4]] | NVIDIA 4-bit float (E2M1 elements, 16-element micro-blocks); 3× peak throughput vs BF16 | [[Nemotron-3]] |
| [[Speculative Decoding]] | Draft tokens verified in one parallel pass; [[Multi-Token Prediction]] heads provide drafts | [[Nemotron-3]] |
| [[GQA]] | Grouped-query attention; 32 Q-heads share 2 KV-heads, cutting [[KV Cache]] size 16× | [[Nemotron-3]] |
| [[RLVR]] | RL with verifiable rewards across 21 simultaneous environments (math, code, tools, …) | [[Nemotron-3]] |

**SSM / xLSTM concept pages** — introduced or formalized in [[Transformers Are SSMs]] and [[xLSTM]]:

| Concept | What it is | First introduced |
|---|---|---|
| [[Selective State Space Model]] | An SSM whose B, C, Δ depend on the current input — the S6 cell at the heart of Mamba | Gu & Dao 2024 |
| [[Selective State Spaces]] | Synonym for [[Selective State Space Model]]; phrase used in Mamba's title | Gu & Dao 2024 |
| [[Semiseparable matrix]] | Matrix whose every off-diagonal block has rank ≤ N; the structure SSMs produce | Dao & Gu 2024 |
| [[Multi-head SSM]] | SSM split into H parallel heads, by analogy with multi-head attention | Dao & Gu 2024 |
| [[Tensor parallelism]] | Shard each layer's weights across GPUs in a node; Mamba-2 needs half the syncs of a Transformer | Shoeybi et al. 2019 |
| [[Sequence parallelism]] | Shard the sequence dimension across GPUs; for SSMs, pass the recurrent state at boundaries | Korthikanti et al. 2022 |
| [[Linear attention]] | Replace softmax with a kernel; rank-1 SSM in the SSD framework | Katharopoulos et al. 2020 |
| [[LSTM]] | 1997 Hochreiter–Schmidhuber recurrent network with constant-error carousel + sigmoid gates | Hochreiter & Schmidhuber 1997 |
| [[Constant error carousel]] | Identity path through the LSTM cell state; the trick that beat vanishing gradients | Hochreiter & Schmidhuber 1997 |
| [[Exponential gating]] | Replace LSTM's sigmoid gates with exp + max-stabilizer; lets a single token reset the cell | Beck et al. 2024 |
| [[Matrix memory]] | $d \times d$ matrix cell state updated by an outer product; learned key→value associative store | Beck et al. 2024 |
| [[Covariance update rule]] | $C_t = f_t C_{t-1} + i_t v_t k_t^\top$ — outer-product update at the heart of mLSTM | Beck et al. 2024 |
| [[sLSTM]] | xLSTM scalar-memory cell with exponential gating + cross-cell memory mixing | Beck et al. 2024 |
| [[mLSTM]] | xLSTM matrix-memory cell, fully parallelizable in training | Beck et al. 2024 |

**Hardware concept pages** — defined in [[Hardware Acceleration for Neural Networks]]:

| Concept | What it is | Where it shows up |
|---|---|---|
| [[GPU]] | Massively parallel SIMT processor with tensor cores + HBM; dominant DL accelerator | All training & inference |
| [[TPU]] | Google ASIC built around large systolic arrays; high matmul density | Google's training stack |
| [[NPU]] | Inference-focused accelerator for mobile/edge devices, INT8-optimized | Phone/laptop on-device inference |
| [[FPGA]] | Reconfigurable silicon — compile a circuit, not a program; flexible, less dense than ASIC | Niche or low-volume workloads |
| [[ASIC]] | Fixed-function silicon for one workload; best perf/W, expensive to design | Inferentia, TPU, LPUs |
| [[LPU]] | Language Processing Unit — ASIC engineered for low-latency LLM inference | Groq, SambaNova |
| [[Systolic array]] | 2D MAC grid that pipelines operands across rows and columns | TPUs, many ASICs |
| [[Tensor Cores]] | Specialized matmul engines inside modern GPUs; produce a small dense matmul per clock | Every NVIDIA GPU since V100 |
| [[HBM]] | Stacked DRAM next to the accelerator die; ~3–8 TB/s; the dominant memory tier | All modern GPUs/TPUs |
| [[SRAM]] | On-chip cache (KB–MB), 10–100× faster than HBM; where you want hot data | Every accelerator |
| [[Memory hierarchy]] | Registers → SRAM → HBM → host RAM → SSD; performance set by where data lives | Universal |
| [[Kernel fusion]] | Merge consecutive ops into one kernel to keep operands in SRAM | FlashAttention, Mamba scan |
| [[Operator fusion]] | Compiler-level kernel fusion (XLA, Inductor, TVM) | Modern DL compilers |
| [[Hardware-aware algorithms]] | Algorithms designed around the memory hierarchy and parallelism of the target chip | FlashAttention, Mamba scan |
| [[Flash Attention]] | IO-aware exact attention that keeps softmax in SRAM; 2–4× wall-clock faster | Production attention kernels |
| [[In-memory computing]] | Multiply-accumulate inside analog memory crossbars | Research / exotic chips |
| [[Near-memory computing]] | Compute next to memory rather than inside it (HBM-PIM, UPMEM) | Research / emerging |
| [[Neuromorphic computing]] | Spiking-neuron-inspired chips (Loihi, TrueNorth) | Niche; not yet for LLMs |
| [[Quantization-aware datapath]] | Hardware multipliers that natively execute INT8/FP8/[[NVFP4]] | Modern tensor cores |

**Inference & KV-cache concept pages** — defined in [[KV Cache Optimization]] and [[Speculative Decoding]]:

| Concept | What it is | Family |
|---|---|---|
| [[Prefill]] | Initial parallel pass that processes the prompt and populates the KV cache | Inference phase |
| [[Decode phase]] | Autoregressive generation pass; bandwidth-bound, where KV-cache tricks pay off | Inference phase |
| [[Cache eviction]] | Drop KV-cache entries to bound memory | KV-cache family 1 |
| [[Cache compression]] | Keep all entries but shrink each (INT8/INT4/[[NVFP4]], low-rank) | KV-cache family 2 |
| [[Hybrid memory]] | Tier the cache across HBM / DRAM / NVMe with paging | KV-cache family 3 |
| [[Novel attention mechanisms]] | Architectural changes that shrink the cache structurally (GQA, MQA, CSA, HCA) | KV-cache family 4 |
| [[Combination strategies]] | Stack two or more of the above; the production reality | KV-cache family 5 |
| [[Eviction policy]] | Rule deciding which entries to drop (LRU / attention-score / sinks-aware) | Eviction sub-knob |
| [[Token budget]] | Hard cap on cache size that the eviction policy works within | Eviction sub-knob |
| [[Sliding window attention]] | Keep the last W tokens; oldest fall off | Eviction recipe |
| [[Attention sinks]] | First few tokens accumulate disproportionate attention; cannot be evicted | Eviction recipe |
| [[H2O eviction]] | Drop tokens with lowest accumulated attention scores | Eviction recipe |
| [[Paged attention]] | Store cache in 16-token pages like OS virtual memory; vLLM's foundation | Hybrid-memory recipe |
| [[KV offloading]] | Page cold blocks to host DRAM or NVMe | Hybrid-memory recipe |
| [[Multi-Query Attention]] | All query heads share one K/V projection; H× cache reduction | Architectural |
| [[Draft model]] | Small/fast model in [[Speculative Decoding]] that proposes K tokens | Speculative decoding |
| [[Verification step]] | Single parallel pass through the target model that scores all K draft tokens | Speculative decoding |

**Inline concepts** — too granular for their own page, defined where they first appear:

| Term | Brief definition | Appears in |
|---|---|---|
| Self-attention | Dot-product of Q, K, V matrices; every token attends to every other in one step | [[Transformer]] |
| Multi-head attention | Run h attention heads in parallel, each learning different relationship types | [[Transformer]] |
| Selective SSM (S6) | SSM where B, C, Δ depend on the current input x (vs. fixed constants in S4) | [[Mamba]] |
| Top-K routing | Router selects the K highest-scoring experts per token; all others contribute zero | [[Mixture-of-Experts]] |
| Expert capacity | Max tokens an expert processes per batch; overflow tokens skip the layer via residual | [[Mixture-of-Experts]] |
| NVFP4 PTQ | Post-training quantization to NVFP4 via AutoQuantize NAS-style sensitivity search | [[Nemotron-3]] |
| WSD schedule | Warmup–Stable–Decay LR schedule; stable phase supports checkpoint merging | [[Nemotron-3]] |
| Checkpoint merging | Weighted average of checkpoints from different training windows; 2–4 point free quality gain | [[Nemotron-3]] |
| LC-Phase | Continued pre-training at 1M context (34B tokens) enabling million-token inference | [[Nemotron-3]] |
| PivotRL | Reuses offline SFT traces; focuses RL updates on high-uncertainty "pivot" turns | [[Nemotron-3]] |
| Async GRPO | Group Relative Policy Optimization with asynchronous rollout generation | [[Nemotron-3]] |
| Mamba SSM cache | Recurrent state cache; quantized to FP16 with stochastic rounding to prevent verbosity | [[Nemotron-3]], [[Mamba]] |

---

## How These Papers Connect

The Transformer (2017) solved sequential training by replacing RNNs with self-attention, achieving O(1) path length between any two tokens — but at the cost of O(n²) compute and a KV cache that grows without bound. That single limitation became the organizing constraint for the next eight years of research.

Mamba (2024) attacks the O(n²) problem directly: it shows that the failure mode of older recurrent models was not compression itself but *blind* compression. Make the SSM parameters depend on the input and you get selective memory — the model decides what to keep and what to forget based on content, not just position. A hardware-aware kernel fusion pass makes this as fast in practice as FlashAttention. Mamba matches Transformer quality at 5× throughput and constant memory per step.

Mixture-of-Experts attacks a different axis: not compute per token but the coupling between parameter count and compute. A dense model can't add knowledge without adding FLOPs. MoE breaks that coupling — route each token to K of N experts, so total parameters scale with N while active compute scales only with K. Switch Transformers prove that top-1 routing works; Mixtral proves that top-2 of 8 SwiGLU experts outperforms LLaMA-2 70B at 5× fewer active parameters.

Nemotron-3 (2025) is the synthesis. It takes Mamba's insight (most sequence processing is cheap recurrence; exact recall needs occasional attention) and MoE's insight (parameters and compute are independent axes), adds LatentMoE to cut the all-to-all communication bottleneck by routing in a compressed latent space, trains the whole thing at NVFP4 precision for 3× hardware throughput, uses multi-token prediction to densify the training signal and enable speculative decoding at inference, and then runs 21-environment simultaneous RL to specialize the model for agentic workloads. The result is a model that is simultaneously cheaper to run, better at long context, and more capable on reasoning tasks than any pure Transformer MoE of comparable size.

DeepSeek-V4 (2026) attacks the same O(n²) problem from a different angle: instead of replacing attention entirely, it keeps the Transformer backbone but radically compresses what attention has to look at. Compressed Sparse Attention (CSA) squeezes every m tokens into one KV entry then sparsely selects the top-k relevant blocks; Heavily Compressed Attention (HCA) compresses even more aggressively and does dense attention over the tiny result. Interleaving the two cuts the KV cache by 10× at 1M tokens relative to DeepSeek-V3.2. Two additional innovations compound the gains: Manifold-Constrained Hyper-Connections (mHC) expand the residual stream and constrain the mixing matrix to be doubly stochastic — guaranteeing spectral norm ≤ 1 and numerically stable deep stacking — and the Muon optimizer replaces AdamW for most layers by orthogonalizing gradient updates via Newton-Schulz iterations, yielding faster convergence on ill-conditioned loss landscapes. Post-training follows a specialist-then-distill paradigm: independent expert models are trained per domain via SFT + GRPO, then unified into a single model through on-policy distillation. DeepSeek-V4-Pro-Max sets the open-model SOTA on reasoning and knowledge benchmarks.

[[Transformers Are SSMs]] (2024) closes the loop on the SSM-vs-attention debate: Dao & Gu prove the two families are different parameterizations of the same structured-matrix object. Selective SSMs are exactly the rank-$N$ semiseparable case; full attention is the unstructured case; linear attention is rank-1. The "Structured State Space Duality" framework gives the new layer **Mamba-2**, whose core is 2–8× faster than Mamba while remaining matmul-friendly enough to inherit Tensor Parallelism, Sequence Parallelism, and FlashAttention-style IO awareness from the Transformer toolbox.

[[xLSTM]] (2024, Hochreiter et al.) takes the third path — neither attention nor SSM, but a modernized [[LSTM]]. Two surgical fixes — exponential gating (so a single token can override stored memory) and a $d \times d$ matrix memory updated via outer product (so the cell becomes a learned key-value store) — close the gap with Transformer and Mamba at billion-parameter scale, while keeping LSTM's $O(1)$-per-step inference cost.

Underneath every algorithm is the silicon. The [[Hardware Acceleration for Neural Networks]] survey (2025) maps the accelerator landscape — GPUs, TPUs, NPUs, FPGAs, ASICs, the new LLM-serving LPUs, and exotic in-/near-memory designs — and shows why the binding constraint for modern LLMs is *not* peak FLOPS but **memory bandwidth and KV-cache I/O**. That same observation drives [[KV Cache Optimization]] (2026), which catalogs the five families of techniques (eviction, compression, hybrid memory, novel attention, combination) production teams use to keep the cache from blowing up — and shows that no single strategy dominates; the right tool is workload-specific. [[Speculative Decoding]] is the complementary inference-side trick: amortize one verifier pass across multiple draft tokens for ~2× throughput at zero quality loss.

The through-line: every paper in this wiki is asking the same question — *how do you get the most intelligence per FLOP, per byte moved, per joule?* — and each one pries open a different dimension: computation graph (Transformer), memory management (Mamba), structural unification (Transformers Are SSMs), modernized recurrence (xLSTM), parameter efficiency (MoE), hardware utilization (Nemotron-3, Hardware Acceleration survey), long-context attention efficiency (DeepSeek-V4, KV Cache Optimization), and inference throughput (Speculative Decoding).

---

## Benchmarks & Evaluation

Quantitative comparisons across frontier models on knowledge, long context, and agentic tasks.

| Note | Coverage | Year | TL;DR |
|---|---|---|---|
| [[LLM Benchmarks]] | DS-V4-Pro, DS-V4-Flash, K2.6, GLM-5.1, Opus-4.6, GPT-5.4, Gemini-3.1-Pro — 22 benchmarks | 2025 | Gemini leads knowledge, Opus-4.6 leads long context, agentic tasks split across GPT-5.4 / Opus-4.6 / K2.6; DS-V4-Pro is strongest open model. |

**Tags:** `benchmarks` `evaluation` `llm-comparison`

---

## Reading Order

**If you're new:** [[Transformer]] → [[Mixture-of-Experts]] → [[Mamba]] → [[Nemotron-3]]

**If you care about efficiency:** [[Mamba]] → [[Transformers Are SSMs]] → [[xLSTM]] → [[Nemotron-3]]

**If you're deploying:** [[KV Cache Optimization]] → [[Speculative Decoding]] → [[Hardware Acceleration for Neural Networks]] → [[Nemotron-3]]

**If you want the math:** [[Transformer]] (attention) → [[Mamba]] (SSM discretization + selection) → [[Transformers Are SSMs]] (semiseparable matrices, SSD) → [[xLSTM]] (exponential gating + matrix memory) → [[Mixture-of-Experts]] (load balancing loss) → [[Nemotron-3]] (LatentMoE projection + NVFP4 format)

**If you care about silicon:** [[Hardware Acceleration for Neural Networks]] → [[KV Cache Optimization]] → [[Hardware-Aware Scan]] → [[Flash Attention]]
