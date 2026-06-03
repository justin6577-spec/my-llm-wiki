---
title: Hardware Acceleration for Neural Networks: A Comprehensive Survey
authors: Bin Xu, Ayan Banerjee, Sandeep Gupta
year: 2026
arxiv: 2512.23914
tags: [hardware, inference, attention, benchmarks, foundational, moe]
citation_count: 0
tldr: A comprehensive survey of neural network hardware acceleration covering GPUs, TPUs, FPGAs, ASICs, LPUs, and in-memory computing, unified under a taxonomy across workloads, execution settings, and optimization levers, identifying memory movement — not arithmetic — as the dominant bottleneck in modern LLM-era systems.
---

## The Problem

For a long time, the intuition was simple: faster neural networks = more FLOPs. Throw more tensor cores at the problem and you win. But that mental model has broken down badly. As models scaled from millions to hundreds of billions of parameters — and as context lengths for LLMs stretched to tens of thousands of tokens — the real bottleneck shifted decisively to **data movement**: how quickly you can shuffle activations, weights, KV-cache entries, and optimizer states through memory hierarchies and across interconnects. Peak arithmetic throughput sits largely idle while the chip waits for data.

The diversity of deployment contexts made things worse. You now need the same class of model to run efficiently in a datacenter training cluster (optimize for throughput, scalability, time-to-train), a latency-critical inference server (optimize for tail latency and token generation rate), and a battery-powered edge device (optimize for energy per inference, area, and thermal budget). No single design point handles all three, and the tradeoffs between compute, memory bandwidth, and communication are fundamentally different across these settings.

On top of that, modern workloads have grown structurally more complex. Transformers introduce quadratic attention with massive KV-caches that scale with sequence length and batch concurrency. Mixture-of-Experts (MoE) models route tokens through sparse subnetworks, creating irregular, load-imbalanced execution patterns. Graph Neural Networks have irregular memory access patterns that defeat standard cache hierarchies. The result is a landscape where hardware design must span the full stack — compute datapaths, on-chip memory, HBM, interconnects, compilers, and runtimes — rather than just optimizing the matrix multiply unit.

## The Idea

The core insight of this survey is that **efficient neural network acceleration is a full-system co-design problem**, not a chip microarchitecture problem, and the unifying organizing principle is that data movement energy and latency dominate arithmetic cost in every modern workload of consequence.

Unpacking that: moving a byte from off-chip DRAM costs orders of magnitude more energy than performing a floating-point multiply-add. Attention's KV-cache grows linearly with sequence length times batch size, quickly exhausting on-chip SRAM and forcing repeated expensive HBM accesses. Sparsity can theoretically reduce compute but creates irregular access patterns that are hard to exploit without specialized hardware. Quantization reduces both arithmetic cost and memory footprint simultaneously — making it doubly valuable — but requires quantization-aware datapaths and careful accumulation to avoid quality loss. The survey synthesizes all of these angles into a unified taxonomy so the tradeoffs become legible.

## How It Works

The survey organizes the acceleration landscape along three axes:

**1. Workloads:** CNNs (spatially regular, convolution-dominated), RNNs (sequential, recurrence-bottlenecked), GNNs (sparse, irregular graph traversal), and Transformers/LLMs (attention, softmax, KV-cache, MoE routing). Each workload has a different compute-to-memory ratio and memory access pattern, which means the optimal hardware looks different for each.

**2. Execution settings:** Training (large batch, gradient accumulation, optimizer state, mixed precision) vs. inference (low latency, KV-cache paging, dynamic batching). Datacenter vs. edge (area and power constraints, quantization, pruning). Online (single-request latency) vs. offline (throughput maximization).

**3. Optimization levers:**
- **Reduced precision:** FP16/BF16 for training, INT8/INT4 for inference. Reduces memory traffic and increases arithmetic density simultaneously. Typically uses low-precision compute with higher-precision accumulation.
- **Sparsity and pruning:** Structured sparsity (e.g., 2:4 fine-grained sparsity on NVIDIA Ampere) enables hardware skip logic; unstructured sparsity is harder to exploit efficiently due to irregular memory access.
- **Operator fusion:** Fusing adjacent kernels (e.g., attention + softmax + dropout) eliminates intermediate memory round-trips. FlashAttention is the canonical example: it tiles attention computation to stay in SRAM, dramatically reducing HBM traffic.
- **Compilation and scheduling:** Compilers like XLA, TVM, and Triton perform layout transformations, tiling, loop reordering, and kernel selection to match hardware capabilities to model semantics.
- **Memory and interconnect design:** Systolic arrays for weight-stationary matrix multiply, High-Bandwidth Memory (HBM) stacked on-package, NVLink/NVSwitch for multi-GPU all-reduce, KV-cache paging (à la vLLM's PagedAttention) for efficient LLM serving.

**Key architectural patterns covered:**
- **Systolic arrays** (TPU's heart): weight matrices flow in fixed directions, activations pump through — maximizes reuse with minimal control overhead.
- **Tensor cores** (NVIDIA): 4×4 matrix multiply-accumulate units that operate on FP16/BF16/INT8/FP8 inputs, giving ~8× throughput uplift over CUDA cores on dense matrix ops.
- **Vector/SIMD engines:** Efficient for element-wise ops (activations, normalization, softmax).
- **LPUs (Language Processing Units, e.g., Groq):** Deterministic, compiler-scheduled dataflow chips with no caches and no branch prediction — optimized for predictable low-latency autoregressive token generation.
- **In-/near-memory computing:** Move computation to where data lives, eliminating the memory wall at the cost of reduced programmability and precision.
- **Neuromorphic/analog:** Event-driven, sparse computation for ultra-low-power workloads; still far from mainstream ML models.

## Key Results

This is a survey paper, so it synthesizes results from across the literature rather than reporting new benchmarks. Key quantitative data points it references and contextualizes include:

- **Data movement energy dominance:** Moving data through memory hierarchies can cost 100–1000× more energy per operation than the arithmetic itself — the central motivation for locality-aware dataflows and on-chip buffer sizing.
- **Quantization gains:** INT8 inference can reduce energy and improve effective memory bandwidth utilization significantly vs. FP32, while maintaining acceptable accuracy on most tasks with quantization-aware training.
- **FlashAttention:** Reduces attention's HBM memory reads/writes by ~5–20× for typical sequence lengths by tiling into SRAM, enabling longer context without memory blowup.
- **Tensor core utilization:** Without careful tiling and layout alignment, real-world GPU utilization on ML workloads can fall well below 50% of peak FLOPs — illustrating the gap between peak arithmetic throughput and actual efficiency.
- **KV-cache bottleneck:** For LLM inference, KV-cache memory consumption scales as `2 × num_layers × num_heads × head_dim × seq_len × batch_size`, quickly dominating available HBM at long contexts.

## Limitations

As a survey, it explicitly acknowledges several open problems it does not resolve:

- **KV-cache management at scale:** Efficient paging, eviction, and compression for very long contexts remains an active research area with no settled solution.
- **Dynamic and sparse workloads:** MoE routing and unstructured sparsity create load imbalance that current hardware handles poorly — utilization drops sharply.
- **Fair benchmarking:** Reported numbers across papers are often incomparable due to different batch sizes, precision settings, hardware generations, and software stack versions. The survey calls out the reproducibility problem but cannot fix it.
- **Energy- and security-aware deployment:** Side-channel attacks via power analysis, and the challenge of measuring true end-to-end energy (not just chip TDP), remain underaddressed.
- **Analog and neuromorphic:** Promising for energy efficiency but lack the programmability and precision needed for mainstream LLM-class models today.
- **Coverage:** The survey focuses on current technology; it does not deeply cover training efficiency at the distributed systems level (e.g., pipeline parallelism strategies, gradient compression).

## Why It Matters

This survey arrives at a moment when the hardware ecosystem for AI has fragmented dramatically: every major cloud provider has custom silicon (Google TPU, AWS Trainium/Inferentia, Microsoft Maia), startups are building LPU-class inference chips, and edge SoCs now ship with dedicated NPU blocks. Without a unified framework for reasoning about the tradeoffs, practitioners make poor hardware selection and optimization decisions, and researchers reinvent the same architectural wheels.

By organizing everything under the workload × execution setting × optimization lever taxonomy and hammering on the memory-movement-dominates-arithmetic theme, the survey gives ML engineers a mental model for diagnosing bottlenecks (is your kernel memory-bound or compute-bound? check the roofline), and gives hardware architects a map of where the unsolved problems are. It also connects the software stack — compilers, runtimes, kernel libraries — to hardware, making clear that a great chip with a bad compiler stack delivers poor real-world performance.

For the LLM era specifically, the synthesis of KV-cache management, FlashAttention, quantization-aware datapaths, and LPU-style deterministic scheduling in one place is particularly valuable as the field moves from "can we train this model?" to "can we serve it at scale, cheaply?"

## See Also

[[Transformer]] · [[Attention Is All You Need]] · [[FlashAttention]] · [[Mixture of Experts]] · [[Quantization]] · [[Systolic Array]] · [[Roofline Model]] · [[PagedAttention]] · [[Tensor Processing Unit]] · [[High-Bandwidth Memory]]
