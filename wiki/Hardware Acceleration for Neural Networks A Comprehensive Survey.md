---
title: Hardware Acceleration for Neural Networks: A Comprehensive Survey
authors: Bin Xu, Ayan Banerjee, Sandeep Gupta
year: 2026
arxiv: 2512.23914
tags: [hardware, inference, attention, benchmarks, foundational, moe]
citation_count: 0
tldr: A comprehensive survey of neural network hardware acceleration spanning GPUs, TPUs, FPGAs, ASICs, and LPUs, showing that end-to-end efficiency is increasingly bottlenecked by data movement rather than arithmetic throughput across the full system stack.
---

## The Problem

Deep neural networks have grown explosively in model size, architectural diversity (CNNs, GNNs, Transformers, diffusion models), and deployment environments (datacenter clusters, latency-critical edge devices, battery-powered mobile). The naive assumption that "more FLOPs = faster models" breaks down almost immediately in practice. Peak arithmetic throughput is rarely the bottleneck — instead, data movement through memory hierarchies and interconnects dominates end-to-end runtime and energy consumption.

Modern LLMs make this concrete and painful. During inference, the KV-cache grows linearly with sequence length and batch size, creating massive memory-capacity and bandwidth demands that dwarf the cost of the matrix multiplications themselves. During training, activation storage and optimizer state (e.g., Adam stores two momentum buffers per parameter) push memory requirements into hundreds of gigabytes for frontier models. Without specialized hardware and software co-design, many state-of-the-art models would simply be impractical to deploy — too slow, too expensive, or too power-hungry.

The hardware landscape has fragmented in response: GPUs with tensor cores, Google's TPUs, mobile NPUs, FPGA-based accelerators, ASIC inference engines, and LLM-specific LPUs (Language Processing Units) all exist on a spectrum between generality and efficiency. Yet there has been no unified framework for understanding how workload characteristics, execution settings, and optimization strategies interact across this entire landscape. This survey attempts to provide exactly that.

## The Idea

The core insight is that neural network acceleration is a **full-stack, multi-objective optimization problem** — you can't optimize compute, memory, interconnect, and software in isolation, and improvements on one axis routinely degrade another. The survey organizes everything along three axes: (i) workloads, (ii) execution settings, and (iii) optimization levers, then synthesizes design patterns that generalize across devices.

Think of it like this: a GPU is a general-purpose workhorse with massive SIMD parallelism, but it pays a high energy cost for flexibility. A systolic array (like in a TPU) is a rigid conveyor belt of multiply-accumulate units — if your workload fits the belt, you get extraordinary efficiency; if it doesn't, you're in trouble. The real world demands systems that can slide along this spectrum dynamically, which is why software stacks (compilers, runtimes, kernel libraries) are just as important as the silicon itself.

## How It Works

**The Taxonomy — Three Axes:**

1. **Workloads**: CNNs (convolution-heavy, spatially local), RNNs (sequential, hard to parallelize), GNNs (irregular, sparse graph structure), Transformers/LLMs (attention + KV-cache, quadratic in sequence length). Each has a distinct arithmetic intensity and memory access pattern.

2. **Execution Settings**: Training (needs backward pass, gradient accumulation, optimizer state — memory-capacity-bound) vs. inference (needs low latency and high throughput — often memory-bandwidth-bound). Datacenter (cost per token, throughput) vs. edge (watts, chip area, latency SLA).

3. **Optimization Levers**:
   - **Reduced precision**: FP16/BF16 training with FP32 accumulation; INT8/INT4 inference. Each halving of precision roughly doubles arithmetic throughput and halves memory traffic.
   - **Sparsity and pruning**: Unstructured sparsity is hard to exploit on dense hardware; structured sparsity (e.g., NVIDIA's 2:4 sparsity) gives 2× speedup on tensor cores.
   - **Operator fusion**: Instead of writing an intermediate tensor to DRAM between two ops (e.g., LayerNorm → Linear), fuse them into one kernel that keeps data in registers/SRAM. FlashAttention is the canonical example — it tiles attention computation to avoid materializing the full N×N attention matrix.
   - **Tiling and dataflow**: Systolic arrays succeed by reusing operands across a 2D grid of processing elements; the key is choosing tile sizes that maximize arithmetic intensity (FLOPs / bytes moved). The roofline model makes this precise: if ops/byte > hardware compute/bandwidth ratio, you're compute-bound; otherwise, you're memory-bound.
   - **Memory hierarchy and interconnects**: HBM (High-Bandwidth Memory) stacked on the same package gives ~3 TB/s vs. ~50 GB/s for LPDDR. NVLink gives ~900 GB/s GPU-to-GPU vs. PCIe's ~64 GB/s. These numbers matter enormously for distributed training and multi-GPU inference.

**Hardware Taxonomy:**

| Platform | Strength | Weakness |
|---|---|---|
| GPU + Tensor Cores | Flexible, programmable, huge ecosystem | Energy per op, memory bandwidth wall |
| TPU / NPU (systolic array) | Throughput/watt for dense matmul | Irregular ops, dynamic shapes |
| FPGA | Reconfigurable, custom I/O pipelines | Lower peak throughput, complex toolchain |
| ASIC inference engines | Best throughput-per-watt | Fixed function, long design cycles |
| LPU | Predictable low-latency token generation | Narrow workload focus |
| In/near-memory computing | Eliminates data movement for bandwidth-bound ops | Programming model, yield, analog noise |
| Neuromorphic | Sparse, event-driven, ultra-low power | Limited model support, immature software |

**Software Stack:**
Frameworks (PyTorch, JAX) → compilers (XLA, TVM, torch.compile) → runtimes and kernel libraries (cuDNN, cuBLAS, Triton). The compiler's job is to lower high-level model semantics to hardware-specific tile sizes, memory layouts, and instruction schedules. Without this layer, even perfectly designed silicon underperforms.

## Key Results

This is a survey paper, so there are no novel experimental results. Instead it synthesizes existing benchmarks and design principles:

- **Data movement energy dominates**: Moving a byte from DRAM costs ~100–200× more energy than a floating-point multiply-add in registers. This single number motivates nearly every architectural choice discussed.
- **Mixed-precision training**: FP16/BF16 compute with FP32 accumulation typically achieves 2–8× higher throughput on tensor cores vs. FP32-only, with negligible accuracy loss when done correctly.
- **INT8 inference quantization**: Can reduce energy and increase throughput by 2–4× vs. FP16 on supported hardware (e.g., NVIDIA A100/H100 INT8 tensor cores).
- **FlashAttention**: Reduces attention memory from O(N²) to O(N) by tiling, enabling 2–4× speedup and enabling much longer context windows without OOM errors.
- **KV-cache paging (vLLM)**: Near-zero memory waste vs. 60–80% wasted memory in naive KV-cache implementations, enabling significantly higher throughput for LLM serving.
- **Sparsity (2:4 structured)**: NVIDIA's Ampere architecture achieves 2× speedup for layers with 50% structured sparsity, with typical accuracy drops under 1% after fine-tuning.

## Limitations

As a survey, the paper's main limitation is coverage depth versus breadth — it necessarily provides a high-level synthesis rather than deep technical derivations for any single topic. Several specific gaps stand out:

- **Dynamic and irregular workloads** (GNNs, MoE routing, variable-length sequences) remain poorly handled by most accelerators, and the survey acknowledges this is an open problem rather than a solved one.
- **Analog and neuromorphic** approaches are covered briefly; they remain far from production deployment and the survey can't meaningfully benchmark them against digital CMOS.
- **Security and privacy** considerations (model extraction, side-channel attacks on accelerators) are mentioned as a challenge but not deeply analyzed.
- **Benchmarking reproducibility** is flagged as a critical open problem — different papers measure latency, throughput, and energy under incomparable conditions, making it nearly impossible to do fair cross-platform comparisons.
- The survey was written through early 2026 and will age as the hardware landscape evolves rapidly (e.g., new GPU generations, custom silicon from hyperscalers).

## Why It Matters

This survey arrives at a pivotal moment when the gap between "what models need" and "what commodity hardware can deliver" is widening fast. LLMs with trillion-parameter MoE architectures, 128K+ token contexts, and real-time serving requirements are pushing every part of the stack simultaneously — compute, memory capacity, memory bandwidth, and interconnect.

The key contribution is providing a **unified mental model**: if you understand the roofline model, the arithmetic intensity of your workload, and where your bottleneck sits in the memory hierarchy, you can reason about which hardware platform is right and which optimizations will actually help. This framework is invaluable for ML engineers choosing infrastructure, hardware designers prioritizing silicon features, and researchers deciding which algorithmic optimizations are worth pursuing.

It also clearly frames the next generation of open problems: KV-cache management at scale (paging, compression, offloading), robust compiler support for dynamic shapes and sparse operators, energy-aware deployment for always-on edge devices, and standardized benchmarking methodology. These are the places where the research community has the most leverage.

## See Also

[[Transformer]] · [[Attention Is All You Need]] · [[FlashAttention]] · [[Roofline Model]] · [[Mixture of Experts]] · [[Quantization-Aware Training]] · [[vLLM]] · [[Tensor Processing Unit]] · [[Systolic Array]] · [[High-Bandwidth Memory]]
