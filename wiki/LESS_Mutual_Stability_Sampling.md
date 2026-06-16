---
title: "LESS: Mutual-Stability Sampling for Diffusion Language Models"
tags: [transformer, speculative decoding, LLM inference, attention, language model]
year: 2026
tldr: "LESS is a training-free, model-agnostic adaptive sampler for diffusion LLMs that treats token commitment as an online stopping problem. Uses mutual-stability sampling: a masked position is unmasked only when its top-1 prediction has high confidence, persists across recent reverse steps, and the distribution is stable under Jensen-Shannon divergence. Evaluated on Dream-7B, LLaDA-8B, LLaDA-1.5-8B; improves accuracy while using 72.1% fewer reverse steps than fixed-budget decoding."
wikilinks: [[transformer]] [[LLM inference]] [[speculative decoding]] [[attention]]
---
# LESS: Mutual-Stability Sampling

**Authors**: arXiv:2606.16908
**Year**: 2026

LESS addresses the efficiency bottleneck in [[diffusion large language models]] (dLLMs), where fixed-budget decoding wastes computation on already-stable positions and commits unstable ones too early.

## Method

**Mutual-stability sampling** — a token is eligible for unmasking when all three conditions hold:
1. **High confidence**: Top-1 prediction probability is high
2. **Temporal persistence**: Top-1 token persists across recent reverse steps
3. **Distribution stability**: Predictive distribution is stable under top-K inter-step Jensen-Shannon divergence

## Results

- Models tested: Dream-7B, LLaDA-8B, LLaDA-1.5-8B
- 7 benchmarks spanning general knowledge, math, and code
- **72.1% fewer reverse steps** than fixed-budget decoding
- Improves average accuracy over strong training-free adaptive samplers
- Reductions translate to lower wall-clock latency and inference compute
