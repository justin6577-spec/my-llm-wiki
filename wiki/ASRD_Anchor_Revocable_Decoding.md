---
title: "ASRD: Anchor Supervised Revocable Decoding for Diffusion LLMs"
tags: [transformer, LLM inference, attention, language model]
year: 2026
tldr: "ASRD is a training-free framework for revocable decoding in diffusion LLMs that operates in embedding space. Decouples context into trusted Anchor Tokens (identified via temporal consistency) and uncertain candidates. Uses Anchor-Guided Generation (injects entropy-weighted anchor signals) and Anchor-Perturbed Verification (applies orthogonal perturbations to destabilize errors). Achieves accuracy improvements up to 6.4% and inference throughput up to 7.2x on math and coding benchmarks."
wikilinks: [[transformer]] [[LLM inference]] [[attention]] [[speculative decoding]]
---
# ASRD: Anchor Supervised Revocable Decoding

**Authors**: arXiv:2606.16847
**Year**: 2026

ASRD addresses error propagation and local error reinforcement in revocable decoding strategies for Diffusion LLMs (dLLMs).

## Method

Operates in **embedding space** with two key mechanisms:

1. **Anchor-Guided Generation**: Injects entropy-weighted anchor signals into masked positions to implicitly rectify attention toward the reliable global skeleton
2. **Anchor-Perturbed Verification**: Applies orthogonal perturbations to uncertain candidate tokens, destabilizing errors driven by fragile local consensus

Uses a dynamic **Anchor Tokens Cache** identified via temporal consistency.

## Results

- Outperforms recent remasking baselines on math and coding benchmarks
- Accuracy improvements up to **6.4%**
- Inference throughput improvement up to **7.2×**
