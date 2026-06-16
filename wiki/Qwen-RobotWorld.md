---
title: "Qwen-RobotWorld: Unifying Embodied World Modeling through Language-Conditioned Video Generation"
tags: [transformer, attention, diffusion, language model]
year: 2026
tldr: "Qwen-RobotWorld is a language-conditioned video world model using a 60-layer double-stream diffusion transformer (MMDiT) with MLLM action encoding. Uses a frozen Qwen2.5-VL coupled with video-VAE latents through layer-wise joint attention. Trained on 8.6M video-text corpus (200M+ frames) across 20+ embodiments and 500+ action categories. Ranks 1st on EWMBench and DreamGen Bench."
wikilinks: [[transformer]] [[attention]] [[diffusion]] [[Scaling_Laws]]
---
# Qwen-RobotWorld

**Authors**: arXiv:2606.17030
**Year**: 2026

Qwen-RobotWorld is a language-conditioned video world model for embodied intelligence, using natural language as a unified action interface.

## Architecture

- **Double-Stream MMDiT**: 60-layer double-stream diffusion transformer
- **MLLM Action Encoding**: Couples frozen Qwen2.5-VL semantics with video-VAE latents through layer-wise joint attention
- **Embodied World Knowledge (EWK)**: 8.6M video-text corpus (200M+ frames), 20+ embodiments, 500+ action categories
- **General+Expert Progressive Curriculum**: Two-stage training (general visual priors → embodied specialization)

## Results

- Ranks **1st overall** on EWMBench and DreamGen Bench
- Outperforms all open-source models on WorldModelBench and PBench
- Zero-shot generalization on RoboTwin-IF benchmark
