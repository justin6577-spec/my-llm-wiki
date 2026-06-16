---
title: "T-Rex: Tactile-Reactive Dexterous Manipulation with Mixture-of-Transformers"
tags: [transformer, mixture of experts, MoE, attention, neural network]
year: 2026
tldr: "T-Rex introduces a variable-rate Mixture-of-Transformers (MoT) architecture for tactile-reactive robotic manipulation. The MoT architecture uses a temporal tactile VQ-VAE encoder to process high-frequency touch signals without sacrificing existing VLA capabilities. Achieves over 30% higher average success rate than strongest baselines on 12 manipulation tasks requiring delicate force control."
wikilinks: [[mixture of experts]] [[MoE]] [[transformer]] [[attention]]
---
# T-Rex: Mixture-of-Transformers

**Authors**: arXiv:2606.17055
**Year**: 2026

T-Rex introduces a **variable-rate Mixture-of-Transformers (MoT)** architecture for tactile-reactive robotic manipulation. This is the first work to effectively integrate high-frequency tactile signals into Vision-Language-Action (VLA) models.

## Key Contributions

- Large-scale 100-hour tactile-rich dataset
- **Variable-rate MoT architecture** with temporal tactile VQ-VAE encoder
- Processes high-frequency touch signals without sacrificing existing VLA capabilities

## Results

- +30% higher average success rate over strongest baseline
- 12 manipulation tasks requiring delicate force control and deformable object manipulation
