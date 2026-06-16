---
title: "The Value Axis: Language Models Encode Whether They're on the Right Track"
tags: [RLHF, reinforcement learning, transformer, language model, LLM inference]
year: 2026
tldr: "Investigates whether LMs internally track the 'value' of their current trajectory — the likelihood of achieving their goals. Using synthetic in-context RL data, constructs a value axis for Qwen3-8B. Activations along this axis distinguish high vs low confidence, correct vs corrupted code, rollouts with vs without backtracking. Steering toward high value suppresses self-correction; steering toward low induces exploration. DPO can increase internal value of rewarded behaviors."
wikilinks: [[RLHF]] [[Reinforcement Learning]] [[DPO]] [[transformer]] [[language model]]
---
# The Value Axis

**Authors**: arXiv:2606.17056
**Year**: 2026

This paper investigates whether language models internally encode a "value" estimate of their ongoing trajectory — the likelihood that their current strategy will achieve its goals.

## Key Findings

Using synthetic in-context [[reinforcement learning]] data with Qwen3-8B:

- Activations along the value axis distinguish between:
  - High vs low verbalized confidence
  - Rollouts without vs with backtracking
  - Correct vs corrupted code
- **Steering toward high value**: Suppresses self-correction, reduces explanatory verbosity
- **Steering toward low value**: Induces backtracking and exploration
- [[DPO]] (Direct Preference Optimization) can increase the internal value of rewarded behaviors

## Applications

- Qwen assigns low value to politically sensitive chat queries after post-training
- Supervised fine-tuning increases internal confidence within the training domain
- Suggests LMs linearly encode an estimate of expected goal success that modulates confidence
