---
title: "KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing"
tags: [KV cache, transformer, LLM inference, attention]
year: 2026
tldr: "KVEraser is a learned KV-cache editing method that efficiently removes spans from a processed context without full recomputation. Given a context and a span to erase, it replaces only the KV states of the erased interval with learned steering states while reusing the remaining cache. A two-stage training pipeline (generic span-neighbor pretraining + task-specific finetuning) enables transferable erasing. Nearly matches full recomputation in post-erasure quality across 1K-32K context lengths, with latency increasing only 24% vs 17.6x for recomputation."
wikilinks: [[KV cache]] [[LLM inference]] [[transformer]] [[attention]]
---
# KVEraser

**Authors**: arXiv:2606.17034
**Year**: 2026

KVEraser addresses the challenge of post-hoc context erasing over the [[KV cache]]. When a span needs to be removed from an already-processed context, exact erasing requires recomputing all subsequent tokens — cost proportional to the suffix length, not the erased span.

## Method

KVEraser replaces only the KV states of the erased interval with learned **steering states** while reusing the remaining cache unchanged. A two-stage training pipeline is used:

1. **Generic span-neighbor pretraining**: Teaches the eraser to suppress the influence of the erased span
2. **Task-specific fine-tuning**: Adapts to downstream scenarios

## Results

- Nearly matches full recomputation in post-erasure performance
- Context lengths: 1K–32K
- Latency increases by only 24% vs 17.6x for full recomputation
- Generalizes to unseen long-document QA with harmful factual distractors
- Achieves 3–4x speedup over full recomputation
