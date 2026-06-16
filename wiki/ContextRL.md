---
title: "ContextRL: Context-Aware RL for Agentic and Multimodal LLMs"
tags: [reinforcement learning, RLHF, transformer, language model, LLM inference]
year: 2026
tldr: "ContextRL is a context-aware reinforcement learning method that improves long-horizon reasoning and multimodal performance through an indirect auxiliary objective. Instead of supervising only the final answer, it rewards the model for selecting the correct context supporting a query-answer pair. Achieves +2.2% over standard GRPO on 5 long-horizon benchmarks and +1.8% across 12 VQA benchmarks. Gains come from the objective design, not just additional data."
wikilinks: [[RLHF]] [[Reinforcement Learning]] [[GRPO]] [[transformer]] [[language model]]
---
# ContextRL

**Authors**: arXiv:2606.17053
**Year**: 2026

ContextRL addresses LLM failures in identifying small but decisive evidence within long or complex contexts.

## Method

A context-aware [[reinforcement learning]] (RL) method with an **indirect auxiliary objective**:

- Presents the model with a query, an answer, and two highly similar contexts
- Rewards the model for selecting the context that supports the query-answer pair
- Encourages fine-grained grounding without direct answer supervision

**Data construction**:
- Coding agents: 1K trajectory pairs via condition filtering
- Multimodal reasoning: 7K image pairs via generative editing and similarity search

## Results

- **+2.2%** over standard [[GRPO]] on 5 long-horizon benchmarks
- **+1.8%** across 12 diverse visual question answering benchmarks
- Ablations show gains arise from the context-selection objective, not from additional data alone
