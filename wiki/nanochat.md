---
title: "nanochat"
tags: [transformer, LLM inference, scaling, language model, neural network, deep learning]
year: 2026
tldr: "nanochat is the successor to nanoGPT by Andrej Karpathy — a minimal, hackable experimental harness for training LLMs on a single GPU node. Covers tokenization, pretraining, finetuning, evaluation, inference, and chat UI. Trains GPT-2 capability models for ~$48 (2 hours on 8×H100). Features automated hyperparameter selection via a single `--depth` dial, compute-optimal scaling, and a leaderboard tracking time-to-GPT-2."
wikilinks: [[nanoGPT]] [[GPT-2]] [[Scaling_Laws]] [[tokenization]]
---
# nanochat

nanochat is a minimal experimental harness for training LLMs from scratch, created by Andrej Karpathy as the successor to [[nanoGPT]]. It is designed for single-GPU-node training and covers all major LLM stages: tokenization, pretraining, finetuning, evaluation, inference, and a ChatGPT-like web UI.

Key features:
- **Single-dial complexity**: A single `--depth` parameter (number of transformer layers) automatically determines all other hyperparameters (width, heads, learning rate, training horizon, weight decay) for compute-optimal training.
- **GPT-2 speedrun leaderboard**: Tracks wall-clock time to match GPT-2 (1.6B) capability on 8×H100 nodes. Current record: ~1.65 hours (down from OpenAI's original 168 hours/$43K in 2019).
- **Cost efficiency**: Can train GPT-2-capability models for ~$48 on 8×H100 GPUs, or ~$15 on spot instances.
- **Full pipeline**: Includes tokenization (via uv), pretraining, finetuning, core metric evaluation (DCLM CORE), and chat UI.

The repo uses PyTorch and the `uv` package manager, and supports CUDA and CPU/MPS backends.

Source: https://github.com/karpathy/nanochat
