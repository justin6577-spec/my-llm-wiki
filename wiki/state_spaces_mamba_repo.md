---
title: "state-spaces/mamba – Official Mamba / Mamba-2 / Mamba-3 Repository"
authors:
  - Albert Gu
  - Tri Dao
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - deep learning
  - language model
  - hardware acceleration
  - github
tldr: >
  The official GitHub repository for the Mamba family of selective state space models,
  covering Mamba-1 (selective SSM with hardware-aware scan), Mamba-2 (SSD: structured
  state space duality with tensor-core matmuls), and Mamba-3 (MIMO, inference-first SSM).
  Provides pip-installable CUDA kernels, pretrained checkpoints up to 2.8B parameters,
  and lm-evaluation-harness integration.
wikilinks:
  - "[[Mamba]]"
  - "[[Hardware-Aware Scan]]"
  - "[[FlashAttention-2]]"
  - "[[Chunkwise recurrent]]"
  - "[[Diagonal recurrence]]"
  - "[[Sequence parallelism]]"
  - "[[Memory hierarchy]]"
  - "[[HBM]]"
  - "[[Inference optimization]]"
---

# state-spaces/mamba – Official Mamba Repository

**URL:** [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)  
**Authors:** Albert Gu (CMU), Tri Dao (Princeton)  
**Papers:**
- Mamba-1: [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- Mamba-2: [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- Mamba-3: [arXiv 2603.15569](https://arxiv.org/abs/2603.15569)

---

## Repository Overview

This is the **official implementation** of the Mamba family of models, including three generations of selective state space models. All models are available as `pip install mamba-ssm`.

---

## Mamba-1

- **Core:** Selective SSM (S6) with input-dependent $(A, B, C)$ parameters  
- **Algorithm:** Hardware-aware parallel associative scan (see [[Hardware-Aware Scan]])  
- **State size:** $N = 16$ (limited by scan efficiency)  
- **Training:** Scan-based, no tensor cores  

```python
from mamba_ssm import Mamba
model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
```

---

## Mamba-2 (SSD)

- **Core:** SSD layer — scalar-times-identity $A_t$, multi-head structure  
- **Algorithm:** Chunkwise matmul + short scan (see [[Chunkwise recurrent]])  
- **State size:** $N = 64$–$128$ (tensor-core friendly)  
- **Training:** 2–6× faster than Mamba-1  

Key implementation files:
- `mamba_ssm/modules/mamba2.py` — full Mamba-2 block  
- `mamba_ssm/modules/mamba2_simple.py` — simplified version  
- `mamba_ssm/modules/ssd_minimal.py` — minimal SSD (Listing 1 from paper)  

```python
from mamba_ssm import Mamba2
model = Mamba2(d_model=16, d_state=64, d_conv=4, expand=2).cuda()
```

---

## Mamba-3

- **Core:** MIMO (multiple-input multiple-output) SSM for improved expressivity  
- **Focus:** Inference-first design, improved long-range recall  
- **Parameters:** `is_mimo=True`, `mimo_rank`, `chunk_size`  

```python
from mamba_ssm import Mamba3
model = Mamba3(d_model=768, d_state=128, headdim=64, is_mimo=True,
               mimo_rank=4, chunk_size=16, dtype=torch.bfloat16).cuda()
```

---

## Pretrained Models (HuggingFace: `state-spaces`)

| Model | Params | Tokens | Notes |
|---|---|---|---|
| `mamba-130m` .. `mamba-2.8b` | 130M–2.8B | 300B | Mamba-1, Pile |
| `mamba2-130m` .. `mamba2-2.7b` | 130M–2.7B | 300B | Mamba-2 / SSD |
| `transformerpp-2.7b` | 2.7B | 300B | Transformer++ baseline |
| `mamba2attn-2.7b` | 2.7B | 300B | Mamba-2 + attention hybrid |
| `mamba-2.8b-slimpj` | 2.8B | 600B | SlimPajama dataset |

Model dimensions follow GPT-3 conventions (24–64 layers, 768–2560 hidden dim).

---

## Hardware Requirements

- Linux, NVIDIA GPU (CUDA 11.6+), PyTorch 1.12+
- AMD GPU support available (see repo docs)
- Optional: `causal-conv1d` for efficient Conv1d in Mamba block

---

## Evaluation

Zero-shot evaluation via `lm-evaluation-harness`:

```sh
lm_eval --model mamba_ssm \
        --model_args pretrained=state-spaces/mamba2-2.7b \
        --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
        --device cuda --batch_size 256
```

---

## Related Wiki Notes

- [[Mamba]] — the foundational selective SSM paper  
- [[Hardware-Aware Scan]] — Mamba-1's training algorithm  
- [[Chunkwise recurrent]] — Mamba-2's SSD training algorithm  
- [[Diagonal recurrence]] — scalar-$a_t$ recurrence in SSD  
- [[FlashAttention-2]] — systems inspiration (IO-aware design)  
- [[Sequence parallelism]] — enabled for SSMs by SSD  
- [[Inference optimization]] — recurrent form for O(1) step inference  
- [[HBM]] / [[Memory hierarchy]] — GPU memory design principles  
