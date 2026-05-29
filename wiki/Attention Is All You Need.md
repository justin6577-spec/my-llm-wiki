---
title: Attention Is All You Need
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
year: 2017
arxiv: 1706.03762
tags: [foundational, attention, transformer, sequence-modeling, machine-translation]
citation_count: 0
tldr: Replace RNNs entirely with self-attention to get a parallelizable sequence model that hits 28.4 BLEU on WMT 2014 En→De, beating the previous best ensemble by over 2 BLEU.
---

## The Problem

For years, the gold standard for sequence-to-sequence tasks like machine translation was the RNN — specifically LSTMs and GRUs wrapped in an encoder-decoder architecture. These models process tokens one at a time: the hidden state at step *t* depends on the hidden state at step *t-1*. This is fundamentally sequential. You can't compute step 5 until step 4 is done, step 4 until step 3, and so on. At training time, this kills GPU utilization because the whole batch has to march through time in lockstep.

The sequential nature also creates a depth problem for long sequences. To relate two tokens that are far apart — say, a pronoun and the noun it refers to 50 tokens earlier — the signal has to flow through 50 recurrent steps. Each step is a potential bottleneck for gradient flow and information compression. Convolutional alternatives like ByteNet and ConvS2S tried to parallelize computation, but relating distant positions still required O(n) or O(log n) operations in sequence depth, not O(1).

Attention mechanisms existed and were already being bolted onto RNNs to help them handle long-range dependencies. But they were always a supplement to recurrence, not a replacement. Nobody had asked: what if attention is *all* you need?

## The Idea

**Throw out recurrence entirely and build the whole model out of attention.** If attention can directly connect any two positions in a sequence in a single operation, you get O(1) path length between any two tokens *and* you can compute all positions in parallel. You get the best of both worlds: long-range dependencies handled cleanly, and full GPU parallelism during training.

The intuition: think of each token as a person in a room full of other tokens. In an RNN, information travels by whisper — token 1 tells token 2, who tells token 3, etc. By the time the message reaches token 50, it's garbled. With attention, every token can directly "look at" every other token simultaneously — like everyone in the room broadcasting on a loudspeaker, and each person selectively tuning in to whoever is relevant to them.

## How It Works

**Scaled Dot-Product Attention** is the core primitive. Given a set of queries Q, keys K, and values V:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The query is "what am I looking for?" The keys are "what does each token advertise?" The dot product measures compatibility. The softmax turns these scores into a probability distribution, and the output is a weighted sum of the values. Dividing by √d_k prevents the dot products from blowing up in magnitude when d_k is large — without scaling, the softmax would get pushed into regions of near-zero gradient, killing learning.

**Multi-Head Attention** runs h=8 attention heads in parallel, each operating in a lower-dimensional subspace (d_k = d_v = 64, d_model/h). Each head learns to attend to different kinds of relationships — syntax in one head, coreference in another, local context in a third. Their outputs are concatenated and projected back to d_model=512:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

**The full Transformer** is an encoder-decoder stack. Both encoder and decoder are composed of N=6 identical layers.
- **Encoder layer**: Multi-head self-attention → Add & LayerNorm → Position-wise FFN → Add & LayerNorm
- **Decoder layer**: Masked multi-head self-attention (can't peek at future tokens) → Add & LayerNorm → Cross-attention over encoder output → Add & LayerNorm → FFN → Add & LayerNorm

**Positional Encodings** are added to the input embeddings to inject sequence order information, since attention itself is permutation-invariant. The paper uses sinusoidal functions of different frequencies, allowing the model to generalize to sequence lengths unseen during training.

The FFN inside each layer is a simple two-layer MLP with ReLU applied to each position independently: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂, with inner dimension 2048.

## Key Results

- **WMT 2014 English→German**: 28.4 BLEU — more than 2 BLEU above the previous best result, including ensembles, achieved in 3.5 days on 8 P100 GPUs.
- **WMT 2014 English→French**: 41.8 BLEU — new single-model state of the art, at a fraction of the training cost of prior best models.
- **Training speed**: The base Transformer can be trained in ~12 hours on 8 P100 GPUs — previously competitive models took days or weeks.
- **English constituency parsing**: The model generalizes well beyond translation, achieving strong results both with abundant and limited training data.

## Limitations

- **Quadratic attention cost**: Self-attention computes all pairwise interactions between tokens, so memory and compute scale as O(n²) in sequence length. For very long sequences (documents, high-resolution images, genomics), this becomes prohibitive. This spawned an entire research cottage industry of efficient attention variants.
- **No inherent notion of sequence order**: The model is permutation-invariant and must be given positional encodings explicitly. Sinusoidal encodings work but are somewhat ad hoc; learning better positional representations remained an open problem.
- **Autoregressive inference is still sequential**: At generation time, the decoder still produces tokens one at a time (each output depends on previous outputs), so the parallelism advantage of training doesn't fully carry over to inference latency.
- **Fixed context window**: The architecture as presented doesn't handle sequences longer than the maximum length seen during training without modification.

## Why It Matters

This paper is arguably the most influential single paper in modern AI. It didn't just propose a better sequence model — it proposed a general-purpose differentiable computation primitive that turned out to scale with data and parameters in ways RNNs never could. The Transformer became the backbone of BERT, GPT, T5, and every major language model since. It also migrated far beyond NLP: Vision Transformer (ViT) brought it to images, AlphaFold brought it to protein structure, and diffusion transformers brought it to image generation.

The key unlock was parallelism during training. Because every position is processed simultaneously, you can throw massive GPU clusters at Transformer training in a way that was structurally impossible with RNNs. This is what enabled the scaling laws era — the observation that performance improves predictably with more compute, data, and parameters. None of that scaling story happens without a parallelizable architecture.

It also demonstrated that strong inductive biases (local connectivity in CNNs, sequential processing in RNNs) might actually be limiting. A relatively agnostic architecture that learns its own structure from data, given enough of it, can surpass hand-crafted priors.

## See Also

[[BERT]] · [[GPT]] · [[Vision Transformer]] · [[Flash Attention]] · [[Scaled Dot-Product Attention]] · [[Positional Encoding]] · [[Layer Normalization]] · [[Mixture of Experts]]