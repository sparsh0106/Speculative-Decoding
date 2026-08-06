# **Speculative Decoding**
---

## `setup.ipynb` - Notebook 01: Setup & Model Inference

This notebook is the first part of a from-scratch implementation of **Speculative Decoding**, based on:
> **Fast Inference from Transformers via Speculative Decoding**
> Yaniv Leviathan, Matan Kalman, Yossi Matias - ICML 2023

The objective of this project is to understand and implement speculative decoding at the algorithmic level rather than relying directly on a pre-built speculative generation API.


### Objective of Notebook 01

Before implementing speculative decoding, we first need to understand the basic inference pipeline of an autoregressive Transformer.

This notebook establishes that foundation by:

- Setting up the PyTorch and Hugging Face environment
- Configuring GPU inference
- Loading a shared tokenizer
- Loading the **Draft Model**
- Loading the **Target Model**
- Comparing their parameter counts and configurations
- Tokenizing an input prompt
- Performing a manual Transformer forward pass
- Inspecting the output logits
- Performing greedy next-token prediction
- Inspecting the model's KV cache (`past_key_values`)

No speculative decoding is performed yet.

#### Models

We use two models from the **SmolLM2** family:

| Role | Model | Parameters |
|---|---|---:|
| Draft Model | `HuggingFaceTB/SmolLM2-360M` | ~360M |
| Target Model | `HuggingFaceTB/SmolLM2-1.7B` | ~1.7B |

The **draft model** will eventually generate inexpensive speculative token proposals.

The **target model** represents the model whose output distribution we want to preserve.

Using models from the same family also allows us to work with a compatible tokenizer and vocabulary.

#### Why Two Models?

Standard autoregressive generation works sequentially:

```text
Prompt
  ↓
Target Model
  ↓
Token 1
  ↓
Target Model
  ↓
Token 2
  ↓
Target Model
  ↓
Token 3
  ↓
