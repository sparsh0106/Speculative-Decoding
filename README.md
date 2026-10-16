# ⚡ Speculative Decoding — Fast Transformer Inference from Scratch

An algorithmic, from-scratch PyTorch implementation of **Speculative Decoding**, based on the ICML 2023 paper:
> **Fast Inference from Transformers via Speculative Decoding**  
> *Yaniv Leviathan, Matan Kalman, Yossi Matias (ICML 2023)*

The goal of this project is to implement and benchmark autoregressive and speculative decoding at the tensor/logit level rather than relying on high-level generation APIs.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `setup.ipynb` | **Notebook 01 — Setup & Model Inference**: Environment configuration, shared tokenizer setup, loading the Target Model (`HuggingFaceTB/SmolLM2-1.7B`) and Draft Model (`HuggingFaceTB/SmolLM2-360M`), parameter comparison, and manual single-step forward passes. |
| `Vanilla_Autoregressive_Decoding.ipynb` | **Notebook 02 — Vanilla Autoregressive Decoding**: From-scratch greedy autoregressive decoding both **without KV-cache** and **with KV-cache**, verifying output equivalence and benchmarking forward-pass latency as a baseline for speculative decoding. |
| `README.md` | Project overview and notebook breakdown. |

---

## 🔬 Model Configuration

- **Target Model**: `HuggingFaceTB/SmolLM2-1.7B` (`float16` on CUDA)
- **Draft Model**: `HuggingFaceTB/SmolLM2-360M` (`float16` on CUDA)
- **Speculative Lookahead ($k$)**: `4` draft tokens per target verification step
- **Shared Vocabulary**: Both models share the SmolLM2 tokenizer family, allowing direct token-ID and probability comparison without re-tokenization.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers numpy tqdm jupyter
```

### Running the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/sparsh0106/Speculative-Decoding.git
   cd Speculative-Decoding
   ```
2. Run `setup.ipynb` followed by `Vanilla_Autoregressive_Decoding.ipynb` in Jupyter Lab / Notebook with a CUDA-enabled GPU.

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106) · [LinkedIn](https://www.linkedin.com/in/sparsh-agarwal-4b443b304/)
