# Class setup · Building GPT-2 in practice

A class taught entirely in notebooks. Each notebook builds one piece (the **construct**) used by the next, and answers the group's questions by running code.

| # | Notebook | Presenter | Construct | Questions answered | Time* |
|---|---|---|---|---|---|
| 1 | [01 · Tokenization](01-tokenization.md) | Elmo | tokenizer + dataloader + embeddings | BPE vs character/word/cl100k/o200k · our own BPE from scratch · embedding size · visualizing weights | 10 s |
| 2 | [02 · Attention](02-attention.md) | Rafael | `MultiHeadAttention` | other scores (additive, cosine) · number of heads · non-divisible case · heatmaps of GPT-2's 12 heads | 5 s |
| 3 | [03 · GPT and generation](03-gpt-and-generation.md) | Victor | `GPTModel` + generation | a block step by step · text without pretraining · without LayerNorm/FFN, ReLU · 12 vs 24 layers | 15 s |
| 4 | [04 · Pretraining](04-pretraining.md) | Lilian | trained model + GPT-2 weights | loss · temperature/top-k · ablations **with training** · alternatives to the Transformer | 2 min |
| 5 | [05 · Classifier](05-classifier.md) | Elmo | spam classifier | classification fine-tuning (≈96% on the test set) | 2.5 min / 5 s with checkpoint |
| 6 | [06 · Assistant](06-assistant.md) | Rafael | GPT that follows instructions | instruction fine-tuning · where to find / how to create data | 2 min / 15 s with checkpoint |
| 7 | [07 · Parameters](07-parameters.md) | Christiane | — | why these values · small→XL · ×3 · GPT-3 · bias and gradient · Llama/GPT-4/Claude | 15 s |

\* MacBook M1 Pro (MPS). Cells marked with 🔧 are parameter experiments.

## How to run

The notebooks live in [victorflll/LLMs-from-scratch](https://github.com/victorflll/LLMs-from-scratch/tree/main/30-09), in the `30-09` folder. The notebooks themselves (code, prints and comments) are in Portuguese.

```bash
git clone https://github.com/victorflll/LLMs-from-scratch.git
cd LLMs-from-scratch/30-09
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Dependencies (`requirements.txt`): `torch`, `tiktoken`, `numpy`, `pandas`, `matplotlib`, `requests`, `tqdm`, `jupyter`.

The model is **not rewritten** in the notebooks: everything comes from `../pkg/llms_from_scratch` (the book's code). `aula.py` only holds the *device*, the configs, the GPT-2 weight download (Hugging Face, no TensorFlow), and `variante()` for the ablations.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 50,000 BPE merges + 256 bytes + <|endoftext|>
    "context_length": 1024,  # maximum number of tokens the model sees
    "emb_dim": 768,          # size of each token vector
    "n_heads": 12,           # attention heads (768 / 12 = 64 per head)
    "n_layers": 12,          # stacked Transformer blocks
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

## Before class

Run the notebooks in order once. This downloads the weights (`checkpoints/`, ~700 MB) and the spam dataset (`data/`), and saves the fine-tuning checkpoints. With `CARREGAR_CHECKPOINT = True` (the default), notebooks 05 and 06 load those checkpoints; set it to `False` to train live.

> The pages in this section reproduce each notebook with the outputs of its last run, so you can read them without running anything.
