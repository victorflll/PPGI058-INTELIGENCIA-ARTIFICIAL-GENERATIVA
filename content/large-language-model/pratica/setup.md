# Setup da aula · Construindo o GPT-2 na prática

Aula 100% em notebooks. Cada notebook constrói uma peça (o **construto**) usada no seguinte e responde às perguntas do grupo rodando código.

| # | Notebook | Quem | Construto | Perguntas respondidas | Tempo* |
|---|---|---|---|---|---|
| 1 | [01 · Tokenização](01-tokenizacao.md) | Elmo | tokenizer + dataloader + embeddings | BPE vs caractere/palavra/cl100k/o200k · BPE próprio do zero · tamanho do embedding · visualizar pesos | 10 s |
| 2 | [02 · Atenção](02-atencao.md) | Rafael | `MultiHeadAttention` | outros scores (aditivo, cosseno) · nº de heads · não divisível · heatmaps das 12 heads do GPT-2 | 5 s |
| 3 | [03 · GPT e geração](03-gpt-e-geracao.md) | Victor | `GPTModel` + geração | bloco passo a passo · texto sem pré-treino · sem LayerNorm/FFN, ReLU · 12 vs 24 camadas | 15 s |
| 4 | [04 · Pré-treino](04-pre-treino.md) | Lilian | modelo treinado + pesos GPT-2 | loss · temperatura/top-k · ablações **com treino** · alternativas ao Transformer | 2 min |
| 5 | [05 · Classificador](05-classificador.md) | Elmo | classificador de spam | fine-tuning de classificação (≈96% no teste) | 2,5 min / 5 s com checkpoint |
| 6 | [06 · Assistente](06-assistente.md) | Rafael | GPT que segue instruções | fine-tuning de instruções · onde achar / como criar dados | 2 min / 15 s com checkpoint |
| 7 | [07 · Parâmetros](07-parametros.md) | Christiane | — | por que esses valores · small→XL · ×3 · GPT-3 · viés e gradiente · Llama/GPT-4/Claude | 15 s |

\* MacBook M1 Pro (MPS). Células marcadas com 🔧 são experimentos de parâmetros.

## Como rodar

Os notebooks estão no repositório [victorflll/LLMs-from-scratch](https://github.com/victorflll/LLMs-from-scratch/tree/main/30-09), na pasta `30-09`:

```bash
git clone https://github.com/victorflll/LLMs-from-scratch.git
cd LLMs-from-scratch/30-09
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Dependências (`requirements.txt`): `torch`, `tiktoken`, `numpy`, `pandas`, `matplotlib`, `requests`, `tqdm`, `jupyter`.

O modelo **não é reescrito** nos notebooks: tudo vem de `../pkg/llms_from_scratch` (o código do livro). O arquivo `aula.py` só tem o *device*, as configurações, o download dos pesos do GPT-2 (Hugging Face, sem TensorFlow) e `variante()` para as ablações.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 50.000 merges do BPE + 256 bytes + <|endoftext|>
    "context_length": 1024,  # máximo de tokens que o modelo enxerga
    "emb_dim": 768,          # tamanho de cada vetor de token
    "n_heads": 12,           # cabeças de atenção (768 / 12 = 64 por cabeça)
    "n_layers": 12,          # blocos Transformer empilhados
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

## Antes da aula

Rode os notebooks em ordem uma vez. Isso baixa os pesos (`checkpoints/`, ~700 MB) e o dataset de spam (`data/`), e salva os checkpoints dos fine-tunings. Com `CARREGAR_CHECKPOINT = True` (padrão), os notebooks 05 e 06 carregam esses checkpoints; mude para `False` para treinar ao vivo.

> As páginas desta seção reproduzem cada notebook com as saídas da última execução, para leitura sem precisar rodar nada.
