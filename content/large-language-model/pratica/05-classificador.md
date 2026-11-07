# 05 · Fine-tuning para classificação (spam) — *Elmo*

**Entra:** GPT-2 124M pré-treinado (04). **Sai:** um classificador spam / não-spam.

Ideia central: trocamos a cabeça de saída (50.257 tokens → **2 classes**) e treinamos pouca coisa por cima do que o GPT já sabe.

> 📓 Página gerada a partir do notebook [`05_classificador.ipynb`](https://github.com/victorflll/LLMs-from-scratch/blob/main/30-09/05_classificador.ipynb) (repositório *LLMs-from-scratch*, pasta `30-09`), com as saídas da última execução. Para rodar, veja o [setup da aula](setup.md).

```python
import os                                  # caminhos de arquivo; checar se o checkpoint já existe
import time                                # medir quanto tempo o treino leva
from pathlib import Path                   # caminhos de arquivo multiplataforma

import pandas as pd                        # ler/tratar o CSV de SMS como tabela (DataFrame)
import tiktoken                            # tokenizador BPE do GPT-2 (o mesmo do notebook 01)
import torch
from torch.utils.data import DataLoader    # agrupa o dataset em lotes (batches)

from aula import DATA, CHECKPOINTS, device, load_gpt2, n_params   # caminhos de dados/checkpoints, dispositivo (cpu/cuda/mps), carregar GPT-2, contar parâmetros
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text  # geração de texto e conversão texto <-> IDs (capítulo 5)
from llms_from_scratch.ch06 import (download_and_unzip_spam_data, create_balanced_dataset, random_split,
                                    SpamDataset, calc_accuracy_loader, train_classifier_simple, classify_review)  # tudo do classificador (capítulo 6)

# True = usa o classificador salvo (se existir) em vez de treinar ao vivo
CARREGAR_CHECKPOINT = True                 # troque para False para treinar do zero 
N_EPOCAS = 5                               # quantas vezes o treino passa pelo conjunto inteiro
bpe = tiktoken.get_encoding("gpt2")        # mesmo tokenizador do notebook 01
```

## 1. Dados: SMS Spam Collection (UCI)

```python
tsv = Path(DATA) / "SMSSpamCollection.tsv"    # onde o arquivo baixado/descompactado vai ficar
download_and_unzip_spam_data("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                             Path(DATA) / "sms_spam_collection.zip", DATA, tsv)  # baixa e descompacta só se o .tsv ainda não existir
df = pd.read_csv(tsv, sep="\t", header=None, names=["Label", "Text"])   # lê o TSV: colunas Label (ham/spam) e Text (a mensagem)
print(df["Label"].value_counts(), "\n")    # quantas mensagens de cada classe (dataset desbalanceado: ~13% spam)

df = create_balanced_dataset(df)  # mesmo número de spam e ham (sorteia "ham" até igualar à quantidade de spam)
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})   # rótulo texto -> número (0 = não-spam, 1 = spam)
treino, val, teste = random_split(df, 0.7, 0.1)         # embaralha e divide: 70% treino, 10% validação, 20% teste
for nome, parte in [("train", treino), ("validation", val), ("test", teste)]:
    parte.to_csv(Path(DATA) / f"{nome}.csv", index=None)   # salva cada parte em CSV (a próxima célula lê esses arquivos)
df.sample(5, random_state=1)   # mostra 5 linhas aleatórias só para conferir visualmente
```

```text
.../LLMs-from-scratch/30-09/data/SMSSpamCollection.tsv already exists. Skipping download and extraction.
Label
ham     4825
spam     747
Name: count, dtype: int64 

      Label                                               Text
733       0  Lol you won't feel bad when I use her money to...
952       0                           Shb b ok lor... Thanx...
3443      1  Save money on wedding lingerie at www.bridal.p...
3415      0                            No pic. Please re-send.
1466      1  YOU 07801543489 are guaranteed the latests Nok...
```

```python
train_ds = SpamDataset(Path(DATA) / "train.csv", bpe)  # padding até a maior mensagem (define o max_length usado no treino)
val_ds = SpamDataset(Path(DATA) / "validation.csv", bpe, max_length=train_ds.max_length)  # mesmo max_length do treino, para o formato bater
test_ds = SpamDataset(Path(DATA) / "test.csv", bpe, max_length=train_ds.max_length)       # tambem para o teste
torch.manual_seed(123)                          # fixa a semente (afeta o embaralhamento do DataLoader de treino)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)   # embaralha a cada época; descarta o último lote se ficar incompleto
val_loader = DataLoader(val_ds, batch_size=8)   # validação: não precisa embaralhar
test_loader = DataLoader(test_ds, batch_size=8) # teste: não precisa embaralhar
print(len(train_ds), "treino |", len(val_ds), "validação |", len(test_ds), "teste | max_length =", train_ds.max_length)
```

```text
1045 treino | 149 validação | 300 teste | max_length = 120
```

## 2. Dá para classificar só com o GPT-2 pré-treinado, via prompt?

```python
gpt2, cfg = load_gpt2("124M")            # carrega o GPT-2 124M com pesos oficiais + sua config (emb_dim, context_length, ...)
gpt2.to(device)                          # move o modelo para a GPU/MPS (ou CPU)
prompt = ("Is the following text 'spam'? Answer with 'yes' or 'no': "
          "'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'")  # pergunta direta, sem treinar nada ainda
out = generate(gpt2, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=20, context_size=cfg["context_length"])  # gera 20 tokens novos, um de cada vez
print(token_ids_to_text(out, bpe)[len(prompt):])   # converte de volta para texto e corta a parte do prompt, sobra só o que o modelo respondeu
```

```text


The following text 'spam'? Answer with 'yes' or 'no': 'You
```

Não: o modelo base só **continua texto**. Ele nunca foi ensinado a seguir instruções (isso é o notebook 06).

## 3. Adaptando a arquitetura
1. congela tudo  2. troca `out_head` por `Linear(768, 2)`  3. libera o último bloco e o LayerNorm final.

```python
for p in gpt2.parameters():
    p.requires_grad = False              # congela TODOS os pesos: nada vai ser atualizado no treino (preserva o que o GPT-2 já sabe)
torch.manual_seed(123)                   # fixa a semente antes de criar a camada nova (pesos iniciais aleatórios, mas reprodutíveis)
gpt2.out_head = torch.nn.Linear(cfg["emb_dim"], 2).to(device)   # troca a cabeça: de 50.257 tokens para 2 classes (spam/não-spam); já nasce treinável
for p in list(gpt2.trf_blocks[-1].parameters()) + list(gpt2.final_norm.parameters()):
    p.requires_grad = True               # libera para treino só o último bloco Transformer (o 12º) e a normalização final
print(f"treináveis: {n_params(gpt2, treinaveis=True):,} de {n_params(gpt2):,}")   # quantos parâmetros vão ser ajustados vs. total do modelo
```

```text
treináveis: 7,090,946 de 124,441,346
```

```python
# Quanto treinar? Contagem de parâmetros treináveis em cada estratégia
blocos = gpt2.trf_blocks
por_bloco = sum(p.numel() for p in blocos[0].parameters())    # nº de parâmetros de UM bloco Transformer (todos têm o mesmo tamanho)
for nome, n in [("só a cabeça", 768 * 2 + 2), ("cabeça + último bloco", 768 * 2 + 2 + por_bloco),
                ("tudo (full fine-tuning)", n_params(gpt2))]:   # compara 3 estratégias possíveis de fine-tuning (só conta, não treina)
    print(f"{nome:25s} {n:>12,}")
```

```text
só a cabeça                      1,538
cabeça + último bloco        7,089,410
tudo (full fine-tuning)    124,441,346
```

Usamos só o **último token** da saída: pela máscara causal, ele é o único que "viu" a mensagem inteira.

```python
with torch.no_grad():   # só estamos olhando a saída, não precisamos calcular gradiente aqui
    print("saída do modelo:", tuple(gpt2(next(iter(train_loader))[0].to(device)).shape), "→ usamos [:, -1, :]")  # roda 1 lote: (8,120) IDs -> (8,120,2) notas por posição
torch.manual_seed(123)
print(f"acurácia ANTES do treino — val: {calc_accuracy_loader(val_loader, gpt2, device, num_batches=10):.0%}")  # cabeça ainda aleatória: deve dar ~chute (perto de 50%)
```

```text
saída do modelo: (8, 120, 2) → usamos [:, -1, :]
acurácia ANTES do treino — val: 45%
```

## 4. Treino (ou checkpoint)

```python
ckpt = os.path.join(CHECKPOINTS, "classificador_spam.pth")   # caminho do arquivo de checkpoint salvo
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)  # strict=False: o arquivo só tem os pesos treinados; o resto já está carregado (GPT-2 original)
    print("checkpoint carregado:", ckpt)
else:
    torch.manual_seed(123)
    t0 = time.time()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)   # só os parâmetros com requires_grad=True são de fato atualizados
    train_classifier_simple(gpt2, train_loader, val_loader, optimizer, device,
                            num_epochs=N_EPOCAS, eval_freq=50, eval_iter=5)   # loop de treino completo (forward, loss, backward, step) por N_EPOCAS épocas
    print(f"{(time.time() - t0) / 60:.1f} min")
    # salva só o que foi treinado (~28 MB); o resto é o GPT-2 original
    torch.save({n: p.detach() for n, p in gpt2.named_parameters() if p.requires_grad}, ckpt + ".part")   # grava primeiro num arquivo temporário
    os.replace(ckpt + ".part", ckpt)   # troca atômica: evita deixar um checkpoint corrompido se o processo for interrompido
```

```text
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
Ep 2 (Step 000150): Train loss 0.561, Val loss 0.489
Ep 2 (Step 000200): Train loss 0.419, Val loss 0.397
Ep 2 (Step 000250): Train loss 0.409, Val loss 0.353
Training accuracy: 82.50% | Validation accuracy: 85.00%
Ep 3 (Step 000300): Train loss 0.333, Val loss 0.320
Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306
Training accuracy: 90.00% | Validation accuracy: 90.00%
Ep 4 (Step 000400): Train loss 0.136, Val loss 0.200
Ep 4 (Step 000450): Train loss 0.153, Val loss 0.132
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000550): Train loss 0.207, Val loss 0.143
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
2.0 min
```

```python
for nome, loader in [("treino", train_loader), ("validação", val_loader), ("teste", test_loader)]:
    print(f"acurácia {nome:10s}: {calc_accuracy_loader(loader, gpt2, device):.1%}")   # acurácia final, em TODOS os lotes de cada conjunto (sem amostragem)
```

```text
acurácia treino    : 97.2%
acurácia validação : 97.3%
acurácia teste     : 95.7%
```

## 5. Testando com mensagens da turma

```python
mensagens = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",  # spam óbvio (mesma mensagem do teste via prompt na ETAPA 4)
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",                 # mensagem normal (ham)
    "URGENT! Your account has been suspended. Click here to verify your password now.",             # spam (nem estava no dataset original)
    "Can you send me the slides from class before Friday?",                                          # mensagem normal (ham)
]
for m in mensagens:
    print(f"{classify_review(m, gpt2, bpe, device, max_length=train_ds.max_length):9s} ← {m}")   # tokeniza, roda o modelo e classifica cada mensagem nova
```

```text
spam      ← You are a winner you have been specially selected to receive $1000 cash or a $2000 award.
not spam  ← Hey, just wanted to check if we're still on for dinner tonight? Let me know!
spam      ← URGENT! Your account has been suspended. Click here to verify your password now.
not spam  ← Can you send me the slides from class before Friday?
```

### ✅ Construto
Um GPT-2 com cabeça de 2 classes. O mesmo truque serve para sentimento, tópicos, intenção… basta trocar o nº de saídas.
Mais experimentos (treinar tudo, usar o primeiro token, LoRA): `ch06/02_bonus_additional-experiments`.
