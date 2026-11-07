# 05 · Fine-tuning for classification (spam) — *Elmo*

**In:** pretrained GPT-2 124M (04). **Out:** a spam / not-spam classifier.

Core idea: we swap the output head (50,257 tokens → **2 classes**) and train only a little on top of what GPT already knows.

> 📓 Page generated from the notebook [`05_classificador.ipynb`](https://github.com/victorflll/LLMs-from-scratch/blob/main/30-09/05_classificador.ipynb) (*LLMs-from-scratch* repository, `30-09` folder), with the outputs of its last run. To run it, see the [class setup](setup.md).

```python
import os                                  # file paths; check whether the checkpoint already exists
import time                                # measure how long training takes
from pathlib import Path                   # cross-platform file paths

import pandas as pd                        # read/process the SMS CSV as a table (DataFrame)
import tiktoken                            # GPT-2 BPE tokenizer (the same as in notebook 01)
import torch
from torch.utils.data import DataLoader    # groups the dataset into batches

from aula import DATA, CHECKPOINTS, device, load_gpt2, n_params   # data/checkpoint paths, device (cpu/cuda/mps), load GPT-2, count parameters
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text  # text generation and text <-> IDs conversion (chapter 5)
from llms_from_scratch.ch06 import (download_and_unzip_spam_data, create_balanced_dataset, random_split,
                                    SpamDataset, calc_accuracy_loader, train_classifier_simple, classify_review)  # everything for the classifier (chapter 6)

# True = use the saved classifier (if it exists) instead of training live
CARREGAR_CHECKPOINT = True                 # switch to False to train from scratch 
N_EPOCAS = 5                               # how many times training passes over the whole set
bpe = tiktoken.get_encoding("gpt2")        # same tokenizer as notebook 01
```

## 1. Data: SMS Spam Collection (UCI)

```python
tsv = Path(DATA) / "SMSSpamCollection.tsv"    # where the downloaded/unzipped file will live
download_and_unzip_spam_data("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                             Path(DATA) / "sms_spam_collection.zip", DATA, tsv)  # downloads and unzips only if the .tsv doesn't exist yet
df = pd.read_csv(tsv, sep="\t", header=None, names=["Label", "Text"])   # reads the TSV: columns Label (ham/spam) and Text (the message)
print(df["Label"].value_counts(), "\n")    # how many messages of each class (imbalanced dataset: ~13% spam)

df = create_balanced_dataset(df)  # same number of spam and ham (samples "ham" until it matches the spam count)
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})   # text label -> number (0 = not spam, 1 = spam)
treino, val, teste = random_split(df, 0.7, 0.1)         # shuffles and splits: 70% train, 10% validation, 20% test
for nome, parte in [("train", treino), ("validation", val), ("test", teste)]:
    parte.to_csv(Path(DATA) / f"{nome}.csv", index=None)   # saves each split as CSV (the next cell reads these files)
df.sample(5, random_state=1)   # shows 5 random rows just for a visual check
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
train_ds = SpamDataset(Path(DATA) / "train.csv", bpe)  # padding up to the longest message (defines the max_length used in training)
val_ds = SpamDataset(Path(DATA) / "validation.csv", bpe, max_length=train_ds.max_length)  # same max_length as training, so the shapes match
test_ds = SpamDataset(Path(DATA) / "test.csv", bpe, max_length=train_ds.max_length)       # likewise for the test set
torch.manual_seed(123)                          # fixes the seed (affects the shuffling of the training DataLoader)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)   # shuffles every epoch; drops the last batch if incomplete
val_loader = DataLoader(val_ds, batch_size=8)   # validation: no need to shuffle
test_loader = DataLoader(test_ds, batch_size=8) # test: no need to shuffle
print(len(train_ds), "treino |", len(val_ds), "validação |", len(test_ds), "teste | max_length =", train_ds.max_length)
```

```text
1045 treino | 149 validação | 300 teste | max_length = 120
```

## 2. Can we classify with the pretrained GPT-2 alone, via a prompt?

```python
gpt2, cfg = load_gpt2("124M")            # loads GPT-2 124M with the official weights + its config (emb_dim, context_length, ...)
gpt2.to(device)                          # moves the model to the GPU/MPS (or CPU)
prompt = ("Is the following text 'spam'? Answer with 'yes' or 'no': "
          "'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'")  # direct question, nothing trained yet
out = generate(gpt2, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=20, context_size=cfg["context_length"])  # generates 20 new tokens, one at a time
print(token_ids_to_text(out, bpe)[len(prompt):])   # converts back to text and cuts off the prompt, leaving only the model's answer
```

```text


The following text 'spam'? Answer with 'yes' or 'no': 'You
```

No: the base model only **continues text**. It was never taught to follow instructions (that is notebook 06).

## 3. Adapting the architecture
1. freeze everything  2. swap `out_head` for `Linear(768, 2)`  3. unfreeze the last block and the final LayerNorm.

```python
for p in gpt2.parameters():
    p.requires_grad = False              # freezes ALL weights: nothing will be updated in training (preserves what GPT-2 already knows)
torch.manual_seed(123)                   # fixes the seed before creating the new layer (random but reproducible initial weights)
gpt2.out_head = torch.nn.Linear(cfg["emb_dim"], 2).to(device)   # swaps the head: from 50,257 tokens to 2 classes (spam/not spam); trainable from the start
for p in list(gpt2.trf_blocks[-1].parameters()) + list(gpt2.final_norm.parameters()):
    p.requires_grad = True               # unfreezes for training only the last Transformer block (the 12th) and the final normalization
print(f"treináveis: {n_params(gpt2, treinaveis=True):,} de {n_params(gpt2):,}")   # how many parameters will be tuned vs. the model total
```

```text
treináveis: 7,090,946 de 124,441,346
```

```python
# How much to train? Count of trainable parameters for each strategy
blocos = gpt2.trf_blocks
por_bloco = sum(p.numel() for p in blocos[0].parameters())    # no. of parameters in ONE Transformer block (they are all the same size)
for nome, n in [("só a cabeça", 768 * 2 + 2), ("cabeça + último bloco", 768 * 2 + 2 + por_bloco),
                ("tudo (full fine-tuning)", n_params(gpt2))]:   # compares 3 possible fine-tuning strategies (only counts, doesn't train)
    print(f"{nome:25s} {n:>12,}")
```

```text
só a cabeça                      1,538
cabeça + último bloco        7,089,410
tudo (full fine-tuning)    124,441,346
```

We use only the **last token** of the output: because of the causal mask, it is the only one that has "seen" the whole message.

```python
with torch.no_grad():   # we're only looking at the output, no need to compute gradients here
    print("saída do modelo:", tuple(gpt2(next(iter(train_loader))[0].to(device)).shape), "→ usamos [:, -1, :]")  # runs 1 batch: (8,120) IDs -> (8,120,2) scores per position
torch.manual_seed(123)
print(f"acurácia ANTES do treino — val: {calc_accuracy_loader(val_loader, gpt2, device, num_batches=10):.0%}")  # head is still random: should be ~a guess (close to 50%)
```

```text
saída do modelo: (8, 120, 2) → usamos [:, -1, :]
acurácia ANTES do treino — val: 45%
```

## 4. Training (or checkpoint)

```python
ckpt = os.path.join(CHECKPOINTS, "classificador_spam.pth")   # path of the saved checkpoint file
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)  # strict=False: the file holds only the trained weights; the rest is already loaded (original GPT-2)
    print("checkpoint carregado:", ckpt)
else:
    torch.manual_seed(123)
    t0 = time.time()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)   # only parameters with requires_grad=True actually get updated
    train_classifier_simple(gpt2, train_loader, val_loader, optimizer, device,
                            num_epochs=N_EPOCAS, eval_freq=50, eval_iter=5)   # full training loop (forward, loss, backward, step) for N_EPOCAS epochs
    print(f"{(time.time() - t0) / 60:.1f} min")
    # saves only what was trained (~28 MB); the rest is the original GPT-2
    torch.save({n: p.detach() for n, p in gpt2.named_parameters() if p.requires_grad}, ckpt + ".part")   # writes to a temporary file first
    os.replace(ckpt + ".part", ckpt)   # atomic swap: avoids leaving a corrupted checkpoint if the process is interrupted
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
    print(f"acurácia {nome:10s}: {calc_accuracy_loader(loader, gpt2, device):.1%}")   # final accuracy, over ALL batches of each set (no sampling)
```

```text
acurácia treino    : 97.2%
acurácia validação : 97.3%
acurácia teste     : 95.7%
```

## 5. Testing with messages from the class

```python
mensagens = [
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",  # obvious spam (same message as the prompt test in STEP 4)
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",                 # normal message (ham)
    "URGENT! Your account has been suspended. Click here to verify your password now.",             # spam (wasn't even in the original dataset)
    "Can you send me the slides from class before Friday?",                                          # normal message (ham)
]
for m in mensagens:
    print(f"{classify_review(m, gpt2, bpe, device, max_length=train_ds.max_length):9s} ← {m}")   # tokenizes, runs the model and classifies each new message
```

```text
spam      ← You are a winner you have been specially selected to receive $1000 cash or a $2000 award.
not spam  ← Hey, just wanted to check if we're still on for dinner tonight? Let me know!
spam      ← URGENT! Your account has been suspended. Click here to verify your password now.
not spam  ← Can you send me the slides from class before Friday?
```

### ✅ Built
A GPT-2 with a 2-class head. The same trick works for sentiment, topics, intent… just change the number of outputs.
More experiments (training everything, using the first token, LoRA): `ch06/02_bonus_additional-experiments`.
