# 06 · Instruction fine-tuning: becoming an assistant — *Rafael*

**In:** pretrained GPT-2 124M (04). **Out:** a model that responds to instructions.

Questions:
1. How does a model that only "continues text" become an assistant?
2. Where can I get fine-tuning data? How do I build it from scratch?

> 📓 Page generated from the notebook [`06_assistente.ipynb`](https://github.com/victorflll/LLMs-from-scratch/blob/main/30-09/06_assistente.ipynb) (*LLMs-from-scratch* repository, `30-09` folder), with the outputs of its last run. To run it, see the [class setup](setup.md).

```python
import json  # read/save the instruction datasets (JSON format)
import os  # build folder paths
import time  # measure training time
from functools import partial  # creates a version of a function with some arguments pre-filled

import tiktoken  # GPT-2 tokenizer: text ↔ IDs
import torch  # PyTorch
from torch.utils.data import DataLoader  # groups the examples into batches

from aula import REPO_DIR, CHECKPOINTS, device, load_gpt2  # paths, device (mps/cuda/cpu) and the GPT-2 loader
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text, train_model_simple  # text generation and the SAME training loop as pretraining
from llms_from_scratch.ch07 import format_input, InstructionDataset, custom_collate_fn  # Alpaca template, instruction dataset and batch assembly

CARREGAR_CHECKPOINT = True  # True = use the saved assistant (if it exists)
N_EXEMPLOS_TREINO = 400      # subset so it fits live (the book uses ~935 and GPT-2 355M)
bpe = tiktoken.get_encoding("gpt2")  # GPT-2 tokenizer
```

## 1. The data: instruction → response pairs

```python
with open(os.path.join(REPO_DIR, "ch07", "01_main-chapter-code", "instruction-data.json")) as f:  # dataset from chapter 7 of the book
    dados = json.load(f)  # reads the content and turns it into a list of dicts: instruction (request), input (optional), output (correct answer)
print(len(dados), "exemplos\n")  # total number of examples
print(json.dumps(dados[50], indent=2, ensure_ascii=False))  # shows one formatted example
```

```text
1100 exemplos

{
  "instruction": "Identify the correct spelling of the following word.",
  "input": "Ocassion",
  "output": "The correct spelling is 'Occasion.'"
}
```

We format it in the **Alpaca** style — the model learns the "template" and where the response begins:

**Goal:** turn the example into text the model can read, because the model doesn't read dictionaries.

```python
print(format_input(dados[50]) + f"\n\n### Response:\n{dados[50]['output']}")  # Alpaca-template prompt + response = one complete training example
```

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

**Goal:** split the examples into three groups: one to learn from, one to monitor training, and one to
test at the end.

Why hold out a test group? To measure the model on examples it never saw during training;
otherwise it might have just memorized them.

```python
n_tr, n_te = int(len(dados) * 0.85), int(len(dados) * 0.10)  # 85% train (935), 10% test (110)
treino, teste, val = dados[:n_tr], dados[n_tr:n_tr + n_te], dados[n_tr + n_te:]  # slices in order; the rest (5%) is validation
treino = treino[:N_EXEMPLOS_TREINO]  # uses only the first 400 so training is fast
print(len(treino), "treino |", len(val), "validação |", len(teste), "teste")  # checks the sizes
```

```text
400 treino | 55 validação | 110 teste
```

## 2. Before fine-tuning

```python
gpt2, cfg = load_gpt2("124M")  # pretrained GPT-2, still without instruction fine-tuning
gpt2.to(device)  # moves the model to GPU/MPS (or CPU)

# asks the model a question and returns only the response text
def responder(model, entrada, max_new_tokens=60):
    prompt = format_input(entrada)  # Alpaca-template prompt, without the response (the model has to write it)
    ids = generate(model, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=max_new_tokens,
                   context_size=cfg["context_length"], eos_id=50256)  # generates up to 60 tokens; stops if it emits <|endoftext|> (50256)
    return token_ids_to_text(ids, bpe)[len(prompt):].replace("### Response:", "").strip()  # IDs → text, without the prompt and the marker

for e in teste[:3]:  # 3 test examples BEFORE fine-tuning
    print(f"▶ {e['instruction']} {e['input']}\n  modelo: {responder(gpt2, e)!r}\n")
```

```text
▶ Rewrite the sentence using a simile. The car is very fast.
  modelo: '### Output:\n\nThe car is very fast.\n\n### Error:\n\nThe car is very fast.\n\n### Error:\n\nThe car is very fast.\n\n### Error:\n\nThe car is very fast.\n\n### Error:\n\nThe'

▶ What type of cloud is typically associated with thunderstorms? 
  modelo: 'Thunderstorms are the most common type of thunderstorm. They are usually caused by lightning, thunderstorms, or other natural phenomena.\n\nThunderstorms are caused by a combination of natural phenomena, such as lightning, thunderstorms, or other natural phenomena.\n\nThunderstorms are caused by'

▶ Name the author of 'Pride and Prejudice'. 
  modelo: "### Description:\n\nThe author of 'Pride and Prejudice' is a young man who has been a member of the Church for over a century. He is a member of the Church of England and has been a member of the Church for over a century. He is a"
```

## 3. Fine-tuning (or checkpoint)
It is the **same loop as pretraining** (`train_model_simple`) — only the data changed.
Saved training run (without checkpoint): loss 3.02 → 0.53 in 100 steps (~1 min); the model starts answering after `### Response:` and stopping at `<|endoftext|>`.

```python
ckpt = os.path.join(CHECKPOINTS, "assistente_124M.pth")  # file with the already fine-tuned model
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):  # if a checkpoint was requested and the file exists...
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))  # ...loads the fine-tuned weights and skips training
    print("checkpoint carregado:", ckpt)
else:  # otherwise, train now (~1 min)
    torch.manual_seed(123)  # reproducible training
    t0 = time.time()  # marks the start
    # training batches: pads the examples with 50256 to the same length; in the target, the extra padding becomes -100 (ignored by the loss)
    collate = partial(custom_collate_fn, device=device, allowed_max_length=1024)  # fixes the device and GPT-2's context limit
    train_loader = DataLoader(InstructionDataset(treino, bpe), batch_size=8, collate_fn=collate, shuffle=True, drop_last=True)  # 400 examples → 50 batches of 8, shuffled
    val_loader = DataLoader(InstructionDataset(val, bpe), batch_size=8, collate_fn=collate)  # validation batches (measure the error during training)
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)  # tunes ALL weights, with small steps (lr) so as not to ruin the pretraining
    train_model_simple(gpt2, train_loader, val_loader, optimizer, device, num_epochs=2,
                       eval_freq=10, eval_iter=5, start_context=format_input(val[0]), tokenizer=bpe)  # same loop as pretraining: 2 epochs, loss every 10 steps
    print(f"{(time.time() - t0) / 60:.1f} min")  # training time
    # bfloat16 = half the size (~330 MB); load_state_dict converts back to float32
    torch.save({k: v.to(torch.bfloat16) for k, v in gpt2.state_dict().items()}, ckpt + ".part")  # saves to a temporary file first
    os.replace(ckpt + ".part", ckpt)  # renames only at the end (avoids a corrupted file if the process stops)
```

```text
checkpoint carregado: .../LLMs-from-scratch/30-09/checkpoints/assistente_124M.pth
```

## 4. After fine-tuning

```python
gpt2.eval()  # evaluation mode (turns off dropout)
torch.manual_seed(123)  # reproducibility (generation here is already deterministic: always the most likely token)
for e in teste[:6]:  # 6 test examples AFTER fine-tuning, with the expected answer alongside
    print(f"▶ {e['instruction']} {e['input']}\n  esperado: {e['output']}\n  modelo:   {responder(gpt2, e)}\n")
```

```text
▶ Rewrite the sentence using a simile. The car is very fast.
  esperado: The car is as fast as lightning.
  modelo:   The car is very fast.

▶ What type of cloud is typically associated with thunderstorms? 
  esperado: The type of cloud typically associated with thunderstorms is cumulonimbus.
  modelo:   A type of cloud is typically associated with thunderstorms.

▶ Name the author of 'Pride and Prejudice'. 
  esperado: Jane Austen.
  modelo:   The author of 'Pride and Prejudice' is William Shakespeare.

▶ What is the periodic symbol for chlorine? 
  esperado: The periodic symbol for chlorine is Cl.
  modelo:   The periodic symbol for chlorine is CH3.

▶ Correct the punctuation in the sentence. Its time to go home.
  esperado: The corrected sentence should be: 'It's time to go home.'
  modelo:   The time to go home is 8:00 am.

▶ Rewrite the sentence. The lecture was delivered in a clear manner.
  esperado: The lecture was delivered clearly.
  modelo:   The lecture was delivered in a clear manner.
```

With 124M and few examples it learns the **format** (answers and stops at `<|endoftext|>`), but gets facts wrong — "Pride and Prejudice" became Shakespeare. Knowledge comes from size/pretraining, not from fine-tuning; the book uses 355M and the full dataset
(and evaluates the answers with another LLM — `ch07/03_model-evaluation`).

## 5. Where to get data? How to build it from scratch?
**Ready-made:** Alpaca (52k, generated with an LLM), Dolly 15k (written by humans, commercial license), OpenAssistant (conversations),
FLAN, and thousands on the Hugging Face Hub (`datasets`). In Portuguese: translations of Alpaca/Dolly and community datasets.

**From scratch:** it's just a list of `instruction / input / output` dictionaries:

```python
meus_dados = [  # your own dataset: same instruction / input / output format
    {"instruction": "Traduza para o inglês.", "input": "Bom dia, turma!", "output": "Good morning, class!"},
    {"instruction": "Classifique o sentimento.", "input": "Adorei a aula de hoje.", "output": "Positivo"},
    {"instruction": "O que é um token?", "input": "", "output": "Um pedaço de texto (palavra, subpalavra ou byte) que o modelo processa."},
]
with open(os.path.join(CHECKPOINTS, "meus_dados.json"), "w", encoding="utf-8") as f:  # creates the JSON file
    json.dump(meus_dados, f, ensure_ascii=False, indent=2)  # saves keeping the accents
print(format_input(meus_dados[2]))  # empty input: the "### Input:" section doesn't appear in the prompt
print(len(InstructionDataset(meus_dados, bpe)), "exemplos prontos para o mesmo DataLoader")  # goes into the same training pipeline, unchanged
```

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
O que é um token?
3 exemplos prontos para o mesmo DataLoader
```

Sources for building your own: anonymized customer-support/FAQ logs, internal documentation, answers written by experts,
or **synthetic data** generated by a larger LLM and reviewed by humans (`ch07/05_dataset-generation`).
Quality > quantity: the LIMA paper showed good results with ~1,000 well-curated examples — on top of a 65B base model (the knowledge came from pretraining).
After that comes preference alignment (RLHF/DPO — `ch07/04_preference-tuning-with-dpo`).

### ✅ Built
Pretrained GPT-2 + instruction data + the same training loop = a (mini) assistant.
