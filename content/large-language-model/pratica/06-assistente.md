# 06 · Fine-tuning de instruções: virando assistente — *Rafael*

**Entra:** GPT-2 124M pré-treinado (04). **Sai:** um modelo que responde instruções.

Perguntas:
1. Como um modelo que só "continua texto" vira um assistente?
2. Onde eu consigo dados de fine-tuning? Como construo do zero?

> 📓 Página gerada a partir do notebook [`06_assistente.ipynb`](https://github.com/victorflll/LLMs-from-scratch/blob/main/30-09/06_assistente.ipynb) (repositório *LLMs-from-scratch*, pasta `30-09`), com as saídas da última execução. Para rodar, veja o [setup da aula](setup.md).

```python
import json  # ler/salvar os datasets de instruções (formato JSON)
import os  # montar caminhos de pastas
import time  # medir o tempo de treino
from functools import partial  # cria uma versão da função com argumentos já preenchidos

import tiktoken  # tokenizador do GPT-2: texto ↔ IDs
import torch  # PyTorch
from torch.utils.data import DataLoader  # agrupa os exemplos em lotes (batches)

from aula import REPO_DIR, CHECKPOINTS, device, load_gpt2  # caminhos, dispositivo (mps/cuda/cpu) e carregador do GPT-2
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text, train_model_simple  # geração de texto e o MESMO loop de treino do pré-treino
from llms_from_scratch.ch07 import format_input, InstructionDataset, custom_collate_fn  # molde Alpaca, dataset de instruções e montagem dos lotes

CARREGAR_CHECKPOINT = True  # True = usa o assistente salvo (se existir)
N_EXEMPLOS_TREINO = 400      # subconjunto para caber ao vivo (o livro usa ~935 e o GPT-2 355M)
bpe = tiktoken.get_encoding("gpt2")  # tokenizador do GPT-2
```

## 1. Os dados: pares instrução → resposta

```python
with open(os.path.join(REPO_DIR, "ch07", "01_main-chapter-code", "instruction-data.json")) as f:  # dataset do cap. 7 do livro
    dados = json.load(f)  # lê o conteúdo e transforma numa lista de dicionários: instruction (pedido), input (opcional), output (resposta certa)
print(len(dados), "exemplos\n")  # total de exemplos
print(json.dumps(dados[50], indent=2, ensure_ascii=False))  # mostra um exemplo formatado
```

```text
1100 exemplos

{
  "instruction": "Identify the correct spelling of the following word.",
  "input": "Ocassion",
  "output": "The correct spelling is 'Occasion.'"
}
```

Formatamos no estilo **Alpaca** — o modelo aprende o "molde" e onde a resposta começa:

**Objetivo:** transformar o exemplo num texto que o modelo consiga ler, porque o modelo não lê dicionário.

```python
print(format_input(dados[50]) + f"\n\n### Response:\n{dados[50]['output']}")  # prompt no molde Alpaca + resposta = um exemplo completo de treino
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

**Objetivo:** separar os exemplos em três grupos: um para aprender, um para acompanhar o treino e um para
testar no final.

Por que separar um grupo de teste? Para medir o modelo com exemplos que ele nunca viu no treino;
senão ele poderia só ter decorado.

```python
n_tr, n_te = int(len(dados) * 0.85), int(len(dados) * 0.10)  # 85% treino (935), 10% teste (110)
treino, teste, val = dados[:n_tr], dados[n_tr:n_tr + n_te], dados[n_tr + n_te:]  # fatia na ordem; o resto (5%) é validação
treino = treino[:N_EXEMPLOS_TREINO]  # usa só os primeiros 400 para o treino ser rápido
print(len(treino), "treino |", len(val), "validação |", len(teste), "teste")  # confere os tamanhos
```

```text
400 treino | 55 validação | 110 teste
```

## 2. Antes do fine-tuning

```python
gpt2, cfg = load_gpt2("124M")  # GPT-2 pré-treinado, ainda sem fine-tuning de instruções
gpt2.to(device)  # move o modelo para GPU/MPS (ou CPU)

# faz uma pergunta ao modelo e devolve só o texto da resposta
def responder(model, entrada, max_new_tokens=60):
    prompt = format_input(entrada)  # prompt no molde Alpaca, sem a resposta (o modelo tem que escrever)
    ids = generate(model, text_to_token_ids(prompt, bpe).to(device), max_new_tokens=max_new_tokens,
                   context_size=cfg["context_length"], eos_id=50256)  # gera até 60 tokens; para se sair <|endoftext|> (50256)
    return token_ids_to_text(ids, bpe)[len(prompt):].replace("### Response:", "").strip()  # IDs → texto, sem o prompt e o marcador

for e in teste[:3]:  # 3 exemplos de teste ANTES do fine-tuning
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

## 3. Fine-tuning (ou checkpoint)
É o **mesmo loop do pré-treino** (`train_model_simple`) — só mudaram os dados.
Treino salvo (sem checkpoint): loss 3,02 → 0,53 em 100 passos (~1 min); o modelo passa a responder após `### Response:` e a parar no `<|endoftext|>`.

```python
ckpt = os.path.join(CHECKPOINTS, "assistente_124M.pth")  # arquivo com o modelo já ajustado
if CARREGAR_CHECKPOINT and os.path.exists(ckpt):  # se pediu checkpoint e o arquivo existe...
    gpt2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))  # ...carrega os pesos ajustados e pula o treino
    print("checkpoint carregado:", ckpt)
else:  # senão, treina agora (~1 min)
    torch.manual_seed(123)  # treino reproduzível
    t0 = time.time()  # marca o início
    # lotes de treino: completa os exemplos com 50256 até o mesmo tamanho; no alvo, o preenchimento extra vira -100 (ignorado na loss)
    collate = partial(custom_collate_fn, device=device, allowed_max_length=1024)  # fixa o dispositivo e o limite de contexto do GPT-2
    train_loader = DataLoader(InstructionDataset(treino, bpe), batch_size=8, collate_fn=collate, shuffle=True, drop_last=True)  # 400 exemplos → 50 lotes de 8, embaralhados
    val_loader = DataLoader(InstructionDataset(val, bpe), batch_size=8, collate_fn=collate)  # lotes de validação (medem o erro durante o treino)
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.1)  # ajusta TODOS os pesos, com passos pequenos (lr) para não estragar o pré-treino
    train_model_simple(gpt2, train_loader, val_loader, optimizer, device, num_epochs=2,
                       eval_freq=10, eval_iter=5, start_context=format_input(val[0]), tokenizer=bpe)  # mesmo loop do pré-treino: 2 épocas, loss a cada 10 passos
    print(f"{(time.time() - t0) / 60:.1f} min")  # tempo de treino
    # bfloat16 = metade do tamanho (~330 MB); load_state_dict converte de volta para float32
    torch.save({k: v.to(torch.bfloat16) for k, v in gpt2.state_dict().items()}, ckpt + ".part")  # salva primeiro num arquivo temporário
    os.replace(ckpt + ".part", ckpt)  # renomeia só no fim (evita arquivo corrompido se o processo parar)
```

```text
checkpoint carregado: .../LLMs-from-scratch/30-09/checkpoints/assistente_124M.pth
```

## 4. Depois do fine-tuning

```python
gpt2.eval()  # modo de avaliação (desliga o dropout)
torch.manual_seed(123)  # reprodutibilidade (a geração aqui já é determinística: sempre o token mais provável)
for e in teste[:6]:  # 6 exemplos de teste DEPOIS do fine-tuning, com a resposta esperada ao lado
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

Com 124M e poucos exemplos ele aprende o **formato** (responde e para no `<|endoftext|>`), mas erra fatos — "Pride and Prejudice" virou Shakespeare. Conhecimento vem do tamanho/pré-treino, não do fine-tuning; o livro usa 355M e o dataset inteiro
(e avalia as respostas com outro LLM — `ch07/03_model-evaluation`).

## 5. Onde conseguir dados? Como construir do zero?
**Prontos:** Alpaca (52k, gerado com LLM), Dolly 15k (escrito por humanos, licença comercial), OpenAssistant (conversas),
FLAN, e milhares no Hugging Face Hub (`datasets`). Em português: traduções do Alpaca/Dolly e datasets da comunidade.

**Do zero:** é só uma lista de dicionários `instruction / input / output`:

```python
meus_dados = [  # dataset próprio: mesmo formato instruction / input / output
    {"instruction": "Traduza para o inglês.", "input": "Bom dia, turma!", "output": "Good morning, class!"},
    {"instruction": "Classifique o sentimento.", "input": "Adorei a aula de hoje.", "output": "Positivo"},
    {"instruction": "O que é um token?", "input": "", "output": "Um pedaço de texto (palavra, subpalavra ou byte) que o modelo processa."},
]
with open(os.path.join(CHECKPOINTS, "meus_dados.json"), "w", encoding="utf-8") as f:  # cria o arquivo JSON
    json.dump(meus_dados, f, ensure_ascii=False, indent=2)  # salva mantendo os acentos
print(format_input(meus_dados[2]))  # input vazio: a seção "### Input:" não aparece no prompt
print(len(InstructionDataset(meus_dados, bpe)), "exemplos prontos para o mesmo DataLoader")  # entra no mesmo pipeline de treino, sem mudanças
```

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
O que é um token?
3 exemplos prontos para o mesmo DataLoader
```

Fontes para montar o seu: logs de atendimento/FAQ anonimizados, documentação interna, respostas escritas por especialistas,
ou **dados sintéticos** gerados por um LLM maior e revisados por humanos (`ch07/05_dataset-generation`).
Qualidade > quantidade: o paper LIMA mostrou bons resultados com ~1.000 exemplos bem curados — sobre um modelo base de 65B (o conhecimento veio do pré-treino).
Depois disso vem o alinhamento por preferência (RLHF/DPO — `ch07/04_preference-tuning-with-dpo`).

### ✅ Construto
GPT-2 pré-treinado + dados de instrução + o mesmo loop de treino = um (mini) assistente.
