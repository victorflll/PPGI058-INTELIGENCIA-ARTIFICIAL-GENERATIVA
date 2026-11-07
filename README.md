# IA Generativa · wiki da turma

Wiki da disciplina **Inteligência Artificial Generativa** (PPGI058 · IC / UFAL · 2026.2).
Cada grupo tem uma seção própria no menu lateral e a alimenta depois da sua apresentação:

| # | Assunto | Pasta | Situação |
|---|---|---|---|
| 1 | Large Language Model | `content/large-language-model/` | publicado |
| 2 | Small Language Model | `content/small-language-model/` | aguardando o grupo |
| 3 | LLM em Produção | `content/llm-em-producao/` | aguardando o grupo |
| 4 | Agentic Coding | `content/agentic-coding/` | aguardando o grupo |
| 5 | Generative Deep Learning | `content/generative-deep-learning/` | aguardando o grupo |

A seção de **Large Language Model** segue o livro *Build a Large Language Model (From
Scratch)*, de Sebastian Raschka: teoria dos capítulos 1 a 3 (a partir dos slides) e a aula
prática com os notebooks de
[victorflll/LLMs-from-scratch](https://github.com/victorflll/LLMs-from-scratch/tree/main/30-09).

É um site **sem build**: o navegador lê `content/structure.json`, que descreve a árvore de
páginas, e renderiza o Markdown na hora. Não há servidor, framework nem compilação, só
HTML, CSS e um arquivo JavaScript, por isso roda direto no **GitHub Pages**.

## Como funciona

- **`content/structure.json`** define a árvore de seções (sidebar, páginas, navegação).
- Cada nó da árvore é uma **página**:
  - nó **com `markdown`** renderiza aquele arquivo (página de conteúdo);
  - nó **com `children`** vira uma página de seção com a descrição e *cards* para as subpáginas.
- A navegação usa o **hash** da URL (`#/large-language-model/pratica/atencao`), então recarregar a página e links
  diretos funcionam sem configurar servidor.

O Markdown é renderizado com títulos, listas, tabelas (com rolagem horizontal no celular),
código com realce de sintaxe, diagramas **Mermaid**, imagens e um **visualizador de PDF
inline** (todo link `.pdf` vira um card que abre o PDF na página).

## Estrutura

```
.
├── index.html              # casca da página (sidebar, topo, libs via CDN)
├── .nojekyll               # desativa o Jekyll no GitHub Pages
├── assets/
│   ├── css/styles.css      # tema obsidian glass (tokens de design em :root)
│   ├── js/app.js           # roteador + renderizador (fetch → marked → highlight)
│   └── images/             # favicons, logo
└── content/
    ├── structure.json      # << a árvore de páginas do curso
    ├── large-language-model/   # grupo 1
    │   ├── teoria/             # caps. 1–3 + figuras/ + pdfs/ (slides)
    │   ├── pratica/            # setup + 7 notebooks convertidos + figuras/
    │   └── equipe/             # integrantes do grupo
    └── <assunto-do-grupo>/     # grupos 2–5 criam a sua pasta aqui
```

## Como o seu grupo publica o conteúdo

A seção do seu grupo já existe em `content/structure.json`, mas ainda sem páginas (por isso
aparece "Conteúdo em breve"). Para publicar:

1. Crie as páginas Markdown na pasta do grupo, ex. `content/agentic-coding/visao-geral.md`,
   com imagens e PDFs ao lado (`figuras/`, `pdfs/`).
2. Em `content/structure.json`, adicione as páginas como `children` do nó do seu grupo,
   na ordem de leitura. Use a seção **Large Language Model** como modelo (subseções
   Teoria / Prática / Equipe do grupo).
3. Não mexa nas seções dos outros grupos; prefira um pull request por grupo.
4. Rode o site localmente e confira cada página antes de abrir o PR.

## Editando o conteúdo

1. Escreva/edite o Markdown em `content/<assunto>/…`.
2. Coloque imagens e PDFs ao lado do `.md` e referencie com caminho relativo, ex.:
   `![](figuras/diagrama.png)`, `[Slides](pdfs/slides.pdf)`.
3. Registre a página em `content/structure.json` com `name`, `slug`, `description` e
   `markdown` (página de conteúdo) **ou** `children` (página de seção).
4. Use *slugs* em `kebab-case`, sem acentos, únicos entre irmãos.

## Rodando localmente

O navegador bloqueia `fetch` via `file://`, então sirva a pasta por HTTP:

```bash
python3 -m http.server 8000
# abra http://localhost:8000
```

## Publicação

É um site estático puro: basta servir o conteúdo do repositório no GitHub Pages. Não há
nada para compilar.

> Detalhes de arquitetura e convenções de manutenção (inclusive para agentes de IA) estão
> em [CLAUDE.md](CLAUDE.md).
