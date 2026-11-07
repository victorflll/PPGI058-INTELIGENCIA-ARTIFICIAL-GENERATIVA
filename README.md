# Generative AI · class wiki

Wiki for the **Generative Artificial Intelligence** course (PPGI058 · IC / UFAL · 2026.2).
Each group has its own section in the sidebar and fills it in after presenting:

| # | Topic | Folder | Status |
|---|---|---|---|
| 1 | Large Language Model | `content/large-language-model/` | published |
| 2 | Small Language Model | `content/small-language-model/` | waiting for the group |
| 3 | LLMs in Production | `content/llms-in-production/` | waiting for the group |
| 4 | Agentic Coding | `content/agentic-coding/` | waiting for the group |
| 5 | Generative Deep Learning | `content/generative-deep-learning/` | waiting for the group |

The **Large Language Model** section follows Sebastian Raschka's *Build a Large Language
Model (From Scratch)*: theory for chapters 1 to 3 (based on the presentation slides) and the
hands-on class built on the notebooks in
[victorflll/LLMs-from-scratch](https://github.com/victorflll/LLMs-from-scratch/tree/main/30-09).

It is a **no-build** site: the browser reads `content/structure.json`, which describes the
page tree, and renders the Markdown live. There is no server, framework, or compilation
step, just HTML, CSS, and a single JavaScript file. That is why it runs directly on
**GitHub Pages**.

## How it works

- **`content/structure.json`** defines the tree of sections (sidebar, pages, navigation).
- Each node in the tree is a **page**:
  - a node **with `markdown`** renders that Markdown file (a content page);
  - a node **with `children`** becomes a section page showing its description and
    clickable *cards* for the subpages.
- Navigation uses the URL **hash** (`#/large-language-model/practice/attention`), so page
  reloads and direct links work without any server configuration.

Markdown is rendered with headings, lists, tables (with horizontal scrolling on mobile),
syntax-highlighted code blocks, **Mermaid** diagrams, images, and an **inline PDF viewer**
(every `.pdf` link becomes a card that opens the PDF inline).

## Repository structure

```
.
├── index.html              # page shell (sidebar, top bar, CDN libs)
├── .nojekyll               # disables Jekyll on GitHub Pages
├── assets/
│   ├── css/styles.css      # obsidian glass theme (design tokens in :root)
│   ├── js/app.js           # router + renderer (fetch → marked → highlight)
│   └── images/             # favicons, logo
└── content/
    ├── structure.json      # << the wiki page tree
    ├── large-language-model/   # group 1
    │   ├── theory/             # ch. 1–3 + figures/ + pdfs/ (slides)
    │   ├── practice/           # setup + 7 converted notebooks + figures/
    │   └── team/               # group members
    └── <group-topic>/          # groups 2–5 create their folder here
```

## How your group publishes its content

Your group's section already exists in `content/structure.json`, but without pages (so it
shows "Content coming soon."). To publish:

1. Write your Markdown pages in your group's folder, e.g. `content/agentic-coding/overview.md`,
   with images and PDFs next to them (`figures/`, `pdfs/`).
2. In `content/structure.json`, add the pages as `children` of your group's node, in reading
   order. Use the **Large Language Model** section as a template (Theory / Practice / Group
   team subsections).
3. Don't edit other groups' sections; prefer one pull request per group.
4. Run the site locally and check every page before opening the PR.

## Editing content

1. Write/edit Markdown inside `content/<topic>/…`.
2. Put images and PDFs next to the `.md` and reference them with a relative path, e.g.
   `![](figures/diagram.png)`, `[Slides](pdfs/slides.pdf)`.
3. Register the page in `content/structure.json` with `name`, `slug`, `description`, and
   `markdown` (content page) **or** `children` (section page).
4. Use `kebab-case` *slugs*, ASCII only, unique among siblings.

## Running locally

The browser blocks `fetch` over `file://`, so serve the folder over HTTP:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Publishing

It is a pure static site: just serve the repository contents on GitHub Pages. There is
nothing to compile.

> Architecture details and maintenance conventions (including for AI agents) are in
> [CLAUDE.md](CLAUDE.md).
