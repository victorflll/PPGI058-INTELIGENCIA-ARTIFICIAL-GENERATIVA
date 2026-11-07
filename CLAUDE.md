# CLAUDE.md: notes for AI agents

Static, **zero-build** class wiki for *Generative AI* (PPGI058 · IC / UFAL).
Each top-level section is one **group's topic** (Large Language Model, Small Language Model,
LLMs in Production, Agentic Coding, Generative Deep Learning); groups fill in their own section
after presenting. Only `large-language-model` has content so far; the others are empty
section nodes (no `children`, no `markdown`), which render "Content coming soon."
Each group's files live under `content/<group-slug>/`.
Everything (content, UI strings, slugs, file names) is in **English**. The notebooks the
practice pages come from are in Portuguese, so code identifiers, string literals and
recorded outputs inside those pages stay in Portuguese; only prose and comments are translated.
The browser fetches `content/structure.json`, walks the tree, and renders Markdown
client-side. There is **no framework, no bundler, no build step, no server code**.
Keep it that way so it can deploy to GitHub Pages as plain files.

## Architecture (one screen)

- `index.html` is the static shell: sidebar (`#tree`), content host (`#view`), and the
  three pinned CDN libs (`marked@12.0.2`, `highlight.js@11.9.0`, `mermaid@10.9.1`).
- `assets/js/app.js` is the whole app: loads `structure.json`, indexes the tree,
  builds the sidebar, routes on `hashchange`, fetches + renders Markdown, enhances it.
- `assets/css/styles.css` holds the *obsidian glass* theme. Design tokens live in `:root`
  (colors, glass, fonts). Reuse tokens; don't hardcode new colors.
- `content/` holds all course content: `structure.json` plus per-section Markdown and assets.

`CONTENT_BASE = "content/"`. Every `markdown` path in `structure.json` and every asset
path resolved at runtime is relative to that.

## Content model (`content/structure.json`)

```jsonc
{
  "course": { "title": "...", "subtitle": "...", "tagline": "..." },
  "sections": [ Node, ... ]
}
```

A `Node` is:

```jsonc
{
  "name": "Display Name",          // shown in sidebar, cards, breadcrumbs, <title>
  "slug": "kebab-case",            // path segment; unique among siblings
  "description": "one paragraph",  // optional; shown on section pages + cards
  "markdown": "large-language-model/theory/overview.md",// optional; path under content/
  "children": [ Node, ... ]        // optional
}
```

The route for a node is the `/`-joined chain of slugs from the root: e.g.
`#/large-language-model/theory/overview`. The full slug path (not the slug alone) is the unique key.

### Branch vs leaf: the one rule that bites

Routing (`route()` in `app.js`):

- Node **has `markdown`** → renders that file as a **leaf** content page.
- Node **has no `markdown`** → renders a **section** page (its `description` +
  one navigation card per child).

**`markdown` wins over `children`.** A node with *both* renders as a leaf page; its
children are then reachable only through the sidebar tree (expandable), not as cards.
This is intentional (no node uses it right now). If you want a section
*landing* page that lists its children as cards, give the node `children` and **no
`markdown`** (this is how the group sections, `Theory` and `Practice` work).

Node order in `structure.json` is meaningful: a DFS pre-order of the tree is the
"book order" that drives the **Prev / Next** footer links.

## Markdown rendering & enhancement (`enhance()`)

Before parsing, raw text passes through `normalizeMarkdown()`, which strips the
invisible characters that break exported Markdown (word joiners, zero-width spaces,
soft hyphens, BOM) and converts no-break/thin spaces to normal spaces. Notion and
Google Docs exports are riddled with these, so the cleanup keeps pasted content
rendering sanely.

Rendered with `marked` (GFM, `breaks:false`), then post-processed:

- **Code blocks**: ` ```mermaid ` → rendered Mermaid diagram; other languages →
  highlight.js (if the language is registered). Plain fences just get a `data-lang`.
- **Tables**: wrapped in `.table-wrap` for horizontal scroll on mobile.
- **Headings** (`h2`/`h3`): get a slug `id` + a `#` anchor button (smooth-scroll only;
  no hash change, so the router isn't disturbed).
- **Images**: relative `src` resolved against the **Markdown file's own folder**;
  `loading=lazy`. Absolute `http(s):`/`data:` left alone.
- **Links**:
  - `*.pdf` → replaced by a collapsible **inline PDF viewer card** (iframe lazy-loads
    on first open). The card title is the link text → give PDFs human-readable link
    text, not the filename.
  - bogus autolink `[file.py](http://file.py/)` (Notion turns a bare filename into a
    dead link) → rendered as inline `code`. Triggers only when the host equals the
    link text and ends in a code-file extension (`FILE_EXT`), so real links are safe.
  - external `http(s)://` → opens in a new tab.
  - internal `*.md` → rewritten to the hash route **only if** that exact md path is
    registered as some node's `markdown` (looked up in `byMd`, normalized by
    `normMd`: strips leading `./` and `content/`, lowercases). Unregistered `.md`
    links are left as dead hrefs, so to cross-link pages, link to a registered
    page's md path.

## Conventions

- **Slugs / filenames**: `kebab-case`, lowercase, ASCII, no spaces or accents. This
  applies to folders, md files, **and** assets (images, PDFs). Slugify on import
  (e.g. `Apresentação Final.pdf` → `apresentacao-final.pdf`) and update the links.
- **Co-locate assets** with the Markdown that references them, via relative paths
  (`pdfs/…`, `figures/…`). A shared `pdfs/` / `figures/` folder per section is fine (see `large-language-model/theory/`).
- **Reuse CSS tokens** from `:root`; match the existing glass/dark aesthetic.
- **Don't add a build step or dependencies.** New libs, if ever truly needed, go in
  `index.html` as a pinned CDN `<script>`, never npm/bundler.
- Keep `.nojekyll` (lets GitHub Pages serve `_`-prefixed and underscore paths as-is).
- **Importing from Notion / Docs:** `normalizeMarkdown()` cleans invisible characters,
  but it cannot recover backticks an export deleted. Notion frequently strips ```` ``` ````
  fences and inline-code backticks, leaving code as plain paragraphs (or, worse, lines
  starting with `#` parsed as headings). After importing, always open the page and
  check that code blocks, inline code, and headings render as intended — repair the
  fences in the source if not.

## Practice pages are generated from notebooks

`content/large-language-model/practice/0N-*.md` and its `figures/` are converted from the Jupyter
notebooks in [victorflll/LLMs-from-scratch/30-09](https://github.com/victorflll/LLMs-from-scratch/tree/main/30-09):
markdown cells translated to English, code cells as ```` ```python ```` fences, text outputs as
```` ```text ```` fences (long ones clipped), image outputs saved as PNGs, and
`NN_name.ipynb` links rewritten to the registered `.md` pages. If a notebook changes,
regenerate the page the same way rather than hand-editing the output blocks, then
translate prose and code comments only (leave code, string literals and outputs untouched).
`setup.md` is hand-written from the folder's README.

## Add a page: checklist

1. Write `content/<group-slug>/<name>.md`; drop its assets nearby with kebab-case names.
2. Use relative links for assets; human-readable link text for PDFs.
3. Add a `Node` to `structure.json` (`name`, `slug`, `description`, and `markdown`
   **or** `children`) in the position you want it to appear / read.
4. Verify (below). Confirm referenced assets exist on disk.

## Verify before claiming done

`fetch` is blocked over `file://`, so always serve over HTTP:

```bash
python3 -m http.server 8000
```

Then load the route in a browser and check: the page renders, PDF cards expand and
load, tables/diagrams look right, and the **console has zero errors**. Confirm every
`markdown` path in `structure.json` resolves to a real file, and every `pdfs/…` /
image link in the Markdown points to a file that exists.
