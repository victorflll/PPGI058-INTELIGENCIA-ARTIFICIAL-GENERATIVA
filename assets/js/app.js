/* =========================================================================
   IA Generativa · static course site
   Zero-build: fetches structure.json + markdown, renders in-browser.
   ========================================================================= */
(() => {
  "use strict";

  const CONTENT_BASE = "content/";
  const $ = (sel, root = document) => root.querySelector(sel);

  const state = {
    course: {},
    sections: [],
    byPath: new Map(),   // "present/guidelines/synthesis" -> node
    byMd: new Map(),     // normalized md path -> node
    order: [],           // DFS pre-order of all nodes (book order)
    rows: new Map(),     // path -> sidebar row element
    items: new Map(),    // path -> sidebar item element (expandable)
    current: null,
  };

  /* ---------- tiny DOM helper ---------- */
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null) continue;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(node.dataset, v);
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children)) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  const ICON = {
    chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
    chevDown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    pdf: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h1.5a1.5 1.5 0 0 1 0 3H9zm0 0v5"/><path d="M16 13h-2v5m0-2.5h1.6"/></svg>',
    external: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17 17 7M9 7h8v8"/></svg>',
  };

  /* ---------- indexing ---------- */
  function indexTree(list, parentPath, parent, depth) {
    list.forEach((n) => {
      n.pathArr = [...parentPath, n.slug];
      n.pathStr = n.pathArr.join("/");
      n.parent = parent;
      n.depth = depth;
      state.byPath.set(n.pathStr, n);
      state.order.push(n);
      if (n.markdown) state.byMd.set(normMd(n.markdown), n);
      if (Array.isArray(n.children) && n.children.length) {
        indexTree(n.children, n.pathArr, n, depth + 1);
      }
    });
  }
  const normMd = (p) => p.replace(/^\.?\//, "").replace(/^content\//, "").toLowerCase();

  /* ---------- sidebar ---------- */
  function buildSidebar() {
    const tree = $("#tree");
    tree.innerHTML = "";
    state.sections.forEach((n) => tree.appendChild(buildNode(n, n.depth, true)));
  }

  function buildNode(n, depth, topLevel) {
    const hasChildren = Array.isArray(n.children) && n.children.length;
    const item = el("div", { class: "tree__item", dataset: { path: n.pathStr } });

    const disc = el("span", {
      class: "tree__disc" + (hasChildren ? " is-branch" : ""),
      html: hasChildren ? ICON.chevron : "",
      "aria-label": hasChildren ? "Expandir" : null,
    });
    if (hasChildren) {
      disc.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation(); item.classList.toggle("is-open");
      });
    }

    const row = el("a", {
      class: "tree__row", href: "#/" + n.pathStr, dataset: { path: n.pathStr },
      style: `--depth:${depth}`,
    }, [disc, el("span", { class: "tree__label", text: n.name })]);

    row.addEventListener("click", (e) => {
      if (hasChildren && state.current === n) { e.preventDefault(); item.classList.toggle("is-open"); return; }
      closeNavSoon();
    });

    item.appendChild(row);

    if (hasChildren) {
      const inner = el("div", {}, n.children.map((c) => buildNode(c, depth + 1, false)));
      item.appendChild(el("div", { class: "tree__sub" }, [inner]));
    }

    state.rows.set(n.pathStr, row);
    state.items.set(n.pathStr, item);
    return item;
  }

  function setActive(node) {
    state.rows.forEach((r) => r.classList.remove("is-active"));
    if (!node) return;
    const row = state.rows.get(node.pathStr);
    if (row) row.classList.add("is-active");
    // open self (if branch) + all ancestors
    let cur = node;
    while (cur) {
      const item = state.items.get(cur.pathStr);
      if (item && cur.children && cur.children.length && cur === node) item.classList.add("is-open");
      if (item && cur !== node) item.classList.add("is-open");
      cur = cur.parent;
    }
    if (row) row.scrollIntoView({ block: "nearest" });
  }

  /* ---------- routing ---------- */
  function parseHash() {
    const h = location.hash || "";
    if (!h.startsWith("#/")) return null;             // ignore in-page anchors
    return h.slice(2).split("/").filter(Boolean);
  }

  async function route() {
    const parts = parseHash();
    if (!parts || parts.length === 0) { renderHome(); return; }
    const node = state.byPath.get(parts.join("/"));
    if (!node) { renderNotFound(parts.join("/")); return; }
    state.current = node;
    setActive(node);
    if (node.markdown) await renderLeaf(node);
    else renderSection(node);
    window.scrollTo({ top: 0, behavior: "auto" });
    $("#content").focus({ preventScroll: true });
  }

  function swap(node) {
    const view = $("#view");
    view.innerHTML = "";
    view.appendChild(node);
  }

  /* ---------- crumbs ---------- */
  function crumbs(node) {
    const chain = [];
    let cur = node;
    while (cur) { chain.unshift(cur); cur = cur.parent; }
    const frag = el("nav", { class: "crumbs", "aria-label": "Trilha de navegação" });
    frag.appendChild(el("a", { href: "#/", text: "Início" }));
    chain.forEach((c, i) => {
      frag.appendChild(el("span", { class: "sep", text: "›" }));
      if (i === chain.length - 1) frag.appendChild(el("span", { class: "here", text: c.name }));
      else frag.appendChild(el("a", { href: "#/" + c.pathStr, text: c.name }));
    });
    return frag;
  }

  /* ---------- home ---------- */
  function renderHome() {
    state.current = null;
    setActive(null);
    const page = el("div", { class: "page" });
    page.appendChild(el("div", { class: "phead" }, [
      el("span", { class: "phead__kicker", text: state.course.subtitle || "PPGI · UFAL" }),
      el("h1", { class: "phead__title", text: state.course.title || "IA Generativa" }),
      el("p", { class: "phead__lead", text: state.course.tagline || "" }),
      el("div", { class: "phead__rule" }),
    ]));

    page.appendChild(el("div", { class: "cards__intro", text: "Assuntos" }));
    const cards = el("div", { class: "cards" });
    state.sections.forEach((n, i) => cards.appendChild(sectionCard(n, i)));
    page.appendChild(cards);
    document.title = `${state.course.title} · PPGI / UFAL`;
    swap(page);
  }

  /* ---------- section page ---------- */
  function renderSection(node) {
    const page = el("div", { class: "page" });
    page.appendChild(crumbs(node));
    page.appendChild(pageHeader(node));

    const kids = node.children || [];
    if (kids.length) {
      page.appendChild(el("div", { class: "cards__intro", text: `${kids.length} ${kids.length === 1 ? "tópico" : "tópicos"}` }));
      const cards = el("div", { class: "cards" });
      kids.forEach((c, i) => cards.appendChild(sectionCard(c, i)));
      page.appendChild(cards);
    } else {
      page.appendChild(el("div", { class: "empty", text: "Conteúdo em breve." }));
    }
    page.appendChild(pageNav(node));
    document.title = `${node.name} · ${state.course.title}`;
    swap(page);
  }

  function pageHeader(node) {
    const head = el("div", { class: "phead" });
    head.appendChild(el("h1", { class: "phead__title", text: node.name }));
    if (node.description) head.appendChild(el("p", { class: "phead__lead", text: node.description }));
    head.appendChild(el("div", { class: "phead__rule" }));
    return head;
  }

  // a child node rendered as a single clickable navigation card
  function sectionCard(node, i) {
    return el("a", { class: "navcard", href: "#/" + node.pathStr, style: `--i:${i}` }, [
      el("div", { class: "navcard__body" }, [
        el("div", { class: "navcard__name", text: node.name }),
        node.description ? el("div", { class: "navcard__desc", text: node.description }) : null,
      ]),
      el("span", { class: "navcard__arrow", html: ICON.arrow }),
    ]);
  }

  /* ---------- leaf (markdown) page ---------- */
  async function renderLeaf(node) {
    const page = el("div", { class: "page" });
    page.appendChild(crumbs(node));
    const article = el("article", { class: "prose" });
    page.appendChild(article);
    page.appendChild(pageNav(node));
    swap(page);
    article.innerHTML = skeletonHTML();
    document.title = `${node.name} · ${state.course.title}`;

    try {
      const url = CONTENT_BASE + node.markdown;
      const res = await fetch(url, { cache: "no-cache" });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${url})`);
      const md = normalizeMarkdown(await res.text());
      article.innerHTML = marked.parse(md);
      enhance(article, node);
    } catch (err) {
      article.innerHTML = "";
      article.appendChild(el("div", { class: "errbox" }, [
        el("p", { html: "<strong>Não foi possível carregar este conteúdo.</strong>" }),
        el("p", { html: `Detalhes: <code>${String(err.message || err)}</code>` }),
        el("p", { class: "acc__desc", text: "Ao rodar localmente, sirva a pasta com um servidor estático (ex.: python3 -m http.server), pois o navegador bloqueia fetch via file://." }),
      ]));
    }
  }

  // strip invisible characters that break Markdown parsing (common in Notion / Google Docs exports:
  // word joiners around code, zero-width spaces, soft hyphens, non-breaking spaces)
  function normalizeMarkdown(s) {
    return s
      .replace(/\uFEFF/g, "")                       // byte-order mark
      .replace(/[\u200B-\u200D\u2060]/g, "")        // zero-width space/joiner + word joiner
      .replace(/\u00AD/g, "")                        // soft hyphen
      .replace(/[\u00A0\u202F\u2009]/g, " ");       // (narrow) no-break space, thin space -> normal space
  }

  // filename-like extensions that are never real TLDs (used to spot bogus autolinks)
  const FILE_EXT = new Set(("py js ts jsx tsx txt json yaml yml toml rb go rs java kt swift c cc cpp h hpp " +
    "cs php css scss sass sql sh bash zsh md mdx ipynb env cfg ini conf lock mod xml html vue svelte").split(" "));

  /* ---------- markdown enhancement ---------- */
  function enhance(root, node) {
    // code blocks: mermaid vs highlight
    root.querySelectorAll("pre > code").forEach((code) => {
      const pre = code.parentElement;
      const lang = (code.className.match(/language-([\w-]+)/) || [])[1] || "";
      if (lang === "mermaid") {
        const wrap = el("div", { class: "mermaid-wrap" }, [el("div", { class: "mermaid", text: code.textContent })]);
        pre.replaceWith(wrap);
        return;
      }
      pre.setAttribute("data-lang", lang || "code");
      if (lang && window.hljs && hljs.getLanguage(lang)) {
        try { hljs.highlightElement(code); } catch (_) {}
      }
    });

    // wrap tables for horizontal scroll
    root.querySelectorAll("table").forEach((t) => {
      if (t.parentElement && t.parentElement.classList.contains("table-wrap")) return;
      const wrap = el("div", { class: "table-wrap" });
      t.replaceWith(wrap); wrap.appendChild(t);
    });

    // heading anchors (scroll only, no hash change to keep router intact)
    root.querySelectorAll("h2, h3").forEach((h) => {
      const id = slugify(h.textContent);
      if (!id) return;
      h.id = id;
      const link = el("button", { class: "headlink", "aria-label": "Link para esta seção", text: "#" });
      link.addEventListener("click", () => h.scrollIntoView({ behavior: "smooth", block: "start" }));
      h.appendChild(link);
    });

    // resolve relative image sources against the markdown file's directory
    root.querySelectorAll("img").forEach((img) => {
      const src = img.getAttribute("src") || "";
      if (src && !/^(https?:|data:)/i.test(src)) img.setAttribute("src", resolveAsset(src, node));
      img.setAttribute("loading", "lazy");
      img.setAttribute("decoding", "async");
      img.classList.add("md-img");
    });

    // links: PDFs -> inline viewer; internal .md -> hash route; external -> new tab
    root.querySelectorAll("a[href]").forEach((a) => {
      const href = a.getAttribute("href");
      if (!href) return;
      if (/\.pdf(["']?)([#?].*)?$/i.test(href)) {
        const clean = href.replace(/^["']|["']$/g, "");
        const url = /^https?:/i.test(clean) ? clean : resolveAsset(clean, node);
        const title = (a.textContent || "").trim() || decodeURIComponent((clean.split("/").pop() || "PDF").replace(/[#?].*$/, ""));
        const host = (a.parentElement && a.parentElement.tagName === "P"
          && a.parentElement.children.length === 1
          && a.parentElement.textContent.trim() === a.textContent.trim()) ? a.parentElement : a;
        host.replaceWith(pdfCard(url, title));
        return;
      }
      // Notion exports a bare filename like `models.py` as a dead autolink to http://models.py/.
      // Detect host == link text with a code-file extension and render it as inline code instead.
      const bogus = href.match(/^http:\/\/([a-z0-9._-]+\.[a-z0-9]{1,6})\/?$/i);
      if (bogus) {
        const txt = (a.textContent || "").trim();
        if (txt.toLowerCase() === bogus[1].toLowerCase() && FILE_EXT.has(bogus[1].split(".").pop().toLowerCase())) {
          a.replaceWith(el("code", { text: txt }));
          return;
        }
      }
      if (/^https?:\/\//i.test(href)) { a.target = "_blank"; a.rel = "noopener noreferrer"; return; }
      if (/\.md(["'])?$/i.test(href) || /\.md[#?]/i.test(href)) {
        const target = resolveMdLink(href, node);
        if (target) a.setAttribute("href", "#/" + target.pathStr);
      }
    });

    // render mermaid diagrams
    const diagrams = root.querySelectorAll(".mermaid");
    if (diagrams.length && window.mermaid) {
      mermaid.run({ nodes: diagrams }).catch((e) => console.warn("mermaid:", e));
    }
  }

  // normalize a relative path (handles ./, ../, subfolders) into a content-root-relative path
  function resolveRel(path, node) {
    let clean = path.replace(/^["']|["']$/g, "").replace(/[#?].*$/, "");
    const baseDir = node.markdown.includes("/") ? node.markdown.replace(/\/[^/]*$/, "/") : "";
    clean = clean.startsWith("/") ? clean.slice(1) : baseDir + clean;
    const parts = [];
    for (const seg of clean.split("/")) {
      if (seg === "" || seg === ".") continue;
      if (seg === "..") parts.pop(); else parts.push(seg);
    }
    return parts.join("/");
  }

  function resolveMdLink(href, node) {
    return state.byMd.get(normMd(resolveRel(href, node))) || null;
  }

  // resolve a relative asset path (image, pdf) against the markdown file's folder
  function resolveAsset(path, node) {
    const clean = path.replace(/^["']|["']$/g, "");
    if (/^(https?:|data:)/i.test(clean)) return clean;
    return CONTENT_BASE + resolveRel(clean, node);
  }

  // build a collapsible inline PDF viewer (iframe loaded on first open)
  function pdfCard(url, title) {
    const card = el("div", { class: "pdf" });
    const head = el("button", { class: "pdf__head", type: "button" }, [
      el("span", { class: "pdf__icon", html: ICON.pdf }),
      el("span", { class: "pdf__meta" }, [
        el("span", { class: "pdf__name", text: title }),
        el("span", { class: "pdf__hint", text: "PDF · clique para ver" }),
      ]),
      el("span", { class: "pdf__chev", html: ICON.chevDown }),
    ]);
    const frameHolder = el("div", { class: "pdf__frame" });
    const actions = el("div", { class: "pdf__actions" }, [
      el("a", { class: "pdf__open", href: url, target: "_blank", rel: "noopener noreferrer" },
        [document.createTextNode("Abrir em nova aba"), el("span", { html: ICON.external })]),
    ]);
    const body = el("div", { class: "pdf__body" }, [el("div", { class: "pdf__inner" }, [frameHolder, actions])]);
    let loaded = false;
    head.addEventListener("click", () => {
      const open = card.classList.toggle("is-open");
      if (open && !loaded) {
        loaded = true;
        frameHolder.appendChild(el("iframe", { class: "pdf__iframe", src: url, title, loading: "lazy" }));
      }
    });
    card.append(head, body);
    return card;
  }

  const slugify = (s) => s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "").slice(0, 60);

  /* ---------- prev / next ---------- */
  function pageNav(node) {
    const i = state.order.indexOf(node);
    const prev = i > 0 ? state.order[i - 1] : null;
    const next = i >= 0 && i < state.order.length - 1 ? state.order[i + 1] : null;
    if (!prev && !next) return el("div");
    const nav = el("div", { class: "pagenav" });
    if (prev) nav.appendChild(el("a", { class: "prev", href: "#/" + prev.pathStr }, [
      el("span", { class: "pn-dir", text: "← Anterior" }), el("span", { class: "pn-name", text: prev.name }),
    ]));
    if (next) nav.appendChild(el("a", { class: "next", href: "#/" + next.pathStr }, [
      el("span", { class: "pn-dir", text: "Próximo →" }), el("span", { class: "pn-name", text: next.name }),
    ]));
    return nav;
  }

  /* ---------- misc states ---------- */
  function skeletonHTML() {
    return `<div class="skeleton"><div class="bar title"></div>
      <div class="bar w90"></div><div class="bar w80"></div><div class="bar w60"></div>
      <div class="bar w80"></div><div class="bar w90"></div></div>`;
  }
  function renderNotFound(path) {
    const page = el("div", { class: "page" }, [
      el("div", { class: "phead" }, [
        el("span", { class: "phead__kicker", text: "404" }),
        el("h1", { class: "phead__title", text: "Página não encontrada" }),
        el("p", { class: "phead__lead", html: `Nenhuma seção corresponde a <code>${path}</code>.` }),
      ]),
      el("a", { class: "acc__open", href: "#/" }, [document.createTextNode("Voltar ao início"), el("span", { html: ICON.arrow })]),
    ]);
    swap(page);
  }

  /* ---------- chrome: drawer + progress ---------- */
  function openNav(on) { document.body.classList.toggle("nav-open", on);
    $("#navToggle").setAttribute("aria-expanded", String(on)); $("#scrim").hidden = !on; }
  function closeNavSoon() { if (window.matchMedia("(max-width: 920px)").matches) setTimeout(() => openNav(false), 120); }

  const atmosphere = $(".atmosphere");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function updateProgress() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
    $("#progress span").style.width = pct + "%";
  }

  // Parallax: the atmosphere is a full-document-height layer that already scrolls with
  // the page. Pushing it DOWN by a fraction of the scroll makes it travel at ~0.75x —
  // it moves with the content but lags slightly, for depth. The offset stays within the
  // layer (it is shorter than the document), so an edge is never exposed.
  function updateParallax() {
    if (!atmosphere || reduceMotion) return;
    atmosphere.style.setProperty("--bg-y", (Math.max(0, window.scrollY) * 0.25).toFixed(1) + "px");
  }

  let scrollTicking = false;
  function onScroll() {
    if (scrollTicking) return;
    scrollTicking = true;
    requestAnimationFrame(() => { updateProgress(); updateParallax(); scrollTicking = false; });
  }

  /* ---------- boot ---------- */
  async function boot() {
    mermaid.initialize({
      startOnLoad: false, securityLevel: "loose", theme: "base",
      fontFamily: "JetBrains Mono, monospace",
      themeVariables: {
        background: "transparent", primaryColor: "rgba(79,141,255,0.16)",
        primaryTextColor: "#e7e9f5", primaryBorderColor: "#4f8dff",
        lineColor: "#7f93cc", secondaryColor: "rgba(92,196,255,0.12)",
        tertiaryColor: "rgba(255,255,255,0.04)", clusterBkg: "rgba(255,255,255,0.03)",
        clusterBorder: "rgba(255,255,255,0.14)", titleColor: "#fff",
        edgeLabelBackground: "#0a1424", nodeTextColor: "#e7e9f5",
      },
    });
    marked.setOptions({ gfm: true, breaks: false });

    try {
      const res = await fetch(CONTENT_BASE + "structure.json", { cache: "no-cache" });
      if (!res.ok) throw new Error(`structure.json: ${res.status}`);
      const data = await res.json();
      state.course = data.course || {};
      state.sections = data.sections || [];
      indexTree(state.sections, [], null, 0);

      if (state.course.title) $("#brandTitle").textContent = state.course.title;
      if (state.course.subtitle) $("#brandSub").textContent = state.course.subtitle;
    } catch (err) {
      $("#view").innerHTML = "";
      $("#view").appendChild(el("div", { class: "errbox" }, [
        el("p", { html: "<strong>Falha ao carregar a estrutura do curso.</strong>" }),
        el("p", { html: `<code>${String(err.message || err)}</code>` }),
        el("p", { class: "acc__desc", text: "Sirva a pasta do site com um servidor estático e recarregue." }),
      ]));
      document.body.dataset.loading = "false";
      return;
    }

    buildSidebar();
    document.body.dataset.loading = "false";

    $("#navToggle").addEventListener("click", () => openNav(!document.body.classList.contains("nav-open")));
    $("#scrim").addEventListener("click", () => openNav(false));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") openNav(false); });
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    window.addEventListener("hashchange", () => { route(); });

    await route();
    updateProgress();
    updateParallax();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
