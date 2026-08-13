# Agentic AI from Scratch — website

Static companion site for the book. Portable by design: **copy this folder anywhere and it works.**

## Deploying

No build step, no dependencies, no server requirements beyond static file hosting.

**Current setup: GitHub Pages** at https://robklause.github.io/agentic/, published automatically by [`.github/workflows/static.yml`](../.github/workflows/static.yml) on any push to `main` that touches `site/`. Repo settings must have **Pages → Source → GitHub Actions**.

The site is served from a subpath (`/agentic/`) and works there because every internal link is relative. Nothing here assumes a domain root.

One file in here exists only for Pages:

- `.nojekyll` — belt and braces. The Actions workflow uploads a plain artifact and never runs Jekyll, but this keeps things predictable if you ever switch to branch-based publishing, where Jekyll would otherwise ignore any file starting with an underscore.

To use a custom domain, add a `CNAME` file here containing just the domain, and set up DNS first. See the [root README](../README.md#adding-a-custom-domain-later).

Anywhere else works too, because every link is relative:

- **Domain root** — copy the contents of `site/` to your web root.
- **Subdirectory** — copy `site/` to `example.com/agentic/`, at any depth.
- **From disk** — open `index.html` in a browser. Works offline.
- **Netlify / S3 + CloudFront / Cloudflare Pages** — point them at this folder; no configuration needed.

The only external reference on the entire site is one link to `ollama.com` on the setup page. No CDNs, no webfonts, no analytics, no JavaScript.

## Files

| Path | What it is |
|------|-----------|
| `index.html` | Landing page |
| `setup.html` | Install: Python, VS Code, Ollama, models, hardware expectations |
| `loop.html`, `first-tools.html`, `knowledge-base.html` | Part 1: the loop and tool calling |
| `why-guardrails.html`, `output-guardrails.html`, `similarity-threshold.html`, `guardrails-together.html`, `input-guardrails.html`, `bigger-models.html` | Part 2: guardrails |
| `mcp-overview.html`, `mcp-server.html`, `mcp-client.html` | Part 3: MCP |
| `tips.html`, `going-further.html`, `troubleshooting.html` | Standalone reference pages |
| `code.html` + `code/*.py` | Runnable checkpoints, one per construction stage |
| `code/Marlow_and_Sage_Handbook.pdf` | The 58-page fictional handbook the agent searches (92KB) |
| `style.css` | The entire stylesheet |

## Editing

Each page is standalone HTML with an inlined header, sidebar nav, and footer. There's no templating layer, which is the tradeoff for zero build tooling: **adding or renaming a page means updating the sidebar `<nav class="sidebar">` block in every file.** A `sed` one-liner or a five-line script handles it.

Conventions worth keeping if you extend the site:

- Every page opens with a `.lede` paragraph and a `.builds-on` line (what it assumes, what it produces).
- Section `<h2>`s carry stable `id` attributes; other pages deep-link to them.
- Callouts come in three flavors: `.callout.problem`, `.callout.design`, `.callout.takeaway`.
- Filenames are search-phrase slugs, not chapter numbers.
- Every page ends with `<nav class="pagenav">` prev/next links.

## Keeping it in sync with the book

The book (`../book/`) is the source of truth for code, model strings, thresholds, and design decisions. The `code/` directory here is a copy of `../book/code/` plus the handbook PDF; re-copy after any change to the checkpoints:

```bash
cp ../book/code/*.py code/
cp ../Marlow_and_Sage_Handbook.pdf code/
```
