# Agentic AI from Scratch

**Build a local agent with Ollama and Python, from the loop to guardrails to MCP.**

A free book and companion website for solution architects who want to understand agentic AI by building one real system, end to end, on their own hardware. No API keys, no credits, no framework hiding the loop.

**Read it online: [agentic.koidev.us](https://agentic.koidev.us)**

---

## What's here

| Path | What it is |
|------|-----------|
| `book/` | The book, one markdown file per chapter, plus both appendices |
| `book/code/` | Runnable checkpoints, one per construction chapter |
| `site/` | The website (static HTML, no build step), published to GitHub Pages |
| `agentic_demo.py` | The finished agent |
| `mcp_weather_server.py` | The MCP server the agent connects to |
| `Marlow_and_Sage_Handbook.pdf` | The 58-page fictional handbook the agent searches |
| `requirements.txt` | Four dependencies |
| `writing-prompts.md` | The prompts used to write the book and site |

## Running the agent

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10+ required
pip install -r requirements.txt

ollama pull qwen3.5:9b              # or qwen3.5:9b-mlx on Apple Silicon
ollama pull nomic-embed-text
ollama pull granite4.1-guardian:8b  # optional

python agentic_demo.py
```

First run ingests the PDF (a minute or so) and calibrates the retrieval threshold. Later runs skip straight to querying. Full setup notes, including hardware expectations and the one misleading pip error, are in [`book/01-setup.md`](book/01-setup.md) or on the [setup page](https://agentic.koidev.us/setup.html).

To start from any chapter instead of the finished system, copy the matching checkpoint:

```bash
cp book/code/agentic-demo-chapter-07.py agentic_demo.py
```

## The website

`site/` is plain HTML with one stylesheet. No build step, no dependencies, no JavaScript. Copy the folder anywhere and it works: domain root, subdirectory, or straight off disk.

It deploys automatically to GitHub Pages on any push to `main` that touches `site/`. See [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) and [`site/README.md`](site/README.md).

## Deploying to GitHub Pages

One-time setup:

1. Push this repo to GitHub as a **public** repo (free Pages requires public).
2. **Settings → Pages → Source → GitHub Actions.**
3. Add a DNS record at your registrar:
   - **Subdomain** (current setup, `site/CNAME` says `agentic.koidev.us`):
     `CNAME  agentic  →  YOURNAME.github.io`
   - **Apex** (`koidev.us`): change `site/CNAME` to `koidev.us`, then add four `A` records to `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`, plus the matching `AAAA` records if you want IPv6.
4. Back in **Settings → Pages**, confirm the custom domain and tick **Enforce HTTPS**. GitHub provisions the certificate automatically once DNS resolves; it can take a few minutes to an hour.

After that, deploys are just `git push`.

## Notes on the source

Everything in the book was built and run against local models on real hardware. Where the book quotes a number (calibration scores, similarity thresholds), it came from an actual run and is hedged accordingly, because those values are a property of your document, chunking, and embedding model together.

## License

Dual-licensed, split by what the thing actually is:

| What | License | Practical effect |
|------|---------|------------------|
| Code — `.py` files, `site/style.css`, HTML structure, and **every code sample in the prose** | [MIT](LICENSE) | Copy it into your own work freely, no attribution required |
| Prose — book chapters, website text, and the handbook PDF | [CC BY 4.0](LICENSE-CONTENT) | Share and adapt for any purpose, including commercially, with credit |

Code samples in the book are deliberately MIT rather than CC BY, so you can lift them into your own projects without attribution obligations.

Attribution for the prose, if you need the boilerplate:

> "Agentic AI from Scratch" by Rob Klause, licensed under CC BY 4.0. https://agentic.koidev.us

**On AI-generated portions:** the prose here was drafted with substantial AI involvement (see the book's introduction). Under current U.S. law, purely AI-generated expression may not be copyrightable, and copyright extends only to the human contributions. The content license is granted to the extent copyright subsists, and is not a claim that every sentence is protected. [`LICENSE-CONTENT`](LICENSE-CONTENT) says this in full. Not legal advice.

## Also available on Kindle

A Kindle edition is available for readers who prefer it on an e-reader. Everything in it is on this site for free; the paid edition is convenience and support, not exclusive access.

## Author

Rob Klause
