# Chapter Code Checkpoints

Each file here is the complete, runnable state of the project at the end of a chapter. Use them to diff against your own file when something drifts, or to jump into the book at any chapter without typing your way there.

## The Files

| File | State at end of |
|------|-----------------|
| `agentic-demo-chapter-03.py` | Chapter 3 — first runnable loop, `calculate` + `compare` |
| `agentic-demo-chapter-04.py` | Chapter 4 — RAG (ChromaDB + PDF ingestion) and history compaction |
| `agentic-demo-chapter-05.py` | Chapter 5 — grounding-hardened system prompt |
| `agentic-demo-chapter-06.py` | Chapter 6 — citation-verification output guardrail |
| `agentic-demo-chapter-07.py` | Chapter 7 — calibrated similarity threshold |
| `agentic-demo-chapter-08.py` | Chapter 8 — termination backstop, chunk filter, versioned cache key |
| `agentic-demo-chapter-09.py` | Chapter 9 — two-pass input guardrails (Part 2 complete) |
| `mcp-weather-server-chapter-12.py` | Chapter 12 — the MCP server (new file; agent unchanged from Ch. 9) |
| `agentic-demo-chapter-13.py` | Chapter 13 — MCP discovery and dual dispatch (final) |

Chapters 1-2, 10-11, and 14-15 change no code, so they have no files. Chapters 10 and 11 say so deliberately; it's their point.

Each `agentic-demo-chapter-NN.py` was produced by applying that chapter's "Code Changed This Chapter" edits to the previous checkpoint, so the sequence is one continuous lineage: diffing consecutive files shows you exactly (and only) what each chapter changed.

## How to Run a Checkpoint

The code expects its real working filenames, so copy rather than run in place:

```bash
cp code/agentic-demo-chapter-07.py agentic_demo.py
python agentic_demo.py
```

For Chapters 12-13, the server must also exist under its runtime name (the agent launches it as a subprocess by that exact filename):

```bash
cp code/agentic-demo-chapter-13.py agentic_demo.py
cp code/mcp-weather-server-chapter-12.py mcp_weather_server.py
python agentic_demo.py
```

## What Each Stage Needs

| Checkpoints | pip packages | Ollama models |
|-------------|--------------|---------------|
| Chapter 3 | `openai` | chat model |
| Chapters 4-8 | + `chromadb`, `pypdf` | + `nomic-embed-text`; and `Marlow_and_Sage_Handbook.pdf` in the working directory |
| Chapter 9 | (same) | + `granite4.1-guardian:8b` (optional; set `USE_LLM_GUARD = False` without it) |
| Chapters 12-13 | + `mcp` (Python 3.10+) | (same) |

On Apple Silicon, change `MODEL` to the `-mlx` tag you pulled (see Chapter 1). If you jump straight to a late checkpoint, the first run still ingests the PDF and calibrates the threshold; that's the expected minute of startup work, not a hang.
