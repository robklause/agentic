# Chapter 1: Tools and Installation

By the end of this chapter you'll have a working local AI stack: Python, Ollama, three models, and a project folder that every later chapter builds inside. There's one version trap in here that produces a genuinely misleading error message, so don't skip the Python section even if your machine "already has Python."

## Python 3.10 or Newer. This Is Not Optional.

The `mcp` package, the official Model Context Protocol SDK you'll use in Part 3, requires Python 3.10 or newer. Many systems still ship 3.9.6 as the default `python3`, and 3.9 reached end of life in October 2025.

Here's the trap: if you `pip install mcp` on Python 3.9, pip doesn't say "your Python is too old." It says there's no matching distribution, a generic error that reads like a network problem or a typo in the package name. You can lose a real chunk of time to that message.

Check first:

```bash
python3 --version
```

If you see 3.9.x, install a current Python (3.12 is a safe choice) from python.org or your package manager, and make sure the project's virtual environment uses it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version   # confirm 3.10+ inside the venv
```

## VS Code

Any editor works, but this book assumes VS Code with the Python extension. You'll spend a lot of time reading trace output next to code, and a split editor pane makes that comfortable. Install it from code.visualstudio.com, add the Python extension, and point it at the virtual environment you just created (Command Palette, "Python: Select Interpreter").

## Ollama

Ollama is the local model runtime everything in this book talks to. Install it from ollama.com (macOS, Windows, and Linux installers are all one-step), then confirm it's serving:

```bash
ollama serve
```

If it's already running as a background service, this command tells you the port is in use. That's fine. Either way, Ollama listens on `localhost:11434`.

The detail that makes this whole book work: Ollama exposes an OpenAI-compatible endpoint at `/v1`. The same `openai` Python SDK and the same tool-calling wire format that work against OpenAI's cloud work unchanged against your laptop. Only three things differ: the base URL, the API key (Ollama requires the field but ignores the value), and the model name. You'll see this in the first code you write in Chapter 2:

```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

## The Three Models

Pull all three now. The downloads are the slow part of setup, so start them and keep reading.

```bash
ollama pull qwen3.5:9b               # chat model — the agent's brain
ollama pull nomic-embed-text         # embedding model — powers search in Chapter 4
ollama pull granite4.1-guardian:8b   # guard model — input screening in Chapter 9
```

What each one is for:

**qwen3.5:9b** is the chat model, the one that runs the loop, decides when to call tools, and writes answers. On a Mac with Apple Silicon, pull `qwen3.5:9b-mlx` instead: it's the same model in Apple's MLX format, which runs faster on that hardware but won't work anywhere else. The book's code uses the portable tag; if you're on a Mac, change the `MODEL` string to match whichever you pulled.

One honest caveat: not every local model supports tool calling well. If your model ignores the tools or hallucinates arguments instead of emitting a proper `tool_calls` response, switch to a model explicitly documented as tool-call capable, `gemma4:12b` is a solid alternative, and compare behavior before assuming your code is wrong.

**nomic-embed-text** is an embedding model. It doesn't chat. It converts text into vectors, lists of numbers where similar meanings land near each other. That's the machinery behind semantic search, and Chapter 4 explains it properly when you need it. It's a separate pull because it's a separate model; chat models and embedding models are different tools.

**granite4.1-guardian:8b** is IBM's Granite Guardian, a model fine-tuned for exactly one job: judging whether a piece of text meets a named harm criterion like `jailbreak`. It's not a chat model asked to police itself; it's a specialist. You won't touch it until Chapter 9, and the code treats it as optional, so if you want to defer this download, the system degrades gracefully without it.

## Hardware, and What to Expect

Everything in this book runs on modest hardware. It was developed on two machines worth describing, because they bracket the realistic range: a MacBook Pro with 16GB of RAM, and a workstation with 32GB of RAM plus a dedicated GPU with 16GB of VRAM. Both run every chapter. They do not run them at the same speed, and knowing why saves you from misreading slowness as breakage.

The mechanics: Ollama loads a model into memory (VRAM if you have it, unified or system RAM otherwise) the first time it's called, and keeps it resident while it fits. This book uses three models, and by Chapter 9 a single prompt can touch two of them back to back: Granite Guardian screens the input, then the chat model runs the loop, with the embedding model joining in whenever a search fires. The chat and guard models are each roughly 5 to 6GB in memory at Ollama's default quantization (treat that as an approximation; it varies by model and version). On a 32GB machine with a GPU, everything stays resident and hot. On 16GB, they don't all fit alongside your OS and editor, so Ollama evicts one model to load another, and every switch costs seconds of load time before the first token appears.

That's the performance hit to expect, and it's a stall, not a hang: on a constrained machine, the pause between the Guardian trace line and the first agent turn is a model being swapped into memory. Be patient, especially on first calls. Your mileage will vary with your hardware, and the ordering is intuitive: best is lots of RAM plus a dedicated GPU; Apple Silicon's unified memory sits in the middle and benefits from the `-mlx` model variants; CPU-only with 16GB works and asks the most patience of you. Dedicated accelerators of any kind, GPU, APU, or NPU, help wherever Ollama supports them.

Two levers if the swapping gets painful on a smaller machine. Set `USE_LLM_GUARD = False` (Chapter 9) to drop the guard model from the rotation; the system degrades gracefully to the regex screen, and that's one less 5GB tenant competing for memory. And nothing in this book's architecture depends on model size, so substituting a smaller chat model is always an option while you're learning the mechanics; swap back up when you care about answer quality.

## The Python Packages

Create `requirements.txt` in your project folder with exactly this:

```
openai>=1.40.0
chromadb>=1.0.0
pypdf>=5.0.0
mcp>=2.0.0
```

Then, inside your activated virtual environment:

```bash
pip install -r requirements.txt
```

What each one does: `openai` is the client SDK (pointed at Ollama, not OpenAI). `chromadb` is the vector database for Chapter 4. `pypdf` extracts text from the handbook PDF. `mcp` is the Model Context Protocol SDK for Part 3, and it's the package that silently demands Python 3.10+.

## The Knowledge Base PDF

Chapters 4 onward search a real document: a 58-page employee handbook for a fictional retail boutique, Marlow & Sage. A real multi-page PDF matters here. Toy three-paragraph examples hide every interesting retrieval problem, and the guardrail chapters depend on the document having enough genuine structure to search badly against.

Any comprehensive, sectioned PDF of 40+ pages will work if you substitute your own. The book's examples, page numbers, and calibration queries assume the Marlow & Sage handbook, so expect to adjust those if you swap documents.

## Project Layout

Everything lives in one flat folder. By the end of the book it looks like this:

```
agentic/
├── .venv/                        # your Python 3.10+ virtual environment
├── agentic_demo.py               # the agent — grows from Chapter 2 onward
├── mcp_weather_server.py         # the MCP server — Part 3
├── requirements.txt
├── Marlow_and_Sage_Handbook.pdf  # the knowledge base
└── chroma_db/                    # created automatically on first ingest
```

No src directory, no package structure. Two Python files is the honest size of this system, and keeping it flat keeps every path in the book copy-pasteable.

## Smoke Test

Before moving on, prove the stack works end to end. Save this as `smoke_test.py` and run it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3.5:9b",   # or qwen3.5:9b-mlx if that's what you pulled on a Mac
    messages=[{"role": "user", "content": "Reply with exactly: local stack is working"}],
)
print(response.choices[0].message.content)
```

If you get a sentence back, you have a working local LLM stack and everything else in this book is just code. If you get a connection error, Ollama isn't running (`ollama serve`). If you get a model-not-found error, the pull didn't finish or the name doesn't match; `ollama list` shows what you actually have.

You won't need `smoke_test.py` again. Delete it or keep it as a diagnostic. Chapter 2 starts the real file.
