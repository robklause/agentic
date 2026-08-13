# Chapter 4: Giving the Agent a Knowledge Base

By the end of this chapter your agent answers questions from a real 58-page PDF: chunked, embedded into a persistent vector database, and searchable as a tool. You'll also hit the context window problem Chapter 2 promised, and fix it. This is the longest chapter in the book because retrieval is where toy examples stop transferring to real systems.

## The Problem Worth Solving

Ask your Chapter 3 agent "How many PTO days do new employees get?" and it will answer. Confidently. From nothing. The model has seen thousands of employee handbooks in training, so it produces a plausible number that has no connection to *your* handbook. This is the failure mode that makes LLMs dangerous in business settings: not ignorance, but fluent fabrication.

The fix has a name, RAG, retrieval-augmented generation, and it's less exotic than the acronym suggests. It's a tool, exactly like `calculate`: the model calls `search_knowledge_base("PTO days")`, your code returns relevant passages from the actual document, and the model answers from those passages instead of from memory. The entire sophistication of RAG lives in one question: how does your code decide which passages are relevant?

## Why Keyword Search Isn't Enough

The obvious first answer is keyword matching: score each passage by how many query words appear in it. It's fast, free, and you could write it in ten lines. It also fails in a way you should see coming before you write a line of retrieval code.

A user asks about "vacation days." The handbook says "paid time off" and "PTO." Zero keyword overlap, zero results, and the failure is silent: the search doesn't error, it just returns nothing or, worse, returns passages that happen to share incidental words. Keyword search matches vocabulary. Users don't share vocabulary with documents. They share *meaning*.

## Embeddings in One Paragraph

An embedding model converts text into a vector, a long list of numbers, built so that texts with similar meaning land near each other in that space. "Vacation days" and "paid time off" produce nearby vectors despite sharing no words. Relevance becomes geometry: embed the query, embed every chunk of the document, and the nearest chunks by cosine similarity are the most relevant. That's semantic search, and the model that does it here is `nomic-embed-text`, the second model you pulled in Chapter 1. It's not a chat model; it does exactly this one thing.

One warning before the code, because it's the classic hard-to-spot RAG bug: **you must embed documents and queries with the same model.** Two different embedding models produce two different vector spaces, and distances between them are meaningless. Nothing errors. Similarity scores still come out looking like numbers. They just don't mean anything. The code below makes this structurally hard to get wrong by routing both through one function.

## The Ingestion Pipeline

Four steps: extract text from the PDF, split it into chunks, embed each chunk, store the vectors. New constants and imports first:

```python
# WHERE: imports join the existing import block at the top of the file;
# the constants go at module level, directly after the compare() function
import hashlib
import os
import re

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"  # pull once: `ollama pull nomic-embed-text`
KB_PDF_PATH = "Marlow_and_Sage_Handbook.pdf"
CHROMA_DB_DIR = "./chroma_db"     # persisted on disk — ingest once, reuse after
COLLECTION_NAME = "handbook"
CHUNK_SIZE = 1000                 # target characters per chunk
CHUNK_OVERLAP = 150               # characters carried into the next chunk
```

ChromaDB is the vector database: it stores chunks alongside their vectors and answers nearest-neighbor queries. The persistent client writes to `./chroma_db` on disk, which is what lets you ingest once and skip the work on every later run.

### One Embedding Function for Both Sides

```python
# WHERE: this class and the next three functions (_extract_pdf_pages,
# _chunk_text, ingest_pdf) go at module level in this order, directly
# after the new constants
class OllamaEmbeddingFunction(EmbeddingFunction):
    """Wraps Ollama's OpenAI-compatible /v1/embeddings endpoint so Chroma can
    call it like any other embedding function. Chroma calls this both when
    documents are added and when a query runs — using the same function for
    both is what keeps them comparable. Embedding docs with one model/function
    and queries with a different one is a classic, hard-to-spot RAG bug: the
    two vector spaces don't actually line up, so similarity scores become
    meaningless even though nothing throws an error.
    """

    def __init__(self) -> None:
        pass  # no state — silences chromadb's "does not implement __init__" deprecation warning

    def __call__(self, input: Documents) -> Embeddings:
        return [
            client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
            for text in input
        ]
```

Registering this class with the Chroma collection means Chroma itself calls it for every document at ingest time and every query at search time. Same function, same model, same vector space, enforced by structure rather than discipline.

### Extraction, and Cleaning What Every Page Repeats

```python
def _extract_pdf_pages(pdf_path: str) -> list:
    """Return [(page_number, cleaned_text), ...], stripping the running
    header every page repeats (e.g. "11MARLOW & SAGE BOUTIQUE ·EMPLOYEE
    HANDBOOK") so it doesn't get embedded into — and pollute — every chunk.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(
            r"^\s*\d*\s*MARLOW & SAGE BOUTIQUE\s*\W?\s*EMPLOYEE HANDBOOK\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if text:
            pages.append((i, text))
    return pages
```

Two things earn their place here. Page numbers are captured with the text, because every chunk will carry a `p.NN` source tag, and in Chapter 6 those tags become the raw material for a guardrail. And the running footer gets stripped, because a string that appears on all 58 pages is pure noise to an embedding model: it dilutes every chunk's vector toward the same meaningless average. If you swap in your own PDF, this regex is the line to change; extract a page with pypdf and look at what boilerplate actually repeats.

### Chunking

You can't embed a 58-page document as one vector, and you wouldn't want to: a query about PTO should retrieve the PTO passage, not the entire handbook. Splitting has real design decisions in it:

```python
def _chunk_text(pages: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Greedily pack paragraphs into ~chunk_size-character chunks tagged with
    the page they came from, keeping whole paragraphs together where
    possible rather than cutting mid-sentence. Carries the tail of one chunk
    into the next (`overlap`) so a fact split across a chunk boundary isn't
    orphaned on one side of it.
    """
    chunks = []
    for page_num, text in pages:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            # Some pages extract as one blob with no blank lines — fall back
            # to single-newline splitting so we still get paragraph-ish units.
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 1 > chunk_size:
                chunks.append({"text": buf, "page": page_num})
                buf = buf[-overlap:] if overlap else ""
            buf = (buf + " " + para).strip()
        if buf:
            chunks.append({"text": buf, "page": page_num})

    return chunks
```

The 1000-character target is a balance: chunks small enough that a match is specific, large enough that a retrieved chunk contains a usable, self-contained statement. The 150-character overlap exists for facts that straddle a boundary; without it, a sentence cut in half is invisible to search from both sides. And packing by paragraph instead of slicing at exact character counts keeps each chunk a coherent unit of meaning, which is what you want when the chunk *is* the retrieval result.

Hold onto one thought for Chapter 8: this function currently keeps everything, including two-word title fragments. That decision has a consequence, and it's a good one to meet after the guardrails exist rather than before.

### Ingest, With a Cache That Knows When to Rebuild

Embedding every chunk of a long document takes a minute or more. Doing it on every run would make the development loop miserable, so ingestion is cached: hash the PDF's bytes, store the hash inside the collection as a sentinel record, and skip everything when the hash matches.

```python
_kb_collection = None


def ingest_pdf(pdf_path: str = KB_PDF_PATH, force: bool = False):
    """Chunk `pdf_path`, embed every chunk via Ollama, and load it into a
    persistent Chroma collection on disk at CHROMA_DB_DIR. Skips
    re-ingestion if this exact file (by content hash) is already indexed —
    re-embedding 58+ pages on every run would be slow for no reason if the
    source hasn't changed. Pass force=True to rebuild anyway.
    """
    global _kb_collection  # cache the handle so a second caller doesn't re-open (and re-log) ingestion
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    collection = None
    already_current = False
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME, embedding_function=OllamaEmbeddingFunction()
        )
        existing = collection.get(ids=["_source_hash"])
        already_current = bool(existing["ids"]) and existing["metadatas"][0].get("hash") == content_hash
    except Exception:
        collection = None

    if collection is not None and already_current and not force:
        print(f"[ingest] '{pdf_path}' already indexed (hash {content_hash}) — skipping re-ingestion.")
        _kb_collection = collection
        return collection

    print(f"[ingest] (re)indexing '{pdf_path}' — embeds every chunk, may take a minute...")
    # Delete unconditionally (not just when the get_collection above visibly
    # succeeded) and swallow "doesn't exist" errors either way. get_collection
    # can fail for reasons other than "not created yet" — e.g. an embedding
    # function config mismatch after a library upgrade — which would leave
    # collection=None here even though a collection with this name already
    # exists on disk. Gating the delete on `collection is not None` hits
    # exactly that: create_collection then fails with "already exists."
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, embedding_function=OllamaEmbeddingFunction()
    )

    pages = _extract_pdf_pages(pdf_path)
    chunks = _chunk_text(pages)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": f"{os.path.basename(pdf_path)} p.{c['page']}", "type": "chunk"}
            for c in chunks
        ],
    )
    collection.add(
        ids=["_source_hash"],
        documents=["source file fingerprint"],
        metadatas=[{"hash": content_hash, "type": "sentinel"}],
    )

    print(f"[ingest] indexed {len(chunks)} chunks from {len(pages)} pages.")
    _kb_collection = collection
    return collection
```

Note the shape of the delete-before-create logic. `get_collection` can fail for reasons other than "doesn't exist yet," a config mismatch after a chromadb upgrade, for instance, so the rebuild path deletes unconditionally inside a try/except rather than trusting the earlier probe. Cache logic is where optimistic assumptions go to die quietly; write it defensively the first time.

Every chunk's metadata carries `source` (like `Marlow_and_Sage_Handbook.pdf p.17`) and a `type` of `"chunk"`, while the sentinel is `type: "sentinel"`. That type field is what keeps the fingerprint record from ever showing up as a search result.

## The Search Tool

Now the tool itself, the third entry in your registry:

```python
# WHERE: insert at module level, directly after ingest_pdf()
def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """Real semantic search: query the persistent Chroma collection, which
    embeds `query` via the same OllamaEmbeddingFunction used at ingest time
    and returns the nearest chunks by cosine distance. This is what the
    agent actually calls. Lazily ingests the PDF on first call in a process.
    """
    global _kb_collection
    if _kb_collection is None:
        _kb_collection = ingest_pdf()

    result = _kb_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"type": "chunk"},  # exclude the sentinel record
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    results = [
        {"source": meta["source"], "text": doc, "similarity": round(1 - dist, 3)}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    return {"results": results}
```

Each result carries three fields, and all three earn their keep later. `text` is what the model reads. `source` is the page citation the model will be required to repeat, and that a guardrail will verify in Chapter 6. `similarity` is the relevance score, `1 - distance`, that Chapter 7 turns into a calibrated cutoff. Right now the tool returns the top 3 chunks no matter how weak the matches are. Remember that; it's the loose thread the next three chapters pull on.

Register it and describe it:

```python
# WHERE: replace the existing TOOL_IMPLEMENTATIONS dict (adds the third entry)
TOOL_IMPLEMENTATIONS = {
    "calculate": calculate,
    "compare": compare,
    "search_knowledge_base": search_knowledge_base,
}
```

And append to `TOOL_SCHEMAS`:

```python
    # WHERE: append as the third entry in the TOOL_SCHEMAS list, after the
    # compare schema's closing brace
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base (employee handbook) for "
                "passages relevant to a question. Returns ranked chunks with "
                "their source. Use this for ANY question about company policy "
                "— remote work, PTO, expenses, etc. — before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'remote work days allowed'"},
                    "top_k": {"type": "integer", "description": "How many chunks to return (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
```

The system prompt gets its matching push. Append this to the system prompt string in `run_agent`, directly after the sentence ending "...rather than deciding in your head.":

```text
For any question about company policy, benefits, or the employee
handbook, always call search_knowledge_base first — do not answer
from your own memory. Only state facts that appear in the returned
chunks, and mention which source each fact came from. If the search
results don't contain the answer, say you don't have that
information rather than guessing.
```

Same belt-and-suspenders pattern as Chapter 3, and the same honest caveat: these are requests. Part 2 is about what happens when requests aren't enough.

## The Bill Comes Due: Context Bloat

Run a handbook question and look at the trace. A single `search_knowledge_base` call returns three chunks of up to a thousand characters each, and because the chat completions API is stateless, that entire payload gets *resent on every subsequent turn*. Two searches into a conversation, most of what you're shipping to the model is stale tool output. This is the context window consequence Chapter 2 flagged: the message list only grows, and everything in it costs processing time on every single call.

The application-level lever is compaction: collapse older tool exchanges into a one-line digest and keep only the recent messages verbatim.

```python
# WHERE: insert both functions at module level, after TOOL_SCHEMAS and
# before run_agent
def _dump_message(m) -> dict:
    """Normalize a message (dict or SDK object) into a plain dict for printing."""
    if isinstance(m, dict):
        return m
    d = {"role": m.role, "content": m.content}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = [
            {"id": c.id, "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls
        ]
    return d


def _compact_history(messages: list, keep_recent: int = 4, verbose: bool = True) -> list:
    """Collapse older tool-call/tool-result exchanges into one short digest
    message, instead of resending every raw message forever.

    Chat completions is stateless — the full `messages` list gets resent
    every turn, so payload only grows. This is an application-level lever
    to fight that: keep the system prompt and the original user question
    verbatim (the model needs those), keep the most recent `keep_recent`
    messages verbatim (recent detail matters most), and boil everything
    older than that down into a single compact summary line per tool call.
    This is a simplification, not free — the model loses exact wording of
    older tool results, so aggressive compaction can lose detail it needs.
    """
    if len(messages) <= 2 + keep_recent:
        return messages  # nothing old enough to compact yet

    head = messages[:2]  # system + original user prompt

    # The split point must never land in the middle of an assistant
    # tool_calls message and its tool result messages — e.g. keeping the two
    # 'tool' results in the tail but pushing the assistant message that named
    # them (and its args) into the middle to be summarized. The next
    # compaction pass would then have no record of which tool a bare number
    # belonged to and fall back to "unknown_tool". So: walk the boundary
    # left until it lands on a clean seam.
    split = len(messages) - keep_recent

    def _call_ids_before(idx):
        ids = set()
        for j in range(2, idx):
            d = _dump_message(messages[j])
            if d["role"] == "assistant":
                for c in d.get("tool_calls", []) or []:
                    ids.add(c["id"])
        return ids

    def _is_orphaned(idx):
        d = _dump_message(messages[idx])
        return d["role"] == "tool" and d.get("tool_call_id") in _call_ids_before(idx)

    while split > 2 and _is_orphaned(split):
        split -= 1

    tail = messages[split:]
    middle = messages[2:split]

    if not middle:
        return messages  # nothing can be safely compacted yet without splitting a pair

    call_args = {}
    digest_lines = []
    for m in middle:
        d = _dump_message(m)
        if d["role"] == "assistant" and d.get("tool_calls"):
            for c in d["tool_calls"]:
                call_args[c["id"]] = (c["function"]["name"], c["function"]["arguments"])
        elif d["role"] == "tool":
            fn, args = call_args.get(d.get("tool_call_id"), ("unknown_tool", "{}"))
            digest_lines.append(f"{fn}({args}) -> {d['content']}")
        elif d["role"] == "assistant" and str(d.get("content", "")).startswith("[compacted"):
            # An already-compacted digest from a prior pass — it's pre-labeled
            # text, not a tool_calls/tool pair, so carry it forward as-is
            # instead of trying to re-derive attribution that no longer exists.
            digest_lines.append(d["content"].split("] ", 1)[-1])

    digest = {
        "role": "assistant",
        "content": (
            "[compacted earlier tool results — raw messages dropped to save "
            "context, but these facts still hold] " + "; ".join(digest_lines)
        ),
    }

    if verbose:
        print(f"--- compacting history: {len(middle)} raw message(s) -> 1 digest message ---")
        print("   ", digest["content"])

    return head + [digest] + tail
```

The subtle part is the boundary walk. The OpenAI wire format pairs every `tool` result message with the assistant `tool_calls` message that requested it, by id. A naive `messages[:-keep_recent]` cut can strand a tool result in the kept tail while its originating request gets summarized away, and the API, or a later compaction pass, no longer knows what that orphaned result was answering. The `_is_orphaned` walk moves the split left until it sits on a clean seam. Message-list surgery always has this property: the list looks like an array, but it has referential structure, and edits must respect the pairs.

The loop invokes it with a threshold:

```python
# WHERE: replace run_agent's signature, and insert the compaction check as
# the FIRST two lines inside the `for turn in range(max_turns):` loop
def run_agent(user_prompt: str, max_turns: int = 5, verbose: bool = True,
              compact_after: int = 6) -> str:
    # ... messages setup as before ...
    for turn in range(max_turns):
        if len(messages) > compact_after:
            messages = _compact_history(messages, verbose=verbose)
        # ... rest of the loop unchanged ...
```

Compaction is a trade, not a win. The digest keeps the facts (`search_knowledge_base({"query": ...}) -> {...}`) but drops exact wording, and a model that needed the exact wording of an old result will now do worse. `keep_recent=4` and `compact_after=6` are reasonable defaults for this system's conversation lengths, not universal constants.

## Run It

Ask the agent something the handbook actually answers:

```
USER: How many PTO days do new employees get?
```

Watch the trace: an ingest line on first run (then the skip message forever after), a `search_knowledge_base` call, three chunks back with `p.NN` sources and similarity scores, then an answer citing the real accrual numbers from the real document.

Then ask it something the handbook *doesn't* answer, and watch closely:

```
USER: What's our company's policy on parental leave?
```

The Marlow & Sage handbook has no parental leave policy. But `search_knowledge_base` returns the top 3 nearest chunks *no matter what*, so the model receives three semi-related passages about other kinds of leave, each arriving with the same authority as a real answer. What the model does with that temptation, sometimes admirable, sometimes an invented policy with a straight face, is the opening problem of Part 2.

## The Complete File: agentic_demo.py, as of Chapter 4

```python
"""
Minimal agentic loop using the OpenAI API's function/tool calling.

An "agent" is just a loop: send the model a prompt plus a list of tools,
execute the tool calls it requests, feed results back, repeat until it
answers in text. This version adds real retrieval: search_knowledge_base
is backed by a PDF chunked and embedded into a persistent Chroma vector
database. First run ingests the PDF (embeds every chunk — can take a
minute); later runs detect the file hasn't changed and skip straight to
querying.

Setup:
    pip install openai chromadb pypdf
    ollama pull qwen3.5:9b          # or qwen3.5:9b-mlx on Apple Silicon
    ollama pull nomic-embed-text    # embedding model, for search_knowledge_base
    ollama serve                    # if not already running
    python agentic_demo.py
"""

import hashlib
import json
import operator
import os
import re

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen3.5:9b"   # on Apple Silicon: "qwen3.5:9b-mlx"


# ---------------------------------------------------------------------------
# 1. Tools: real Python functions the model is allowed to invoke.
# ---------------------------------------------------------------------------

def calculate(expression: str) -> dict:
    """Evaluate a basic arithmetic expression safely (no eval of arbitrary code)."""
    import ast  # operator is already imported at module scope

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def compare(a: float, b: float, operator_: str) -> dict:
    """Compare two numbers so the model never has to judge magnitude itself."""
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if operator_ not in ops:
        return {"error": f"unsupported operator {operator_!r}, use one of {list(ops)}"}
    return {"result": ops[operator_](a, b)}


# ---------------------------------------------------------------------------
# Real retrieval: chunk a PDF, embed each chunk with Ollama, store/query in
# a persistent Chroma vector database.
# ---------------------------------------------------------------------------

EMBED_MODEL = "nomic-embed-text"  # pull once: `ollama pull nomic-embed-text`
KB_PDF_PATH = "Marlow_and_Sage_Handbook.pdf"
CHROMA_DB_DIR = "./chroma_db"     # persisted on disk — ingest once, reuse after
COLLECTION_NAME = "handbook"
CHUNK_SIZE = 1000                 # target characters per chunk
CHUNK_OVERLAP = 150               # characters carried into the next chunk


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Wraps Ollama's OpenAI-compatible /v1/embeddings endpoint so Chroma can
    call it like any other embedding function. Chroma calls this both when
    documents are added and when a query runs — using the same function for
    both is what keeps them comparable. Embedding docs with one model/function
    and queries with a different one is a classic, hard-to-spot RAG bug: the
    two vector spaces don't actually line up, so similarity scores become
    meaningless even though nothing throws an error.
    """

    def __init__(self) -> None:
        pass  # no state — silences chromadb's "does not implement __init__" deprecation warning

    def __call__(self, input: Documents) -> Embeddings:
        return [
            client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
            for text in input
        ]


def _extract_pdf_pages(pdf_path: str) -> list:
    """Return [(page_number, cleaned_text), ...], stripping the running
    header every page repeats so it doesn't get embedded into — and
    pollute — every chunk.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(
            r"^\s*\d*\s*MARLOW & SAGE BOUTIQUE\s*\W?\s*EMPLOYEE HANDBOOK\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if text:
            pages.append((i, text))
    return pages


def _chunk_text(pages: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Greedily pack paragraphs into ~chunk_size-character chunks tagged with
    the page they came from, keeping whole paragraphs together where
    possible rather than cutting mid-sentence. Carries the tail of one chunk
    into the next (`overlap`) so a fact split across a chunk boundary isn't
    orphaned on one side of it.
    """
    chunks = []
    for page_num, text in pages:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 1 > chunk_size:
                chunks.append({"text": buf, "page": page_num})
                buf = buf[-overlap:] if overlap else ""
            buf = (buf + " " + para).strip()
        if buf:
            chunks.append({"text": buf, "page": page_num})

    return chunks


_kb_collection = None


def ingest_pdf(pdf_path: str = KB_PDF_PATH, force: bool = False):
    """Chunk `pdf_path`, embed every chunk via Ollama, and load it into a
    persistent Chroma collection on disk at CHROMA_DB_DIR. Skips
    re-ingestion if this exact file (by content hash) is already indexed.
    Pass force=True to rebuild anyway.
    """
    global _kb_collection  # cache the handle so a second caller doesn't re-open (and re-log) ingestion
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    collection = None
    already_current = False
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME, embedding_function=OllamaEmbeddingFunction()
        )
        existing = collection.get(ids=["_source_hash"])
        already_current = bool(existing["ids"]) and existing["metadatas"][0].get("hash") == content_hash
    except Exception:
        collection = None

    if collection is not None and already_current and not force:
        print(f"[ingest] '{pdf_path}' already indexed (hash {content_hash}) — skipping re-ingestion.")
        _kb_collection = collection
        return collection

    print(f"[ingest] (re)indexing '{pdf_path}' — embeds every chunk, may take a minute...")
    # Delete unconditionally and swallow "doesn't exist" errors —
    # get_collection can fail for reasons other than "not created yet"
    # (e.g. an embedding function config mismatch after a library upgrade),
    # which would leave collection=None even though a collection with this
    # name already exists on disk.
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, embedding_function=OllamaEmbeddingFunction()
    )

    pages = _extract_pdf_pages(pdf_path)
    chunks = _chunk_text(pages)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": f"{os.path.basename(pdf_path)} p.{c['page']}", "type": "chunk"}
            for c in chunks
        ],
    )
    collection.add(
        ids=["_source_hash"],
        documents=["source file fingerprint"],
        metadatas=[{"hash": content_hash, "type": "sentinel"}],
    )

    print(f"[ingest] indexed {len(chunks)} chunks from {len(pages)} pages.")
    _kb_collection = collection
    return collection


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """Real semantic search: query the persistent Chroma collection, which
    embeds `query` via the same OllamaEmbeddingFunction used at ingest time
    and returns the nearest chunks by cosine distance. Lazily ingests the
    PDF on first call in a process.
    """
    global _kb_collection
    if _kb_collection is None:
        _kb_collection = ingest_pdf()

    result = _kb_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"type": "chunk"},  # exclude the sentinel record
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    results = [
        {"source": meta["source"], "text": doc, "similarity": round(1 - dist, 3)}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    return {"results": results}


TOOL_IMPLEMENTATIONS = {
    "calculate": calculate,
    "compare": compare,
    "search_knowledge_base": search_knowledge_base,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. '(3 + 5) * 2'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": (
                "Compare two numbers with an operator and return true/false. "
                "Use this for ANY magnitude judgment — 'is X more than Y', "
                "'is X at least Y', 'is X equal to Y', etc. Do not judge "
                "comparisons yourself; always call this tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operator_": {
                        "type": "string",
                        "enum": [">", "<", ">=", "<=", "==", "!="],
                        "description": "Comparison to apply as: a <operator_> b",
                    },
                },
                "required": ["a", "b", "operator_"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base (employee handbook) for "
                "passages relevant to a question. Returns ranked chunks with "
                "their source. Use this for ANY question about company policy "
                "— remote work, PTO, expenses, etc. — before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'remote work days allowed'"},
                    "top_k": {"type": "integer", "description": "How many chunks to return (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2. The agent loop.
# ---------------------------------------------------------------------------

def _dump_message(m) -> dict:
    """Normalize a message (dict or SDK object) into a plain dict for printing."""
    if isinstance(m, dict):
        return m
    d = {"role": m.role, "content": m.content}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = [
            {"id": c.id, "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls
        ]
    return d


def _compact_history(messages: list, keep_recent: int = 4, verbose: bool = True) -> list:
    """Collapse older tool-call/tool-result exchanges into one short digest
    message, instead of resending every raw message forever.

    Chat completions is stateless — the full `messages` list gets resent
    every turn, so payload only grows. Keep the system prompt and the
    original user question verbatim, keep the most recent `keep_recent`
    messages verbatim, and boil everything older down into one compact
    summary line per tool call. This is a simplification, not free — the
    model loses exact wording of older tool results.
    """
    if len(messages) <= 2 + keep_recent:
        return messages  # nothing old enough to compact yet

    head = messages[:2]  # system + original user prompt

    # Never let the split point strand a tool-result message whose
    # originating assistant tool_calls message is on the other side of the
    # cut — walk the boundary left until it lands on a clean seam.
    split = len(messages) - keep_recent

    def _call_ids_before(idx):
        ids = set()
        for j in range(2, idx):
            d = _dump_message(messages[j])
            if d["role"] == "assistant":
                for c in d.get("tool_calls", []) or []:
                    ids.add(c["id"])
        return ids

    def _is_orphaned(idx):
        d = _dump_message(messages[idx])
        return d["role"] == "tool" and d.get("tool_call_id") in _call_ids_before(idx)

    while split > 2 and _is_orphaned(split):
        split -= 1

    tail = messages[split:]
    middle = messages[2:split]

    if not middle:
        return messages  # nothing can be safely compacted yet without splitting a pair

    call_args = {}
    digest_lines = []
    for m in middle:
        d = _dump_message(m)
        if d["role"] == "assistant" and d.get("tool_calls"):
            for c in d["tool_calls"]:
                call_args[c["id"]] = (c["function"]["name"], c["function"]["arguments"])
        elif d["role"] == "tool":
            fn, args = call_args.get(d.get("tool_call_id"), ("unknown_tool", "{}"))
            digest_lines.append(f"{fn}({args}) -> {d['content']}")
        elif d["role"] == "assistant" and str(d.get("content", "")).startswith("[compacted"):
            digest_lines.append(d["content"].split("] ", 1)[-1])

    digest = {
        "role": "assistant",
        "content": (
            "[compacted earlier tool results — raw messages dropped to save "
            "context, but these facts still hold] " + "; ".join(digest_lines)
        ),
    }

    if verbose:
        print(f"--- compacting history: {len(middle)} raw message(s) -> 1 digest message ---")
        print("   ", digest["content"])

    return head + [digest] + tail


def run_agent(user_prompt: str, max_turns: int = 5, verbose: bool = True,
              compact_after: int = 6) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Use tools when they help answer the question. "
            "Never trust your own math — if a calculation is needed, always call the "
            "calculate tool for it. Never judge a comparison yourself either — for "
            "anything like 'is X more than Y' or 'is X at least Y', always call the "
            "compare tool and use its result, rather than deciding in your head. "
            "For any question about company policy, benefits, or the employee "
            "handbook, always call search_knowledge_base first — do not answer "
            "from your own memory. Only state facts that appear in the returned "
            "chunks, and mention which source each fact came from. If the search "
            "results don't contain the answer, say you don't have that "
            "information rather than guessing."
        )},
        {"role": "user", "content": user_prompt},
    ]

    for turn in range(max_turns):
        if len(messages) > compact_after:
            messages = _compact_history(messages, verbose=verbose)

        if verbose:
            print(f"\n--- turn {turn}: request ---")
            print(f"  sending {len(messages)} message(s):")
            for m in messages:
                print("   ", json.dumps(_dump_message(m)))

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        message = response.choices[0].message

        if verbose:
            print(f"--- turn {turn}: raw response message ---")
            print("   ", json.dumps(_dump_message(message)))

        # Case 1: model is done — it returned a normal text answer.
        if not message.tool_calls:
            if verbose:
                print(f"--- turn {turn}: no tool_calls -> final answer, loop ends ---")
            return message.content

        # Case 2: model wants to call one or more tools.
        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            impl = TOOL_IMPLEMENTATIONS.get(name)
            result = impl(**args) if impl else {"error": f"unknown tool {name}"}

            if verbose:
                print(f"--- turn {turn}: executing tool ---")
                print(f"    {name}({args}) -> {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Hit max_turns without a final answer."


if __name__ == "__main__":
    # Ingest explicitly before the loop so ingest-time output doesn't get
    # interleaved with the agent trace.
    ingest_pdf()

    prompts = [
        "How many PTO days do new employees get?",
        "What's our company's policy on parental leave?",  # not in this handbook — watch what happens
    ]
    for p in prompts:
        print(f"\nUSER: {p}")
        print(f"AGENT: {run_agent(p)}")
```

Part 1 is done. You have an agent with real tools and a real knowledge base, and you've already seen the crack in it: three chunks come back whether or not the document has anything to say. Part 2 is about that crack, and the family of failures that crawl out of it.
