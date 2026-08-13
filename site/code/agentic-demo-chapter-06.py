# === Book checkpoint: complete file as of Chapter 6 ===
# To run: copy/rename to agentic_demo.py (see code/README.md)

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



# --- Output guardrail: citation verification -------------------------------

def _extract_cited_pages(text: str) -> set:
    """Find 'p.NN' style page citations anywhere in text."""
    return {int(n) for n in re.findall(r"\bp\.\s*(\d+)\b", text)}


def _check_citation_grounding(answer: str, retrieved_pages: set) -> list:
    """Return any page numbers the answer cites that were never actually
    retrieved this conversation — a hallucinated or misremembered citation.
    """
    cited = _extract_cited_pages(answer)
    return sorted(cited - retrieved_pages)


def run_agent(user_prompt: str, max_turns: int = 5, verbose: bool = True,
              compact_after: int = 6) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Use tools when they help answer the question. "
            "Never trust your own math — if a calculation is needed, always call the "
            "calculate tool for it. Never judge a comparison yourself either — for "
            "anything like 'is X more than Y' or 'is X at least Y', always call the "
            "compare tool and use its result, rather than deciding in your head. "
            "Do not state any numeric claim, ratio, or comparison in your final "
            "answer unless it came directly from a tool result earlier in this "
            "conversation — if you haven't called a tool for a number, leave it out. "
            "For any question about company policy, benefits, or the employee "
            "handbook, always call search_knowledge_base first — do not answer "
            "from your own memory. Only state facts that appear in the returned "
            "chunks, and mention which source each fact came from. If the search "
            "results don't contain the answer, say you don't have that "
            "information rather than guessing. This is a general principle, "
            "not limited to one example: a retrieved chunk simply not "
            "mentioning something is NEVER grounds to state a negative or "
            "exclusion as fact. This applies to every kind of unstated claim "
            "— tenure-based changes, eligibility restrictions (e.g. don't "
            "say part-time employees are excluded from a benefit unless a "
            "chunk explicitly excludes them), exceptions, caps, anything. "
            "Before adding any qualifier, aside, or 'note:' to your answer, "
            "check it is explicitly stated in a retrieved chunk — if it "
            "isn't, cut it, even if it seems like a reasonable inference. "
            "Citation format is strict: every fact you state from the "
            "handbook — including facts you're citing only to explain that "
            "they DON'T answer the question — must be tagged inline with the "
            "literal page citation exactly as it appears in the chunk's "
            "source field, e.g. 'p.17'. Section numbers like 'Section 5.3' "
            "are not a substitute for the page citation; include the page "
            "citation every time, even alongside a section number."
        )},
        {"role": "user", "content": user_prompt},
    ]

    retrieved_pages = set()  # pages actually returned by search_knowledge_base this run

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

            # Output guardrail: does every page citation in the answer trace
            # back to something actually retrieved this conversation? And,
            # separately — if real content WAS retrieved, does the answer
            # cite anything at all? A "passed" check that never had a
            # citation to verify is a false sense of security, not a real
            # pass.
            cited_pages = _extract_cited_pages(message.content or "")
            unverified = _check_citation_grounding(message.content or "", retrieved_pages)
            if unverified:
                print(
                    f"--- GUARDRAIL: answer cites page(s) {unverified} that were "
                    f"never retrieved this conversation (retrieved: {sorted(retrieved_pages) or 'none'}) ---"
                )
            elif retrieved_pages and not cited_pages:
                print(
                    f"--- GUARDRAIL: real content was retrieved (pages {sorted(retrieved_pages)}) "
                    f"but the answer contains zero page citations — cannot verify grounding ---"
                )
            elif verbose:
                print(f"--- GUARDRAIL: citation check passed (retrieved: {sorted(retrieved_pages) or 'none'}) ---")

            return message.content

        # Case 2: model wants to call one or more tools.
        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            impl = TOOL_IMPLEMENTATIONS.get(name)
            result = impl(**args) if impl else {"error": f"unknown tool {name}"}

            if name == "search_knowledge_base":
                found = result.get("results", [])
                for r in found:
                    retrieved_pages |= _extract_cited_pages(r.get("source", ""))

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
