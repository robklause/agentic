# Appendix B: Troubleshooting

Organized by symptom, because that's what you have when you arrive here. Each entry: what you're seeing, what's actually wrong, what to do. Chapter references point to the fuller explanation.

## Installation and Environment

**Symptom: `pip install mcp` fails with "Could not find a version that satisfies the requirement" / "No matching distribution found."**
It reads like a network problem or a typo. It's almost certainly your Python version: the `mcp` SDK requires Python 3.10+, and pip on an older interpreter reports "no distribution" instead of "Python too old." Run `python3 --version`; if you see 3.9.x, create your venv with a newer interpreter (`python3.12 -m venv .venv`) and reinstall. (Chapter 1)

**Symptom: packages installed fine, but the script uses the wrong Python or can't find them.**
The venv isn't active in this shell, or VS Code is pointed at a different interpreter. `source .venv/bin/activate`, and in VS Code run "Python: Select Interpreter" and pick the project's `.venv`. `which python` should answer inside the project folder.

## Ollama and Models

**Symptom: `Connection refused` or `Connection error` on any model call.**
Ollama isn't serving. Run `ollama serve` (if it's already running as a background service, this says the port's in use, which is fine). Confirm the endpoint matches the code: `localhost:11434`. (Chapter 1)

**Symptom: model-not-found error naming your chat model.**
The pull didn't finish or the tag doesn't match. `ollama list` shows what you have; compare the exact string against `MODEL`. Mac users who pulled `qwen3.5:9b-mlx` while the code says `qwen3.5:9b` (or the reverse) hit exactly this: the tag must match what you pulled, character for character. (Chapter 1)

**Symptom: the model answers in text but never calls tools, or emits tool-call-shaped prose instead of a real `tool_calls` response, or hallucinates argument JSON.**
Not every local model supports tool calling well, and the failure is behavioral, not an error. Before debugging your code, swap in a model explicitly documented as tool-call capable (`gemma4:12b` is a solid alternative) and compare. If gemma calls tools and your original model doesn't, the code was never the problem. (Chapters 1, 3)

**Symptom: everything works, but slowly, and the first handbook question takes a minute-plus.**
First-run ingestion embeds every chunk of the PDF; that's the printed "may take a minute" doing what it said. Subsequent runs print the "already indexed... skipping" line and go straight to querying. If ingestion runs on *every* start, see the cache entries below. (Chapter 4)

**Symptom: long pauses mid-run, especially between the Granite Guardian trace line and the first agent turn, or whenever the prompt type changes.**
Model swapping, not a hang. This book rotates three models, and on a machine without enough memory to keep them all resident (16GB is the practical squeeze point), Ollama evicts one to load another, and each swap costs seconds before the first token. The fix hierarchy: patience (it's a stall, not a failure); `USE_LLM_GUARD = False` to take the guard model out of the rotation; a smaller chat model; or more capable hardware, where the ordering is simply more RAM plus a dedicated GPU/APU/NPU beats unified memory beats CPU-only. On Apple Silicon, prefer the `-mlx` model variants. (Chapter 1)

## Retrieval and the Vector Store

**Symptom: similarity scores look plausible but retrieval results are consistently nonsense.**
Classic signature of mismatched embedding models: documents embedded with one model, queries with another, two vector spaces whose distances mean nothing, no error anywhere. In this codebase both sides route through `OllamaEmbeddingFunction`, so this typically appears after someone changes `EMBED_MODEL` without re-ingesting. Bump `CHUNK_LOGIC_VERSION` (or run `ingest_pdf(force=True)`) to rebuild, and recalibrate the threshold. (Chapters 4, 7)

**Symptom: you changed chunking logic (or `MIN_CHUNK_CHARS`, or the header-strip regex) and behavior didn't change at all.**
The stale-cache trap. The ingest cache is keyed on PDF bytes plus `CHUNK_LOGIC_VERSION`; if you changed logic without bumping the version, the cache correctly concluded nothing changed. Bump `CHUNK_LOGIC_VERSION` and rerun. (Chapter 8)

**Symptom: Chroma errors on startup after a `chromadb` upgrade, or `create_collection` complains the collection already exists.**
`get_collection` can fail for reasons other than absence (embedding-function config mismatch after a library upgrade is the known one), which is why `ingest_pdf` deletes unconditionally inside a try/except. If a truly wedged on-disk state persists, the blunt fix is safe: delete the `chroma_db/` directory entirely and let the next run re-ingest from the PDF. Nothing in it is the source of truth. (Chapter 4)

**Symptom: obviously-covered topics get rejected by the threshold, or obviously-absent topics sail through.**
The threshold is calibrated to one document, one chunking, one embedding model; change any of those and the old number describes a geometry that no longer exists. Rerun `calibrate_similarity_threshold()` and *read the printed table*: if labeled queries land on the wrong side, the classes overlap and the table shows exactly which ones. Add more labeled queries around the boundary. Also check your eval labels against the corpus (grep the extracted text); a mislabeled example poisons the midpoint. (Chapter 7)

**Symptom: a search returns a chunk that's just a title or header fragment, with a suspiciously high score.**
The junk-chunk artifact (a bare "EMPLOYEE HANDBOOK" fragment scored 0.827 in this system's testing). The `MIN_CHUNK_CHARS` filter exists for this; if you're seeing it, either the filter was removed, the cache is stale (see above), or your document produces junk longer than 60 characters, in which case raise the floor and bump the version. (Chapter 8)

## The Loop and Guardrails

**Symptom: "Hit max_turns without a final answer."**
From Chapter 8 on, this message should be unreachable: the final turn never offers tools, so the model must answer in text before the cap can expire. If you're seeing it, either you're running a pre-Chapter-8 checkpoint (expected there; it just means the turn budget ran out), or the final-turn condition didn't make it into your file. Check that the `force_text_only` line includes `or turn == max_turns - 1`. (Chapters 2, 8)

**Symptom: the agent's answer is "(the model returned an empty answer)".**
The guardrails worked; the model didn't. A text-only forced turn (backstop fired, or final turn) can make a small model emit an empty string, and this fallback string is what stops that from rendering as a blank line. Re-ask the question, and if it recurs often, that's a model-capability signal per Chapter 10, not a code bug: try a more capable chat model. On checkpoints before Chapter 8, the same event shows up as a literally empty `AGENT:` line. (Chapters 8, 10)

**Symptom: the citation guardrail flags "cites page(s) that were never retrieved" on answers that look right.**
Working as designed; the question is which side is wrong. If the cited page really contains the fact (check the PDF), the model likely knew the page from an earlier conversation or from a compacted digest while `retrieved_pages` was reset, which is the persistence caveat from Chapter 15. If the page is plausible-but-wrong, you just watched the guardrail do its job. (Chapter 6)

**Symptom: constant "zero page citations — cannot verify grounding" flags.**
The model isn't complying with the strict citation format, common with smaller models. Confirm the system prompt includes the full strict-format rule (with the `p.17` example, which does real work), and consider whether your model is up to instruction-following at this density; this flag firing often is itself a signal per Chapter 10. (Chapter 6)

**Symptom: legitimate prompts get blocked by the pattern screen.**
A false positive, the known cost of pass 1 (a question containing "ignore the store hours for a second" can trip the first pattern). The block message names the matched category; find the pattern, narrow it the way the "pretend" pattern was narrowed (require paired attack language), and let ambiguity fall through to Granite Guardian, which exists for exactly those cases. (Chapter 9)

**Symptom: "Granite Guardian check failed, failing open" on every prompt.**
The guard model isn't reachable: not pulled (`ollama pull granite4.1-guardian:8b`), or Ollama is down. The system is designed to keep working on the regex layer alone; set `USE_LLM_GUARD = False` to silence the warnings until you pull the model. (Chapter 9)

**Symptom: Guardian responds but never flags anything, or flags everything.**
Check the raw output the trace prints (`raw: ...`) against the parser's expectation: a literal `<score>yes</score>` or `<score>no</score>` tag. A different Guardian version with a different output shape will parse as never-flagged. Verify the tag against your live model's actual response and adjust the regex; the docs-versus-live-contract lesson, again. (Chapters 9, 12)

## MCP

**Symptom: you ran `python3 mcp_weather_server.py` and it just sits there doing nothing.**
Correct behavior. A stdio server blocks on stdin waiting for a client to speak the handshake; it will wait forever when run by hand. Ctrl+C to exit. No output plus no traceback equals a working server. (Chapter 12)

**Symptom: `agentic_demo.py` hangs at startup, before the `[mcp] connected` line.**
The client is waiting for a handshake that isn't coming. Usual causes, in order: `mcp_weather_server.py` isn't in the working directory you launched from (the launch recipe uses a relative path); the server crashes on import, so nothing is listening (run it by hand once, you should get silence, not a traceback); or the `command` in `MCP_SERVER_PARAMS` doesn't resolve, `python3` versus `python` on Windows being the common case. Fix by making the command explicit for your platform. (Chapters 12, 13)

**Symptom: MCP tool results arrive as escaped JSON strings inside strings, and the model mangles them.**
`structured_content` came back `None` and something upstream passed the raw text block through. A plain `-> dict` server return does not populate `structured_content`; only a Pydantic model return does. `_call_mcp_tool` handles this by parsing the text block; if you've written your own client path, add the same fallback. (Chapters 12, 13)

**Symptom: every MCP tool call is slow, and the process table shows Python processes churning.**
The respawn trap: something is opening a connection per call (typically `asyncio.run(...)` at the dispatch site) instead of one connection for the run. For a stdio server the connection owns the subprocess, so per-call connections mean per-call process spawns. Restructure to the single long-lived `async with` in `main`. (Chapter 13)

**Symptom: `RuntimeError: asyncio.run() cannot be called from a running event loop`.**
Something inside the async call stack (often a notebook, or a nested helper) called `asyncio.run` while `main` was already running under it. Inside async code, `await` things; `asyncio.run` belongs in exactly one place, the `__main__` entry point. (Chapter 13)

## When Nothing Here Fits

Turn `verbose` on and read the trace end to end before theorizing: the raw messages, the tool dispatches with their `(local)`/`(mcp)` labels, and the guardrail lines tell you what actually happened, which beats any hypothesis about what should have. Then diff your file against the nearest checkpoint listing (Chapters 3, 4, 9, 13). Ten minutes of diff has settled every "but I didn't change anything" in the history of software.
