# Appendix A: The Complete Code

The finished system is two files, and both appear in full, exactly once, at their final checkpoints:

- **`agentic_demo.py`** (final version): the Chapter 13 checkpoint listing, "The Complete File: agentic_demo.py, Final."
- **`mcp_weather_server.py`**: the Chapter 12 listing, "The Server." That listing is the entire file.

Intermediate checkpoints, if you need to reset to an earlier point in the book: Chapter 3 (first runnable loop, two tools), Chapter 4 (RAG and compaction, end of Part 1), and Chapter 9 (all guardrails, end of Part 2).

The book's companion `code/` directory carries all of this as runnable files, one per code-changing chapter (`agentic-demo-chapter-03.py` through `agentic-demo-chapter-13.py`, plus the server). Each was produced by applying that chapter's documented edits to the previous checkpoint, so consecutive files diff cleanly to show exactly what a chapter changed. See `code/README.md` for how to run any checkpoint and what each stage requires.

Rather than reprint seven hundred lines, this appendix is the map: every definition in the final `agentic_demo.py`, in file order, with what it does and the chapter that explains why it looks the way it does. Use it to navigate the code, to re-find the reasoning behind any piece, or as a checklist when adapting the architecture to your own system.

## agentic_demo.py, Top to Bottom

| # | Definition | What it is | Chapter |
|---|------------|-----------|---------|
| 1 | module docstring | The system's own summary: loop concept, guardrail layers in execution order, setup commands | 13 |
| 2 | imports | stdlib (`asyncio`, `hashlib`, `json`, `operator`, `os`, `re`), `chromadb`, `mcp` client pieces, `openai`, `pypdf` | 1, 4, 13 |
| 3 | `client`, `MODEL` | OpenAI-compatible client pointed at local Ollama; the chat model tag | 2 |
| 4 | `calculate(expression)` | Arithmetic tool; AST walk over an operator allowlist, never `eval`; errors returned as data | 3 |
| 5 | `compare(a, b, operator_)` | Magnitude-judgment tool; exists to close the gap `calculate` leaves (the model eyeballing 101 vs 204) | 3 |
| 6 | retrieval constants | `EMBED_MODEL`, `KB_PDF_PATH`, `CHROMA_DB_DIR`, `COLLECTION_NAME`, `CHUNK_SIZE` 1000, `CHUNK_OVERLAP` 150, `MIN_CHUNK_CHARS` 60, `CHUNK_LOGIC_VERSION` | 4, 8 |
| 7 | `OllamaEmbeddingFunction` | One embedding function for both documents and queries; prevents the mismatched-vector-space bug structurally | 4 |
| 8 | `_extract_pdf_pages(pdf_path)` | pypdf extraction, page numbers kept, per-page running header stripped | 4 |
| 9 | `_chunk_text(pages, ...)` | Paragraph-aware packing with overlap; ends with the minimum-length filter (the 0.827 junk-chunk fix) | 4, 8 |
| 10 | `_kb_collection` | Process-lifetime lazy handle to the Chroma collection | 4 |
| 11 | `ingest_pdf(pdf_path, force)` | Chunk, embed, store; hash-keyed cache including `CHUNK_LOGIC_VERSION`; sentinel record; unconditional delete-before-rebuild | 4, 8 |
| 12 | `EVAL_SET` | Ten labeled queries, five present/five absent, labels grep-verified against the corpus | 7 |
| 13 | `SIMILARITY_THRESHOLD` | Module global; `None` means the embedding guardrail is off | 7 |
| 14 | `calibrate_similarity_threshold(...)` | Scores the eval set through real retrieval; midpoint of the actual margin, honest fallback when classes overlap, prints the classification table | 7 |
| 15 | `search_knowledge_base(query, top_k)` | The retrieval tool; threshold filter converts weak matches into an explicit rejection note with scores | 4, 7 |
| 16 | `TOOL_IMPLEMENTATIONS` | Local dispatch dict, three entries; `get_weather` deliberately absent | 3, 4, 13 |
| 17 | `TOOL_SCHEMAS` | Hand-written JSON Schemas for the three local tools; description fields doing behavioral work | 3, 4 |
| 18 | `_dump_message(m)` | Normalizes dict-or-SDK-object messages for trace printing | 4 |
| 19 | `_compact_history(messages, ...)` | Digest-based history compaction; boundary walk that never orphans a tool result from its request | 4 |
| 20 | `_extract_cited_pages(text)` | The `p.NN` regex, used on both tool results and final answers so the two sets are comparable | 6 |
| 21 | `_check_citation_grounding(...)` | Set difference: cited pages minus retrieved pages | 6 |
| 22 | `INJECTION_PATTERNS` | Regex screen, category-labeled; the narrowed "pretend" pattern that spares the pirate | 9 |
| 23 | `_check_prompt_injection(...)` | Pass-1 input guardrail; runs on raw untrusted input before it joins any message list | 9 |
| 24 | `GUARD_MODEL`, `USE_LLM_GUARD` | Granite Guardian tag and the off-switch | 9 |
| 25 | `_check_prompt_injection_llm(...)` | Pass-2 input guardrail; criteria-as-system-message, `<score>` parse, fail-open | 9 |
| 26 | `MCP_SERVER_PARAMS` | Launch recipe for the weather server subprocess | 13 |
| 27 | `_discover_mcp_tools(mcp_client)` | `list_tools()` relabeled into OpenAI tool-schema shape, plus the names set the router needs | 13 |
| 28 | `_call_mcp_tool(mcp_client, name, args)` | Wire-call normalization: `is_error` to error dict, `structured_content` when present, text-block JSON parse otherwise | 12, 13 |
| 29 | `run_agent(...)` | The loop: input guardrails, system prompt, compaction, termination backstop, merged tool list, dual-dispatch router, citation check at exit | 2, and everything since |
| 30 | `main()` | Ingest, calibrate, one `async with` MCP connection, discovery, the seven demo prompts including three designed guardrail probes | 13 |
| 31 | `asyncio.run(main())` | Entry point | 13 |

## mcp_weather_server.py

| # | Definition | What it is | Chapter |
|---|------------|-----------|---------|
| 1 | module docstring | What the server stands in for; how to run it by hand and why it sits silent | 12 |
| 2 | `mcp = MCPServer(...)` | Server declaration with client-facing `instructions` | 12 |
| 3 | `FAKE_WEATHER` | Three cities of fake data (Austin 101 and sunny, Seattle 68 and cloudy, New York 84 and humid) | 12 |
| 4 | `get_weather(city)` | The tool; `@mcp.tool()` derives name, description, and schema from the function itself | 12 |
| 5 | `mcp.run()` | Stdio transport; blocks awaiting a client | 12 |

## The System Prompt, Rule by Rule

The final system prompt is one long string built up over five chapters. Its rules, in order of appearance, with the failure each one answers:

| Rule | Failure it addresses | Chapter |
|------|---------------------|---------|
| Always use `calculate` for math | Fluent wrong arithmetic | 3 |
| Always use `compare` for magnitude judgments | Post-calculation eyeballing | 3 |
| No numeric claim without a tool result behind it | Freehand numbers garnishing grounded answers | 5 |
| Always search before answering policy questions | Fluent fabrication from training memory | 4 |
| Only state facts from retrieved chunks; name sources | Ungrounded additions | 4 |
| Say "I don't have that" rather than guess | Plausible invention on empty retrievals | 4 |
| Silence is never a stated negative (general rule, with categories) | Absence of evidence becoming "confirmed no" | 5 |
| Check every qualifier against a chunk before including it | The "reasonable inference" tacked onto a real answer | 5 |
| Strict `p.NN` citation format, even for negative findings | Unverifiable citation styles (`Section 5.3`), and answers the citation guardrail can't check | 6 |
| Stop searching after two empty results | The infinite reformulation loop | 8 |

Every rule in the table is a request; the ones that mattered most are backed by code (the citation verifier, the termination backstop). That pairing, and the judgment about which rules need backing, is Chapter 5's principle in table form, and it's the single most reusable artifact in this appendix.
