# Chapter 15: Going Further

By the end of this chapter you'll know exactly what changes when this system leaves your laptop, three lines, and what to reach for next in each direction the book deliberately didn't go. This is a chapter of doors, each opened far enough to see through.

## Swapping in a Cloud Model

The promise from the front matter, kept literally. Everything in this book runs through an OpenAI-compatible client pointed at Ollama, which means pointing it somewhere else is a configuration change, not a rewrite.

OpenRouter is the lowest-friction first step, one API fronting most major hosted models. The change, in its entirety:

```python
# WHERE: replace the two lines at the top of agentic_demo.py
#   client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#   MODEL = "qwen3.5:9b"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],   # a real key now — never hardcode it
)
MODEL = "<model string from OpenRouter's catalog>"
```

That's the whole migration. Take inventory of what rides along untouched, because the list is the payoff of every architectural decision since Chapter 1: the loop, all three local tools, the MCP server and its discovery and dispatch, the compaction logic, both input guardrails, the citation verifier, the termination backstop, and the calibrated retrieval threshold's *enforcement*. Azure OpenAI and Bedrock work the same way with their OpenAI-compatible endpoints; same three lines, different values.

Two genuine caveats, so the three-line story stays honest.

First, the embedding side. `OllamaEmbeddingFunction` still points at local Ollama, and that's a feature: your document embeddings, your vector store, and your calibrated threshold all stay valid, and your corpus never leaves the machine. Chat traffic goes to the cloud; retrieval stays home. It's a genuinely good hybrid. If you *do* move embeddings to a hosted model, you've changed vector spaces: re-ingest the corpus (bump `CHUNK_LOGIC_VERSION`; that's what it's for) and rerun `calibrate_similarity_threshold()`, because the old threshold describes a geometry that no longer exists.

Second, the meter is back. Every turn resends the conversation, every guardrail probe in your test prompts costs real money, and the experiment-freely economics that shaped this book stop applying. The workflow that follows is the one this book quietly prepared you for: develop and iterate against local models for free, then swap three lines for the deployment that needs frontier quality. Keep the local config in a comment. You'll toggle back constantly.

And per Chapter 10: the guardrails all stay. A better model makes the flags rarer, not the checks unnecessary.

## More MCP Servers

One server taught the pattern; the pattern was built for many. `_discover_mcp_tools` already returns whatever a server offers, and the merge line already concatenates lists. Generalizing from one server to N is bookkeeping: hold several `(client, schemas, names)` sets, merge all the schemas, and route each call by whichever names-set claims it.

The interesting problems at N servers are architect problems, not plumbing. Name collisions: two servers both exposing `search`, disambiguated by prefixing or by curating which tools you forward to the model at all. Trust tiers, Chapter 11's warning made operational: a server you wrote, a vendor's server, and something found on a community registry do not deserve the same standing, and description text from low-trust servers lands in your model's context, which is an injection surface. And context budget: every discovered tool's schema ships with every model call, so connecting to a server with forty tools taxes every turn whether or not the model uses any of them. Curate; don't just connect.

## Persisting Conversations

`run_agent` starts fresh every call because `messages` is built inside it. Real assistants remember. The message list is plain JSON-serializable data, so persistence is structurally easy, load a saved list instead of building a new one, but two design points deserve care. Compaction becomes essential rather than optional, because a conversation that lives for weeks will not fit any context window; you'll compact on load as well as in the loop. And guardrail state must persist with the transcript: `retrieved_pages` in particular, because a resumed conversation that forgot what it retrieved will flag every citation from yesterday as a hallucination.

## Streaming

This book printed answers when they finished, which is right for reading traces and wrong for users watching a spinner. The API streams tokens as they generate; the wrinkle worth knowing before you start is that *tool calls stream too*, arriving as incremental argument fragments you assemble before dispatching. The loop's logic doesn't change, but the tidy "message in, message out" turn structure becomes an accumulation state machine. Budget a real afternoon, not an hour.

## Multi-Agent Architectures

The current frontier, and the reason this book's mental model matters more there, not less. Strip the vocabulary away and a "multi-agent system" is agents-as-tools: a coordinator loop whose tool schemas happen to invoke other loops. A researcher agent that delegates to a summarizer is `run_agent` calling something that calls `run_agent`, one more entry in a dispatch table, one more seam. Which means everything in Part 2 applies at every level: each inner agent needs its own termination backstop (`max_turns` compounds across nesting), the coordinator needs discipline about trusting sub-agent output (an inner agent's answer is a tool result, exactly as unverified as any other), and the composition lesson from Chapter 8 applies with interest, because now entire *agents* are the components interacting in untested ways. If you evaluate a multi-agent framework, evaluate it the way Chapter 2 taught you to evaluate everything: find the loop, find the seam, find what catches the model when it ignores an instruction.

## Frameworks: When to Adopt One, and How to Read One

This book skipped frameworks on purpose, and it's worth being clear that the reason was visibility, not disdain. You built the loop bare so nothing could hide from you. That job is done. Whether to adopt a framework now is an ordinary engineering decision, and you're unusually well equipped to make it, because every agent framework on the market contains the loop you wrote, wearing different clothes.

That's the evaluation lens, and it beats any feature-matrix comparison you'll find online. Reading a framework's docs, ask the questions this book trained into you. Where's the loop, and can I see a turn? Where's the seam, the moment between "the model asked for a tool call" and "run it," and can I put my own code there? When the model ignores an instruction, what catches it, structure or a politely worded request? And what exactly is in the context each turn, since somebody's compaction policy is running whether or not the docs mention one?

Applied to the current landscape, as of this writing:

**LangGraph** (the agent-focused core of the LangChain ecosystem) makes the loop explicit as a state graph: your `for turn in range(max_turns):` becomes nodes and conditional edges, which pays off when the workflow genuinely branches, needs checkpointing mid-run, or requires human approval steps between turns. Of the major frameworks it's the one whose abstraction is most honest about being a loop. LangChain proper is the broader toolkit around it; LangGraph is the part that competes with what you built.

**CrewAI** is role-based multi-agent: define a crew of agents with goals and tool sets and let them collaborate. Reread the multi-agent section above and you can see through it immediately: agents-as-tools with personas, fast to prototype, and every question about inner-loop termination and sub-agent trust still applies, just under friendlier vocabulary.

**Microsoft's agent tooling** (AutoGen, and Semantic Kernel converging with it into the Microsoft Agent Framework) targets multi-agent conversation patterns and enterprise .NET/Azure shops. If your organization is a Microsoft shop, this is the gravity well you'll be evaluating whether you like it or not, so evaluate it with the lens rather than the deck.

**Pydantic AI** is the lightweight one, typed, schema-validated agent responses. You already believe in its core idea: it's Chapter 13's normalize-at-the-boundary principle productized, with `_call_mcp_tool`'s "whatever happens, the caller sees the promised shape" contract enforced by types everywhere.

**The vendor SDKs** (OpenAI's Agents SDK, Anthropic's Claude Agent SDK) are the loop plus handoffs plus their platform's tooling, thinner than the frameworks above and correspondingly easier to see through. **LlamaIndex** sits beside all of this as the retrieval specialist: Chapter 4 as a mature product, with fifty chunking strategies where you wrote one.

What a good framework genuinely buys you is the commodity plumbing this book waved at and moved past: retries and backoff, streaming assembly, conversation persistence, checkpointing, provider abstraction, and observability hooks, each solved once by someone else instead of again by you. That last one deserves its own sentence: "the trace is the product" grows up into tracing platforms (LangSmith, Langfuse, and OpenTelemetry-based tooling), and once an agent has users, searchable production traces stop being optional. There's a parallel ecosystem for the guardrail layer too (NVIDIA's NeMo Guardrails, Guardrails AI, Meta's Llama Guard as a sibling of Granite Guardian); evaluate those exactly the way Chapter 9 taught you, by asking what's deterministic, what's a model judging, and what happens when the judge is unavailable.

What a framework costs you is the thing this book just spent fifteen chapters giving you: nothing hides in a file you wrote, and something always hides in a dependency. You'll debug through someone else's abstraction at the worst possible moment, and agent frameworks are young enough that major-version churn is a real line item.

So the heuristic: adopt a framework when your needs are boring, in the best sense, checkpointing, streaming, tracing, retries, the things every agent needs and none differentiates on. Stay bare, or stay thin, when the control flow *is* the product. And in either case, keep the guardrails yours. A framework can host your guardrails, but Part 2's real output was a ledger of which judgments your system takes away from the model and what catches each failure; that ledger is specific to your risk profile, and no dependency ships it. Framework names will have churned by the time you read this. The four questions won't.

## Evals: Growing the Ten-Query Habit

You already built an eval harness. It's small, but it's real: Chapter 7's `EVAL_SET` is labeled cases scored by the live pipeline with misclassifications flagged, and the designed probes in your standard run (the crypto question for the backstop, the injection string for pass 1, the pirate for pass 2) are behavioral test cases in everything but name. The production discipline is that same habit at scale, run automatically: suites of labeled inputs with expected behaviors, executed against the real system on every change, because a prompt edit is a code change and deserves regression tests exactly the way code does. Chapter 5 showed you why this matters more for agents than for ordinary software: fixing one unstated-negative case can quietly un-fix another, and without a suite you only find out from a user.

Tooling exists so you don't build the runner yourself: promptfoo (config-driven suites, comfortable in CI) and DeepEval (pytest-style, evals as ordinary test functions) are the established starting points, and the tracing platforms from the previous section score live production traffic against the same kinds of checks, which closes the loop between "passes the suite" and "behaves in the wild." One caution carries over intact from Chapter 9: many eval metrics are LLM-as-judge under the hood, with everything that implies, nondeterminism, cost per scored case, and a judge that can itself be wrong. Prefer deterministic assertions wherever the behavior allows one ("the answer contains a `p.NN` citation," "the blocked-prompt response names the layer that fired"), and spend model-judged scoring on the questions that genuinely require judging meaning. That's not new advice. It's the Chapter 5 principle, applied to the tests instead of the system.

## Where to Keep Reading

Primary sources, deliberately few. The MCP specification and SDK documentation live at **modelcontextprotocol.io**, and the spec is short enough to actually read; after Chapter 13, you'll recognize everything in it. The Ollama model library at **ollama.com/library** is where new local models appear, tool-calling support is called out per model, and both your chat and embedding models will be superseded by something better within months; the swap procedure is this chapter's first section, minus the API key. IBM's Granite Guardian model cards document the full criteria list beyond `jailbreak`, which is the natural next step for the Chapter 9 layer.

## The Code Changed This Chapter

Nothing, unless you counted the three commented lines of the OpenRouter swap, and those are a door, not an edit. The system you finished in Chapter 13 is the system this book set out to build.

What you actually built is a set of reflexes. An agent is a loop you can read. A tool is a judgment taken out of the model's hands. A guardrail is code where a request would fail silently. A protocol is a seam you already had, standardized. Every system you evaluate from here, every framework demo, every vendor deck, every incident review, will sort itself against those four sentences faster than any slide can talk. That was the point.
