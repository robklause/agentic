# Agentic AI from Scratch

### Build a Local Agent with Ollama and Python, from the Loop to Guardrails to MCP

---

## Introduction

This book exists because I wanted to understand agentic AI. Not the conference-keynote version, the behind-the-scenes, nuts-and-bolts version: what exactly is the agentic process, and how does it all come together? The most basic premise of agentic AI, "the loop." Tooling. Vector embeddings. All the way through implementing guardrails and MCP.

I'm a hands-on learner. I prefer to learn by doing. So one evening I sat down with Claude Cowork (Sonnet 5: medium), set up a project, and gave it a prompt: "I want to understand agentic AI. I want to use local models with Ollama so I don't burn through credits. Teach me by walking me through Python code examples."

From there it was back-and-forth question-and-answer sessions, with code. Curiosity drove the path. Day 1 was the basic agentic loop. Day 2, I came back with "now tell me about RAG, and how it's different with agents." Day 3: "what about guardrails? Can I do more than just regex, like use Granite Guardian?" And day 4: "now add MCP." (Not the exact prompts. I'm paraphrasing.)

When all was said and done, I had a solid understanding, a decent amount of sample code, and working examples. On day 5 came the thought that this might be helpful to other solutions architects and software engineers looking for exactly the introduction I'd just given myself.

Right about then, Anthropic gave me $100 in free credits to try out Fable 5. So I did the immediately obvious thing: I had it build a crypto trading bot. Then, with the $98 remaining (and $0.15 in crypto earnings), I worked with it to turn all of our lessons and time spent together into the book you're reading now.

Enjoy.

---

## About This Book

An agent is just a loop. You send a model a prompt plus a list of tools it's allowed to call. The model doesn't execute anything itself. It replies with a request like "call get_weather(city='Austin')". Your code runs that function, feeds the result back, and the model continues. It keeps going until it produces a plain text answer instead of another tool call.

That's the entire mechanism behind "agentic" behavior. There's no magic beyond structured turn-taking.

This book teaches that mechanism by building one real system, end to end, on your own laptop. Not a survey of frameworks. Not a LangChain tutorial. One Python file that grows, chapter by chapter, from a bare loop into an agent that answers questions from a real PDF, refuses to hallucinate citations, screens its own inputs, and pulls one of its tools from a live MCP server running in a separate process.

You'll write every line of it. By the end, when a vendor says "our platform handles agent orchestration," you'll know exactly what's behind that sentence, because you'll have built it.

## Who This Is For

Solution architects, cloud architects, and technical leads who understand distributed systems, APIs, and infrastructure but haven't built an LLM agent from scratch.

You should be comfortable reading Python and working a command line. That's the whole prerequisite list. This book doesn't assume you know what an embedding is, what "tool calling" means on the wire, or how a context window behaves. Every LLM-specific concept gets defined the first time it matters, tied to the code in front of you, not as an abstract detour.

What this book won't do is explain what an API is, what a virtual environment is for, or why you'd want version control. You already know.

## Should You Use a Framework?

Yes. When it's time to build something real, use a framework.

That's worth saying up front, before you spend a few evenings building an agent by hand. LangGraph, the vendor SDKs, and their neighbors solve problems you don't want to solve twice: retries and backoff, streaming, checkpointing, conversation persistence, provider abstraction, observability. None of that is where your product differentiates, and all of it is tedious to get right.

This book is not an argument against those tools, and the code in it is not a production foundation. Don't ship it.

So why build the thing by hand at all?

Because a framework gives you plumbing, not judgment. Everything in Part 2 stays your problem no matter which framework you pick. No library decides what your retrieval threshold should be, or notices that your eval set is measuring your assumptions rather than your corpus. None of them will tell you that a two-word chunk is quietly disarming a guardrail three layers above it, that a document's silence is not the same as a "no," or that two individually correct protections just composed into a new failure. Those are design decisions, and they stay yours.

There's a second payoff, and it shows up the first time you read a framework's documentation after this. The abstractions stop being magic words. You know there's a loop in there. You know where the seam between "the model asked for a tool" and "run it" has to be. You know somebody's compaction policy is running whether the docs mention one or not. That's the difference between evaluating a tool and being sold one.

Some of what's here isn't really about agents at all. Semantic search behaves in ways worth seeing directly: how embeddings score things that merely sound related, why a similarity threshold has to be measured rather than guessed, what "no match" actually looks like in a coherent corpus. That knowledge outlives whatever framework is fashionable when you read this.

So: build it once, by hand, to see the mechanism. Then go use a framework, and be the person on the team who knows what it's doing. Chapter 15 comes back to this with a lens for evaluating specific ones.

## Why Local-First

Everything in this book runs on your own machine through Ollama, a free runtime that serves open-weight models on localhost. No API keys. No credits. No rate limits. No data leaving your laptop.

That matters for two reasons.

First, you can experiment freely. Agent development is iterative in a way that punishes metered APIs: you'll rerun the same loop dozens of times watching how one prompt change shifts behavior. Locally, a hundred runs cost the same as one. Nothing.

Second, and more important for an architect: running locally forces you to see everything. There's no hosted framework hiding the loop from you, no SDK abstracting away the message list. When the model requests a tool call, you'll watch the raw JSON arrive. That visibility is the point of the book.

When you're ready to swap in a cloud model, Chapter 15 shows what changes. It's three lines: a base URL, an API key, and a model name. The loop, the guardrails, and the MCP wiring don't change at all. That's not a coincidence. It's the payoff of building on the OpenAI-compatible wire format from day one.

## What You'll Build

By the last chapter, you'll have a working agent in two files, `agentic_demo.py` and `mcp_weather_server.py`, that:

Answers questions from a real 58-page employee handbook PDF, ingested into a persistent vector database with page-level source tracking.

Refuses to state facts the handbook doesn't contain, and backs that refusal with three layers of guardrails: a calibrated retrieval threshold, a citation verifier, and a structural backstop that physically removes the model's ability to keep searching forever.

Screens incoming prompts through two input guardrails, a free regex pass and a purpose-built guard model, before the main model ever sees them.

Calls tools from two sources at once: local Python functions and a real MCP server running as a separate process, indistinguishable to the model.

Every threshold, model name, and design decision in this book comes from actually building this system, including the reasoning for choices that aren't obvious, like why the retrieval threshold is calibrated from a labeled eval set instead of guessed, and why one guardrail is code instead of a prompt instruction.

## How to Read This Book

In order, with a terminal open. Each chapter builds on the reasoning of the one before it, and the code accumulates. Chapter 3 gets the loop running for the first time, and from there every construction chapter (4 through 9, 12, and 13) leaves you with a runnable checkpoint. Chapters 10 and 11 change no code at all, deliberately; each one's argument is its own punchline.

If you're the kind of reader who skips ahead: Chapter 14 is a standalone tips reference, and Appendix A has the complete final code listing. But the middle chapters explain *why* the code looks the way it does, and that reasoning is the part you'll reuse on systems that look nothing like this one.

## How Code Edits Are Specified

From Chapter 4 onward, most chapters modify the file you already have rather than starting fresh, and every modification names its placement the same way: an action (**insert**, **replace**, or **append**), an anchor (the function, constant, or exact line of existing code it attaches to), and a position relative to that anchor. Sometimes that's a `WHERE:` comment at the top of the code block, sometimes it's the sentence introducing the block or the numbered edit list at the chapter's end; either way, you're never left guessing where code goes. A `WHERE:` comment looks like this:

```python
# WHERE: insert at module level, directly after the ingest_pdf() function
```

Anchors are used instead of line numbers deliberately: line numbers drift the moment your file differs from the book's by a blank line, and every chapter would renumber all the ones after it. Anchors don't drift, because each chapter ends at a known checkpoint.

From Chapter 5 onward, each modifying chapter also closes with a **The Code Changed This Chapter** section listing every edit as a numbered step, in the order you should apply them. (Chapters 3 and 4 don't need one; they end with the complete file instead, since nearly everything in them is new.) If you ever lose track of the file's state, the nearest checkpoint listing (Chapters 3, 4, 9, and 13 contain the complete file) is your reset point: diff against it, or paste over and rerun. The companion `code/` directory carries a runnable checkpoint file for every code-changing chapter, built by applying each chapter's edits in sequence; see `code/README.md`.
