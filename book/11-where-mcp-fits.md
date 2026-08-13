# Chapter 11: Where MCP Actually Fits

By the end of this short chapter you'll know exactly which two lines of your architecture MCP replaces, and, just as important, everything it doesn't touch. Most MCP explanations start with the protocol. This one starts with your loop, because you've built the thing MCP plugs into, and that makes the explanation nearly free.

## The Problem MCP Standardizes

Your agent has three tools, and you hand-built both halves of each: the implementation in `TOOL_IMPLEMENTATIONS`, the JSON Schema in `TOOL_SCHEMAS`. That's fine for three tools you wrote yourself. Now scale the picture the way an architect has to.

Every team building an agent needs tools for the same systems: Jira, GitHub, Postgres, Slack, the company data warehouse. Without a standard, every agent integrates every tool bilaterally, N agents times M tools, each pairing hand-typing schemas and writing dispatch glue against a different vendor API. You've seen this movie. It's why ODBC exists, why LSP exists, why REST conventions exist. Agent tooling in 2024 was the N×M matrix; MCP, the Model Context Protocol, is the standard interface that collapses it.

The shape: a tool provider ships an **MCP server**, a process that exposes tools over a standard wire protocol (JSON-RPC, over stdio for local processes or HTTP for remote ones). Any agent that speaks the protocol as a **client** can connect to any server, ask "what tools do you have?", get JSON Schema answers back, and invoke those tools. Write the server once; every MCP-speaking agent can use it. The tool ecosystem stops being a matrix and becomes a bus.

That's the strategic value, and it's real. But strategy talk is where most MCP explanations float away from the ground. So: down to your code.

## One Seam, The Same Seam As Always

Here is the entire claim, and it's precise: **MCP slots into exactly one seam of the loop you already built, the moment between "the model asked for a tool call" and "run it." Nothing about the loop's control flow changes.** What changes is where a tool's definition comes from, and where its execution happens.

You marked this seam in Chapter 2, before you knew you'd need it. Your loop today does two things with tools:

```python
# (today — no changes yet, this is orientation, not an edit)

# 1. Tell the model what exists: a hand-typed list, sent every turn.
tools=TOOL_SCHEMAS

# 2. Run what it asks for: a dict lookup, called in-process.
impl = TOOL_IMPLEMENTATIONS.get(name)
result = impl(**args)
```

MCP replaces the *source* behind each line:

```python
# (after Part 3 — where each half comes from)

# 1. Tool definitions: discovered at runtime by asking the server.
discovered = await mcp_client.list_tools()

# 2. Execution: invoked across the process boundary.
result = await mcp_client.call_tool(name, args)
```

Hand-typed schemas become `list_tools()`. The dispatch dict becomes `call_tool()`. That's the whole footprint. The model still receives one list of tool schemas and still emits the same `tool_calls` JSON you saw on the wire in Chapter 2; it neither knows nor cares that a schema was discovered rather than typed, or that a call crossed a subprocess boundary rather than a dict lookup. Your guardrails don't care either: input screening ran before the loop started, the backstop counts search results wherever they came from, the citation check reads the final answer. Every layer you built in Part 2 rides through Part 3 untouched.

If you hold one sentence from this chapter, hold that one. It's the difference between evaluating MCP as an architect and evaluating it as a reader of press releases: MCP is not an agent framework, not a reasoning layer, not a competitor to your loop. It's a standardized answer to two questions your loop was already asking: *what tools exist, and how do I run one?*

## What This Buys You in Practice

Three concrete consequences follow from the seam being that narrow, and each becomes visible in the next two chapters.

**Tools become deployable units.** A tool in `TOOL_IMPLEMENTATIONS` lives and dies with your agent's process: same Python version, same dependencies, same crash domain. A tool behind an MCP server is a separate process with its own everything. The weather server you'll build in Chapter 12 could be rewritten in TypeScript tomorrow and your agent wouldn't change a character.

**Discovery replaces documentation.** `list_tools()` returns name, description, and input schema, the exact trio you hand-authored in Chapter 3, straight from the running server. When the server adds a tool, your agent sees it on the next connection without a code change. The schema-as-prompt-engineering lesson from Chapter 3 still applies, it just moves: now the *server author* owns the description that steers the model.

**Trust boundaries become explicit.** You've treated the tool boundary as a trust boundary since `calculate` refused to `eval`. MCP makes that boundary a process boundary with a wire protocol across it, which is where an architect can actually enforce things: what a server is allowed to see, what its process can reach, what happens when it dies. A dict entry can't be sandboxed. A subprocess can.

One honest caution to carry into the ecosystem, Chapter 9's lesson wearing new clothes: an MCP server is a *supply chain* component. Its tool descriptions go into your model's context, and its results arrive wearing the authority of trusted tool output. A malicious or compromised server is the retrieved-content injection problem with a vendor logo. Connecting to a server is trusting its author; the protocol standardizes the plumbing, not the trustworthiness of what flows through it.

## The Plan for Part 3

Chapter 12 builds the server: a real MCP server in about twenty lines, serving one tool over the actual wire protocol, plus the one genuine gotcha in the SDK's result format. Chapter 13 converts the loop's tool layer to consume it: async plumbing, runtime discovery, dual dispatch, and the finished system, local tools and a remote one merged into a single list the model can't tell apart.

## The Code Changed This Chapter

Nothing. Like Chapter 10, that's the thesis: if MCP required rewriting your loop, it would be a framework. It's a socket.
