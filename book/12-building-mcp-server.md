# Chapter 12: Building a Local MCP Server

By the end of this chapter you'll have a real MCP server running as a separate process, serving a weather tool over the actual wire protocol, and you'll have verified with your own eyes the one place the SDK's behavior diverges from what its documentation implies. The server is about twenty lines. The verification habit is the durable part.

## First, the Five-Minute Local Version

To make Chapter 11's claim testable, "what a tool does doesn't change, only how it's found and called", start by writing the weather tool the way you've written every tool so far. If this were staying local, you'd add it in five minutes flat:

```python
# (illustration only — we're NOT adding this to agentic_demo.py;
#  this logic is about to live somewhere else)

FAKE_WEATHER = {
    "austin": {"temp_f": 101, "condition": "sunny"},
    "seattle": {"temp_f": 68, "condition": "cloudy"},
    "new york": {"temp_f": 84, "condition": "humid"},
}

def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return FAKE_WEATHER.get(city.lower(), {"temp_f": 75, "condition": "unknown"})
```

Fake data, deliberately. A real weather API would add an HTTP dependency and an API key to a chapter that's about neither; the interesting part is the plumbing, and fake data keeps the plumbing legible. You know the rest of the drill from Chapter 3: an entry in `TOOL_IMPLEMENTATIONS`, a hand-typed schema in `TOOL_SCHEMAS`, done.

We're not doing that. This exact logic is going to live in a different process, and by the end of Chapter 13 the model will call it anyway, without either of those hand-written registrations existing.

## The Server

Check your Python version one more time, `python3 --version`, 3.10 or better, because this is the moment Chapter 1's warning was about: the `mcp` SDK (v2.0.0 or later, from your `requirements.txt`) is what refuses to install on 3.9 with that misleading "no matching distribution" error.

Create a new file, `mcp_weather_server.py`, next to `agentic_demo.py`. Here it is in full, and this listing is the complete file, not an excerpt:

```python
"""
A minimal MCP server, standing in for what a real third-party tool
provider looks like: a separate process, exposing one tool over the actual
MCP wire protocol (JSON-RPC over stdio), discovered and called by
agentic_demo.py at runtime instead of hardcoded into it.

The weather logic is deliberately the same fake-data lookup you'd have
written as a local tool. Serving it here changes nothing about *what* the
tool does — same city lookups, same fake data — only *how* the agent loop
finds and calls it: discovered via list_tools() and invoked via
call_tool() across a subprocess boundary, instead of a hardcoded dict
lookup in the same process.

You don't normally run this file directly — agentic_demo.py launches it
as a subprocess via StdioServerParameters + stdio_client. Run it by hand
only to confirm it starts without errors; it will then sit blocked on
stdin waiting for a client to speak first (Ctrl+C to exit):
    python3 mcp_weather_server.py
"""

from mcp.server import MCPServer

mcp = MCPServer(
    "WeatherServer",
    instructions="Provides current weather lookups for a small set of cities.",
)

FAKE_WEATHER = {
    "austin": {"temp_f": 101, "condition": "sunny"},
    "seattle": {"temp_f": 68, "condition": "cloudy"},
    "new york": {"temp_f": 84, "condition": "humid"},
}


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return FAKE_WEATHER.get(city.lower(), {"temp_f": 75, "condition": "unknown"})


if __name__ == "__main__":
    mcp.run()  # stdio by default — blocks, waiting for a client on stdin
```

Twenty-odd lines, three moves.

`MCPServer("WeatherServer", instructions=...)` declares the server. The `instructions` string is server-level context a client can surface to its model, a system-prompt fragment from the tool provider's side. One more place where, per Chapter 3, description text is quietly prompt engineering, now authored by whoever ships the server.

`@mcp.tool()` is the whole registration story, and it's worth pausing on what you're *not* writing. In Chapter 3, every tool cost you a hand-typed JSON Schema block: name, description, properties, required fields. Here the SDK derives all of it by introspection: the function name becomes the tool name, the docstring becomes the description, the type hints (`city: str`) become the parameter schema. The `TOOL_SCHEMAS` half of your bilateral integration work just became the server's job, done by a decorator. When Chapter 13's `list_tools()` call returns this tool, the schema you receive is the one this decorator generated.

`mcp.run()` starts serving on stdio: JSON-RPC messages in on stdin, responses out on stdout. Stdio transport is what makes the local case so clean, no port, no TLS, no auth story, the client owns the subprocess and its pipes. A production remote server would swap this for HTTP transport; the tool code above it wouldn't change.

## Run It by Hand, Once

```bash
python3 mcp_weather_server.py
```

What you want from this run is an anticlimax: no import errors, no traceback, and then... nothing. The process sits silently, blocked on stdin, waiting for a client to speak the protocol's opening handshake. It will wait forever; Ctrl+C to exit. That silence is correct behavior for a stdio server, and knowing that saves you from "it hangs, something's broken" the first time you see it. You won't normally launch this file yourself again: from Chapter 13 on, `agentic_demo.py` spawns it as a subprocess, wires the pipes, and speaks the handshake for you.

## The Gotcha: Verify the Result Shape Against a Live Server

Now the one genuine trap in this integration, and the reason this chapter insists on a verification habit rather than doc-reading.

When a client calls a tool, the MCP result object offers two ways to carry the payload: `content`, a list of content blocks (text, images), and `structured_content`, a parsed structured value. The SDK's happy-path examples read as though returning a dict from your tool populates `structured_content`, and your client can just reach for it.

Test it against the *running* server and you find the real contract: **a plain `-> dict` return type does not populate `structured_content`.** It comes back `None`. Your dict gets JSON-serialized into a *text* content block instead, a string that merely looks like your data. Only a Pydantic `BaseModel` return type (or an explicitly declared output schema) produces a populated `structured_content`.

This matters because of who's downstream. Your loop's tool results flow into `json.dumps` and then into the model's context. Pass the raw text block through as if it were the value and nothing crashes; the model just receives a JSON string inside a JSON string, quotes escaped, and gets measurably worse at reading it. The client in Chapter 13 handles the real contract explicitly: use `structured_content` when present, otherwise parse the text block back into a dict yourself. You'll see the code there; what belongs in *this* chapter is where the knowledge came from.

It came from calling the live server and looking at the actual result object, not from the documentation. That's the habit worth generalizing, and it applies double at protocol boundaries you don't own: **the docs describe the happy path; the running system defines the contract.** You've met this principle before, Chapter 9 told you to verify Granite Guardian's `<score>` tag against live output before trusting the parse. Same move here. Ten minutes with a live endpoint beats an afternoon debugging an assumption.

(Why not just return a Pydantic model from the server and take the happy path? Because Chapter 13's client code shouldn't get to assume every MCP server it ever meets was written by someone who did. The parse-the-text fallback is the code that makes your client robust to the servers you *didn't* write, which, per Chapter 11, is the entire point of speaking a standard protocol.)

## The Code Changed This Chapter

One new file, `mcp_weather_server.py`, shown complete above. `agentic_demo.py` is untouched, still exactly the Chapter 9 checkpoint, and the two processes have never spoken. Chapter 13 makes the introduction.
