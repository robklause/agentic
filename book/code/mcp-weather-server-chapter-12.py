# === Book checkpoint: mcp_weather_server.py as of Chapter 12 (unchanged through 13) ===
# To run: copy/rename to mcp_weather_server.py (see code/README.md)

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
