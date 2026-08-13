# === Book checkpoint: complete file as of Chapter 3 ===
# To run: copy/rename to agentic_demo.py (see code/README.md)

"""
Minimal agentic loop using the OpenAI API's function/tool calling.

Concept: an "agent" is just a loop. You send the model a prompt plus a list
of tools it's allowed to call. The model doesn't execute anything itself —
it replies with a request like "call calculate(expression='12 / 3')". Your
code runs that function, feeds the result back into the conversation, and
the model continues. It keeps going until it produces a plain text answer
instead of another tool call.

This version points at a local Ollama server instead of OpenAI. Ollama
exposes an OpenAI-compatible endpoint at /v1, so the same `openai` SDK and
the same tool-calling wire format work unchanged — only base_url, api_key,
and model name differ. api_key is a required-but-unused placeholder.

Setup:
    pip install openai
    ollama pull qwen3.5:9b     # or qwen3.5:9b-mlx on Apple Silicon
    ollama serve               # if not already running
    python agentic_demo.py
"""

import json
import operator

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen3.5:9b"   # on Apple Silicon: "qwen3.5:9b-mlx"


# ---------------------------------------------------------------------------
# 1. Tools: real Python functions the model is allowed to invoke.
#    Each one needs (a) an implementation and (b) a JSON Schema description
#    so the model knows it exists and what arguments it takes.
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
    """Compare two numbers so the model never has to judge magnitude itself.

    This exists specifically to close the gap 'calculate' leaves open: a model
    can compute 3*68=204 via calculate and then still silently eyeball
    '101 < 204' in its own head when writing the final answer. Giving it a
    dedicated compare tool takes that judgment out of its hands too — there's
    no comparison left for it to perform without a tool call.
    """
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


# Registry mapping tool name -> actual callable, used when dispatching calls.
TOOL_IMPLEMENTATIONS = {
    "calculate": calculate,
    "compare": compare,
}

# JSON Schema descriptions sent to the model. This is the model's entire
# knowledge of what these tools do — name, description, and argument shapes.
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
]


# ---------------------------------------------------------------------------
# 2. The agent loop.
# ---------------------------------------------------------------------------

def run_agent(user_prompt: str, max_turns: int = 5, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Use tools when they help answer the question. "
            "Never trust your own math — if a calculation is needed, always call the "
            "calculate tool for it. Never judge a comparison yourself either — for "
            "anything like 'is X more than Y' or 'is X at least Y', always call the "
            "compare tool and use its result, rather than deciding in your head."
        )},
        {"role": "user", "content": user_prompt},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        message = response.choices[0].message

        # Case 1: model is done — it returned a normal text answer.
        if not message.tool_calls:
            if verbose:
                print(f"--- turn {turn}: no tool_calls -> final answer, loop ends ---")
            return message.content

        # Case 2: model wants to call one or more tools.
        # Append the assistant's tool-call request to the transcript first —
        # the API requires it before the matching tool result messages.
        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            impl = TOOL_IMPLEMENTATIONS.get(name)
            result = impl(**args) if impl else {"error": f"unknown tool {name}"}

            if verbose:
                print(f"--- turn {turn}: executing tool ---")
                print(f"    {name}({args}) -> {result}")

            # Feed the result back as a "tool" role message tied to this call id.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Hit max_turns without a final answer."


if __name__ == "__main__":
    prompts = [
        "If I have 12 apples and give away a third of them, then buy 7 more, how many do I have?",
        "Is 101 more than 3 times 68?",
    ]
    for p in prompts:
        print(f"\nUSER: {p}")
        print(f"AGENT: {run_agent(p)}")
