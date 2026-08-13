# Chapter 3: Your First Tools

By the end of this chapter the loop runs. You'll give the model two tools, a calculator and a comparator, watch it choose to use them, and learn the design rule that decides what deserves to be a tool at all. The second tool looks redundant until you see the failure it closes.

## A Tool Is Two Things

Every tool in this system is exactly two artifacts: an implementation (a plain Python function) and a schema (a JSON description the model sees). Keep that split sharp in your head. The model never sees your code. The schema is the model's *entire* knowledge of what a tool does: its name, its description, its argument shapes. Your function could be empty, could be brilliant, could be a network call to another continent; from the model's side of the wire, the schema is the tool.

That has a practical consequence you'll use all book: the description field is prompt engineering. It's not documentation for humans. It's instructions to the model about when and how to call, and you'll see in a moment that the difference between a tool being used correctly and ignored can live entirely in that string.

## The First Tool: calculate

LLMs are unreliable at arithmetic. Not incapable, unreliable, which is worse: a model that's right 95% of the time about `3 * 68` will state the wrong answer with the same fluent confidence as the right one. Arithmetic is deterministic, so it should never be left to a probabilistic engine. That's the general design rule, and it's worth stating as one: **if a computation has a correct answer, take it out of the model's hands and put it in a tool.**

Here's the implementation:

```python
import operator

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
```

Two decisions in here deserve a sentence each.

It parses the expression into an AST and walks it against an allowlist of operators, instead of calling `eval()`. The expression string comes from the model, and the model's input comes from the user. That's a chain of untrusted input, and `eval()` on untrusted input is code execution. This is the book's first guardrail, two chapters before the word appears: the tool boundary is a trust boundary, and it stays one all the way through Part 3.

And it returns errors as data, `{"error": str(e)}`, instead of raising. A raised exception kills your loop. An error dict goes back to the model as a tool result, and the model can read it, recover, and try differently. Tools talk to the model, and the model can't catch exceptions.

## The Second Tool: compare, and Why It Exists

Give the model `calculate` and ask: "Is Austin's temperature more than 3 times as warm as Seattle's?" Watch what happens. The model dutifully calls `calculate("3 * 68")`, gets back 204, and then, writing its final answer, silently eyeballs whether 101 is bigger than 204 in its own head.

It computed the product with a tool and then did the *comparison* on vibes. The judgment you took out of its hands came right back in through a gap you didn't know was there.

That gap is why `compare` exists:

```python
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
```

Trivial code. The design lesson isn't. When you delegate a computation to a tool, audit what judgment remains on the model's side of the handoff. "Compute 3 times 68" and "decide whether 101 exceeds 204" feel like one operation to a human. They're two, and the model will happily do the second one freehand unless you take it too. You'll re-meet this exact principle at system scale in Part 2: every guardrail in this book is the same move, finding a judgment the model is quietly making and taking it away.

## The Schemas

Now the model-facing half. These go in a list called `TOOL_SCHEMAS`, which is what the loop passes as `tools=` on every call:

```python
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
```

Look at the two descriptions side by side. `calculate`'s is one flat sentence. `compare`'s is four sentences of increasingly explicit instruction, capitalized ANY, a direct "do not judge comparisons yourself." That asymmetry is earned: models reach for a calculator naturally, but they do not naturally believe magnitude judgment is beyond them. The description is doing behavioral work, not describing an API.

The `enum` on `operator_` is the same instinct in schema form. You could accept any string and validate in the function. Declaring the six legal values in the schema means the model knows the contract before it calls, and malformed calls mostly don't happen instead of mostly getting caught.

## Dispatch, and the Reinforcement in the System Prompt

The last piece is the registry the loop's seam reads from:

```python
TOOL_IMPLEMENTATIONS = {
    "calculate": calculate,
    "compare": compare,
}
```

A dict from name to callable, nothing else. In Chapter 13 this dict becomes one of *two* places a tool call can route, and the fact that dispatch is this dumb is exactly why that change will be small.

One more thing before you run it. The tool descriptions push the model toward the tools, but the system prompt pushes from the other side. In the final system it opens like this:

```text
You are a helpful assistant. Use tools when they help answer the question.
Never trust your own math — if a calculation is needed, always call the
calculate tool for it. Never judge a comparison yourself either — for
anything like 'is X more than Y' or 'is X at least Y', always call the
compare tool and use its result, rather than deciding in your head.
```

Belt and suspenders, deliberately. A local 9B model follows instructions less reliably than a frontier model, so the same rule lives in both places the model looks. Keep an honest grip on what this buys you, though: these are requests. Well-phrased, redundantly placed, and still just requests. The model can ignore every word, and Part 2 exists because sometimes it does. The hierarchy this book keeps returning to: prompts reduce a failure rate, structure eliminates a failure mode. You've already built one of each in this chapter, the prompt above and the AST allowlist in `calculate`.

## Run It

The complete file below wires this chapter and the last together. Run it with a question that needs both tools, and with `verbose=True` you'll see the loop turn over for real: the model calls `calculate`, a result comes back, maybe another call, then a text answer and the loop exits. The trace output is the product here, not the apple count. You're watching a model decide, turn by turn, that it needs your code, and you can see every message that produced every decision.

Sit with the trace for a minute before moving on. From here forward the system only gets more layered, and the trace is how you'll debug all of it.

## The Complete File: agentic_demo.py, as of Chapter 3

This is the first runnable version. Every chapter from here on ends the same way, with the full file as it stands, so you always have a known-good checkpoint to diff against.

```python
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
```

Next: the model gets something real to be wrong about, a 58-page PDF, and you'll meet the retrieval problem that the entire guardrails part of this book grows out of.
