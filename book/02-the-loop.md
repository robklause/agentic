# Chapter 2: What Is an Agentic Loop

By the end of this chapter you'll have run a single model call, seen the exact shape of a tool-call request on the wire, and written the loop that turns those two things into an agent. Everything else in this book hangs off the seam you'll see here.

## One Model Call, No Tools

Strip everything away and an LLM interaction is one HTTPS request. You send a list of messages; you get one message back.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen3.5:9b"   # on Apple Silicon: "qwen3.5:9b-mlx"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 3 times 68?"},
    ],
)
print(response.choices[0].message.content)
```

Two things about this shape matter more than they look like they do.

First, `messages` is the entire conversation. The chat completions API is stateless: the server keeps nothing between calls. Every request resends the full history, and the model's sense of "what has happened so far" is exactly what's in that list, nothing more. This is what people mean by the context window, the bounded amount of conversation the model can see at once. It has real consequences you'll hit in Chapter 4, when tool results start bloating the list.

Second, the model can only answer from what it memorized during training. Ask it for live weather, today's date, or the contents of your company handbook and it will either refuse or, worse, guess fluently. The fix isn't a smarter model. It's giving the model a way to ask your code for things.

## What a Tool Call Looks Like on the Wire

Tool calling is the mechanism for that. Alongside your messages, you send a list of tool descriptions in JSON Schema: each tool's name, what it does, and what arguments it takes. The model can't execute anything. What it *can* do is reply with a structured request instead of prose.

Send a question the model can't answer alone, with one tool declared, and look at the raw response message:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Austin\"}"
      }
    }
  ]
}
```

No answer text. `content` is null. Instead there's `tool_calls`: the model is asking your code to run `get_weather(city="Austin")` and report back. (A weather tool is the classic example for showing this shape, so that's what's used here; you'll actually build one in Part 3, and the first tools you write in Chapter 3 are a calculator and a comparator.) The `arguments` field is a JSON *string*, not an object, so you'll `json.loads` it. The `id` matters too: when you send the result back, you tag it with this id so the model knows which request it answers.

This response is the seam of the whole book. Everything that comes later, guardrails in Part 2, MCP in Part 3, slots into the moment between "the model asked for a tool call" and "run it." The loop's control flow never changes again after this chapter.

## The Loop

An agent is what you get when you handle that response in a loop: execute the requested tool, append the result to the message list, and call the model again. Repeat until the model replies with text instead of another tool call.

Here's the loop in full. This is a simplified excerpt of the `run_agent` function from the book's final system; the real version (Appendix A) adds guardrails, history compaction, and MCP dispatch at exactly the seams you'll see marked, but its control flow is character-for-character this shape:

```python
import json

def run_agent(user_prompt: str, max_turns: int = 5, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. "
         "Use tools when they help answer the question."},
        {"role": "user", "content": user_prompt},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,   # the JSON Schema list — Chapter 3
        )
        message = response.choices[0].message

        # Case 1: model is done — it returned a normal text answer.
        if not message.tool_calls:
            return message.content

        # Case 2: model wants to call one or more tools.
        # Append the assistant's request first — the API requires it
        # before the matching tool result messages.
        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            # >>> the seam: everything later in this book plugs in here <<<
            impl = TOOL_IMPLEMENTATIONS.get(name)   # the dispatch dict — Chapter 3
            result = impl(**args) if impl else {"error": f"unknown tool {name}"}

            if verbose:
                print(f"--- turn {turn}: {name}({args}) -> {result}")

            # Feed the result back as a "tool" message tied to this call id.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Hit max_turns without a final answer."
```

Read the exit conditions, because they're the design.

The loop ends one of two ways. Either the model returns a message with no `tool_calls`, which is your answer, or you hit `max_turns` and bail. That ceiling isn't decoration. A model can get stuck requesting tool call after tool call without ever concluding, and without a turn limit that's an infinite loop that burns compute forever. In Chapter 8 you'll meet a subtler version of this failure, a model that reformulates the same fruitless search five different ways, and build a sharper backstop than a blunt turn cap. But the cap stays. Defense in depth starts here.

Notice also what the loop does *not* do. It doesn't parse the model's prose for intent, doesn't regex an answer out of text, doesn't maintain any state beyond the message list. The message list is the agent's entire memory, and the model's structured `tool_calls` field is the entire protocol. That austerity is why this loop, unchanged, will later drive tools it discovers at runtime from a separate process.

## Why "Agentic" Is Just This

It's worth saying plainly, because the industry vocabulary suggests otherwise: there is no separate "agent runtime," no planner module, no hidden reasoning engine. Frameworks that sell agent orchestration are selling this loop with configuration on top. The model proposes; your code disposes; the transcript accumulates. Structured turn-taking.

That's not a dismissal of frameworks. It's a claim about understanding. When you know the loop, framework documentation stops being magic words and becomes a map of where someone else put their guardrails and their dispatch. You'll evaluate those products by asking where the seam is, because you'll know there has to be one.

You can't run this file yet. `TOOL_SCHEMAS` and `TOOL_IMPLEMENTATIONS` don't exist, and the model has nothing to call. Chapter 3 fixes that, and then the loop turns over for the first time. That's also why this chapter has no complete-file listing at the end: the first runnable version of `agentic_demo.py` closes Chapter 3, and every chapter from there on ends with the full working file as it stands.
