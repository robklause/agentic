# Chapter 5: Why Guardrails, Not Just Better Prompts

By the end of this chapter you'll be able to name the failure that motivates every guardrail in this book, and you'll know the one-sentence engineering principle that decides whether a fix belongs in your prompt or in your code. There's little new code here. There's a new way of reading model output, and it's the chapter the rest of Part 2 stands on.

## The Failure: Silence Becomes a Stated Negative

Ask the Chapter 4 agent a two-part question:

```
USER: How many PTO days do new employees get, and does it change over time?
```

The first half goes beautifully. The model calls `search_knowledge_base`, retrieves the PTO section, and reports the real accrual policy with a page source. This is RAG working as designed.

Then the second half. The retrieved chunk says nothing about accrual rates changing with tenure. Nothing for it, nothing against it. And the model, wanting to answer the whole question, writes something like: "No, the accrual rate does not change over time."

Read that failure precisely, because its shape is the whole point. The model didn't invent a fact from nowhere. It converted *absence of evidence* into a *stated negative*. The document was silent; the model reported the silence as a "no." And a "no" about your PTO policy is exactly as much a factual claim as a "yes." If the handbook elsewhere grants senior staff a higher accrual rate, the agent just confidently misinformed an employee, while citing a real page.

This pattern generalizes viciously. Not mentioned in this chunk becomes "not offered." No exception listed becomes "no exceptions exist." Part-time employees not named in the benefit paragraph becomes "part-time employees aren't eligible." Every one of those is fluent, plausible, cited-adjacent, and unsupported. It's the most dangerous failure class in RAG systems precisely because it *looks* like grounded behavior. The retrieval happened. The citation is real. Only the claim is manufactured.

## Why "Just Fix the Prompt" Underdelivers

The instinctive fix is a system prompt rule, and it's worth doing. Here's the thing to be honest about before you do: a rule with an example teaches the example more than the rule.

Add "if the text doesn't say whether the rate changes with tenure, don't claim it does or doesn't" and the model stops making *that* claim. Then it finds a new unstated negative you didn't enumerate: eligibility, probation periods, caps, whatever your example didn't cover. You aren't closing the gap; you're playing whack-a-mole with an opponent that generates moles. One worked example doesn't reliably teach the general rule.

What *does* help is stating the general rule as a general rule, explicitly refusing to let it be read as one example, and naming the categories it covers. That's what the final system prompt does, and it's this chapter's code contribution. Appended to the handbook rules from Chapter 4, directly after the sentence ending "...say you don't have that information rather than guessing.":

```text
This is a general principle, not limited to one example: a retrieved
chunk simply not mentioning something is NEVER grounds to state a
negative or exclusion as fact. This applies to every kind of unstated
claim — tenure-based changes, eligibility restrictions (e.g. don't
say part-time employees are excluded from a benefit unless a chunk
explicitly excludes them), exceptions, caps, anything. Before adding
any qualifier, aside, or 'note:' to your answer, check it is
explicitly stated in a retrieved chunk — if it isn't, cut it, even
if it seems like a reasonable inference.
```

Notice the construction. "This is a general principle, not limited to one example" exists because the model treats examples as boundaries. The named categories exist because generalization improves when you sketch the space. The last sentence targets the exact spot where these fabrications appear, the helpful little qualifier tacked onto an otherwise grounded answer, and the phrase "even if it seems like a reasonable inference" exists because that's precisely what these claims feel like from the inside. Silence-as-negation *is* reasonable inference. That's why the model keeps doing it. It's just not grounded.

And one more addition, closing a sibling gap on the numeric side. This one goes earlier in the prompt, directly after the sentence ending "...rather than deciding in your head.":

```text
Do not state any numeric claim, ratio, or comparison in your final
answer unless it came directly from a tool result earlier in this
conversation — if you haven't called a tool for a number, leave it out.
```

This is Chapter 3's `compare` lesson arriving at system scale. The model computes with tools, then garnishes the final answer with a freehand number anyway. Same move, same countermove: name the judgment, take it away.

## The Principle That Organizes Part 2

Now the honest question: after all that careful prompt engineering, is the failure gone?

No. Reduced, meaningfully. Gone, no. And that's not a defect in the prompt's wording that a better sentence would fix. It's the nature of the instrument. So here is the principle the next four chapters are built on, stated once, plainly:

**What you can check syntactically, fix with code. What requires judging meaning, a prompt can only make less frequent, never eliminate.**

Whether a cited page number appears in the set of retrieved pages is a syntactic question. Code answers it with certainty, every time, in Chapter 6. Whether a similarity score clears a threshold is arithmetic. Code, Chapter 7. Whether the model has burned two searches with nothing to show is a counter. Code, Chapter 8. Whether a sentence is a "reasonable inference" versus a "grounded claim" requires judging meaning, so it lives in the prompt, and the prompt's job is honest: push the failure rate down and let the *checkable* consequences of any leakage get caught by the code layers.

That's what "layered guardrails" means when it's an architecture and not a slogan. The prompt reduces. The code verifies what's verifiable. Neither pretends to do the other's job.

One more habit to take from this chapter: prompts are requests, not guarantees. You'll see that sentence again in Chapter 8, where the model ignores a perfectly clear instruction with the whole system watching, and the fix is to remove the option rather than sharpen the request. Keep the distinction in your pocket. It's the most reusable design test in this book: *if the model ignored this sentence, what catches it?* If the answer is "nothing," you have a request where you need a structure.

## The Code Changed This Chapter

One edit. **Replace the entire `messages = [...]` system-prompt block inside `run_agent`** (from `messages = [` down to the closing `]` after the user-prompt line) with the version below, which contains this chapter's two additions: the numeric-claim rule after "...rather than deciding in your head.", and the silence-is-not-a-negative block after "...rather than guessing."

```python
    # WHERE: replace the messages = [...] block at the top of run_agent
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Use tools when they help answer the question. "
            "Never trust your own math — if a calculation is needed, always call the "
            "calculate tool for it. Never judge a comparison yourself either — for "
            "anything like 'is X more than Y' or 'is X at least Y', always call the "
            "compare tool and use its result, rather than deciding in your head. "
            "Do not state any numeric claim, ratio, or comparison in your final "
            "answer unless it came directly from a tool result earlier in this "
            "conversation — if you haven't called a tool for a number, leave it out. "
            "For any question about company policy, benefits, or the employee "
            "handbook, always call search_knowledge_base first — do not answer "
            "from your own memory. Only state facts that appear in the returned "
            "chunks, and mention which source each fact came from. If the search "
            "results don't contain the answer, say you don't have that "
            "information rather than guessing. This is a general principle, "
            "not limited to one example: a retrieved chunk simply not "
            "mentioning something is NEVER grounds to state a negative or "
            "exclusion as fact. This applies to every kind of unstated claim "
            "— tenure-based changes, eligibility restrictions (e.g. don't "
            "say part-time employees are excluded from a benefit unless a "
            "chunk explicitly excludes them), exceptions, caps, anything. "
            "Before adding any qualifier, aside, or 'note:' to your answer, "
            "check it is explicitly stated in a retrieved chunk — if it "
            "isn't, cut it, even if it seems like a reasonable inference."
        )},
        {"role": "user", "content": user_prompt},
    ]
```

Everything else in the file is unchanged from the Chapter 4 listing. Run the two-part PTO question again and compare answers with and without the new rules; you should see the unstated-negative rate drop and the phrase "the handbook doesn't say" start appearing where fabricated qualifiers used to be.

Next chapter, the first code guardrail: the model claims its citations, and the loop stops taking its word for it.
