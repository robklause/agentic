# Chapter 6: Output Guardrails, Verifying Citations

By the end of this chapter your loop independently verifies every page citation in the model's final answer against what was actually retrieved, and flags answers that dodge the check by citing nothing at all. It's about forty lines of code, and it's the first time the system stops trusting the model about anything.

## The Gap Being Closed

Chapter 4's system prompt asks the model to "mention which source each fact came from." Chapter 5 taught you the reflex: that's a request. If the model cites a page it never saw, a page from its training-data memory of what employee handbooks look like, or just a plausible-sounding number, nothing in the system notices. The citation *looks* identical to a real one. A user reading "per p.23" has no way to know whether page 23 was retrieved this conversation or invented this sentence.

Whether a cited page was actually retrieved is a syntactic question, and by the Chapter 5 principle, syntactic questions get answered in code. The design is small: track which pages search returned, parse which pages the answer cites, and compare sets.

## Tracking What Was Actually Retrieved

Every search result already carries its source string, `Marlow_and_Sage_Handbook.pdf p.17`, built during ingestion in Chapter 4. The loop just needs to remember the page numbers as results flow through. Two small functions:

```python
def _extract_cited_pages(text: str) -> set:
    """Find 'p.NN' style page citations anywhere in text."""
    return {int(n) for n in re.findall(r"\bp\.\s*(\d+)\b", text)}


def _check_citation_grounding(answer: str, retrieved_pages: set) -> list:
    """Return any page numbers the answer cites that were never actually
    retrieved this conversation — a hallucinated or misremembered citation.
    """
    cited = _extract_cited_pages(answer)
    return sorted(cited - retrieved_pages)
```

The same extractor runs on both sides, on tool results to build the retrieved set, on the final answer to build the cited set, so the two sets are comparable by construction. Same instinct as the shared embedding function in Chapter 4: when two things must agree, route them through one piece of code.

In `run_agent`, initialize the tracker before the loop:

```python
    retrieved_pages = set()  # pages actually returned by search_knowledge_base this run
```

And in the tool-dispatch section, right after a `search_knowledge_base` result comes back:

```python
            if name == "search_knowledge_base":
                found = result.get("results", [])
                for r in found:
                    retrieved_pages |= _extract_cited_pages(r.get("source", ""))
```

## The Check, and the Trap Inside the Obvious Version

The obvious check runs at the final-answer exit: any cited page not in the retrieved set gets flagged. Necessary, and not sufficient, and the insufficiency is the real lesson of this chapter.

Ask what makes the obvious check print "passed." An answer with zero citations passes: nothing cited, nothing to flag. An answer citing "Section 5.3" instead of "p.18" passes: the regex finds no `p.NN`, the unverified set is empty, green light. In both cases the guardrail verified *nothing* and reported success. A "passed" check that had nothing to verify is a false sense of security dressed up as a green light, and guardrails that mostly print "passed" get trusted exactly as much as they shouldn't be.

So the check gets a second clause: if real content was retrieved this conversation and the answer contains zero page citations, that's a flag too. The full exit block in `run_agent`:

```python
        # Case 1: model is done — it returned a normal text answer.
        if not message.tool_calls:
            if verbose:
                print(f"--- turn {turn}: no tool_calls -> final answer, loop ends ---")

            # Output guardrail: does every page citation in the answer trace
            # back to something actually retrieved this conversation? And,
            # separately — if real content WAS retrieved, does the answer
            # cite anything at all? A "passed" check that never had a
            # citation to verify is a false sense of security, not a real
            # pass.
            cited_pages = _extract_cited_pages(message.content or "")
            unverified = _check_citation_grounding(message.content or "", retrieved_pages)
            if unverified:
                print(
                    f"--- GUARDRAIL: answer cites page(s) {unverified} that were "
                    f"never retrieved this conversation (retrieved: {sorted(retrieved_pages) or 'none'}) ---"
                )
            elif retrieved_pages and not cited_pages:
                print(
                    f"--- GUARDRAIL: real content was retrieved (pages {sorted(retrieved_pages)}) "
                    f"but the answer contains zero page citations — cannot verify grounding ---"
                )
            elif verbose:
                print(f"--- GUARDRAIL: citation check passed (retrieved: {sorted(retrieved_pages) or 'none'}) ---")

            return message.content
```

The zero-citation clause needs a cooperating model to be useful: an answer must cite in a parseable format before the parseable format can be verified. So the system prompt gets strict about it:

```text
Citation format is strict: every fact you state from the handbook —
including facts you're citing only to explain that they DON'T answer
the question — must be tagged inline with the literal page citation
exactly as it appears in the chunk's source field, e.g. 'p.17'.
Section numbers like 'Section 5.3' are not a substitute for the page
citation; include the page citation every time, even alongside a
section number.
```

The oddly specific clauses are load-bearing. "Including facts you're citing only to explain that they DON'T answer the question" exists because partially-relevant retrievals produce exactly that kind of sentence, and it needs grounding like any other. The Section-numbers sentence exists because section numbers are the natural way a model summarizes a handbook chunk, they're visible right there in the text, and they're unverifiable by this guardrail, which tracks pages. The prompt bends the model toward the format the code can check. That pairing, prompt shapes the output, code verifies it, is the standing pattern for every guardrail that follows.

## Design Decisions Worth Defending

**Why flag and not block?** The guardrail prints a warning and returns the answer anyway. In this system a human reads the trace, and the right response to a flagged citation is scrutiny, not silence. A production variant might retry with the flag appended to the conversation, or route to a human queue. Blocking outright trades a hallucinated citation for no answer, which isn't obviously better. What matters architecturally is that detection exists and is code, not model self-report; the response policy is a separate, swappable decision.

**Why only citations, and not every claim?** Deliberate narrowness. A fuller guardrail would check whether the answer's numeric claims appear in retrieved text, but that's noisy in a specific, predictable way: a model paraphrasing "10 days per year" as "ten days annually" would false-positive as ungrounded even though it's a faithful restatement. Checking paraphrase faithfulness means judging meaning, and the Chapter 5 principle sends judging-meaning problems to a different layer. Citations are the highest-confidence signal available to a syntactic check: consistently formatted, requested explicitly, verifiable exactly. A guardrail that's precise about a narrow thing beats one that's mushy about a broad thing, because you can *trust* the narrow one's output.

Keep an honest ledger of what this guardrail does not catch, though. A fabricated claim wearing a *retrieved* page number sails through: the page checks out, the sentence doesn't, and set membership can't see it. And an answer built purely from training memory, zero tool calls, zero citations, also passes, because with an empty retrieved set the zero-citation clause never triggers. That second gap gets a name and a decision in Chapter 10. Writing down what a guardrail can't see is as much a part of building it as the code; the alternative is discovering the boundary in production, with confidence you shouldn't have had.

## Run It

Ask the PTO question and watch the exit:

```
--- GUARDRAIL: citation check passed (retrieved: [17, 18]) ---
```

(The exact page numbers depend on your PDF; what matters is that the set matches what search actually returned.) That line now means something specific: every `p.NN` in the answer traces to a chunk that came back through the tool this conversation, and at least one citation was present to check. Not that the answer is *true*, set membership doesn't prove faithfulness, but that the citations are real. One judgment the model used to hold on faith now sits behind a check it can't charm.

## The Code Changed This Chapter

Five edits, in order. Everything else is unchanged from the Chapter 4 listing (with Chapter 5's system prompt).

**Edit 1: new helper functions.** Insert at module level, directly after `_compact_history()`:

```python
def _extract_cited_pages(text: str) -> set:
    """Find 'p.NN' style page citations anywhere in text."""
    return {int(n) for n in re.findall(r"\bp\.\s*(\d+)\b", text)}


def _check_citation_grounding(answer: str, retrieved_pages: set) -> list:
    """Return any page numbers the answer cites that were never actually
    retrieved this conversation — a hallucinated or misremembered citation.
    """
    cited = _extract_cited_pages(answer)
    return sorted(cited - retrieved_pages)
```

**Edit 2: system prompt.** Append the strict-citation-format block (shown in full in "The Check" section above) to the system prompt string in `run_agent`, directly after the sentence ending "...even if it seems like a reasonable inference."

**Edit 3: the tracker.** Inside `run_agent`, insert one line between the `messages = [...]` block and the `for turn in range(max_turns):` line:

```python
    retrieved_pages = set()  # pages actually returned by search_knowledge_base this run
```

**Edit 4: feeding the tracker.** Inside the tool-dispatch `for call in message.tool_calls:` loop, insert directly after the line `result = impl(**args) if impl else {"error": f"unknown tool {name}"}`:

```python
            if name == "search_knowledge_base":
                found = result.get("results", [])
                for r in found:
                    retrieved_pages |= _extract_cited_pages(r.get("source", ""))
```

**Edit 5: the check at the exit.** Replace the entire `if not message.tool_calls:` block (the final-answer exit) with the version shown in "The Check, and the Trap Inside the Obvious Version" above: same entry line, but with the `cited_pages` / `unverified` computation and the three-way GUARDRAIL print before `return message.content`.

Next: the other end of the pipeline. The citation check verifies what comes *out*; Chapter 7 stops weak matches from going *in*, and teaches you why the threshold that does it has to be measured, not guessed.
