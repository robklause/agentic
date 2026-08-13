# Chapter 8: Designing Guardrails That Work Together

By the end of this chapter your agent is guaranteed to terminate with a real answer instead of burning its turn budget on futile searches, and your ingestion pipeline stops junk chunks from quietly undermining every guardrail built on top of retrieval. The organizing lesson: guardrails compose, and composition is where correct pieces produce new failures.

## Two Correct Guardrails, One New Failure

Chapter 7 ended with a loose end. Before the similarity threshold existed, the model handled no-answer questions tolerably: it read three mediocre chunks, usually noticed they didn't answer the question, and said so. After the threshold, the model receives an explicit, well-engineered rejection note, and sometimes responds by searching again. New phrasing, same rejection. Again. Until `max_turns` runs out and the user gets "Hit max_turns without a final answer," a strictly worse outcome than the pre-threshold behavior.

Sit with the mechanics, because the pattern recurs in every layered system you'll ever build. The threshold is correct: those chunks *were* weak, and suppressing them *was* the job. The model's reasoning isn't crazy either: an empty result reads like "bad query," and reformulating a bad query is what a diligent searcher does. Each component, evaluated alone, behaves. The failure lives entirely in the interaction: the threshold changed what "no answer" looks like, from three visible-but-weak chunks the model could reason about to a repeatable rejection that invites retry. Nobody wrote a retry loop. The system grew one.

The testing moral, stated once and bluntly: **testing guardrails in isolation validates components, not the system.** The threshold passed its calibration table. The citation check passed its cases. The infinite search loop lived between them, invisible to both.

## The Prompt Request, and Why It's Not Enough Alone

First response, per the standing pattern: tell the model what discipline looks like. Appended to the system prompt:

```text
Search discipline: if two search_knowledge_base calls in a row
come back with no results or a below-threshold rejection note,
stop searching — do not keep reformulating indefinitely.
Answer immediately with whatever you found (even if nothing),
stating plainly that the topic doesn't appear to be covered in
the handbook. Never spend your remaining turns searching without
producing a final answer.
```

Clear rule, explicit trigger, named consequence. And the model still, in some runs, sails past it and keeps searching, with the instruction sitting right there in context. This is Chapter 5's distinction cashing out in a trace you can watch: a prompt asking nicely is a request, not a guarantee, and the model demonstrated the difference with the whole system watching.

Run the design test from Chapter 5: *if the model ignores this sentence, what catches it?* Nothing. So structure.

## The Structural Backstop

The API accepts `tools=None`. A model that receives no tool declarations cannot emit a tool call; the only legal move left is text. That's the enforcement mechanism, and it gets used twice: count consecutive empty searches and take the tools away at two, and, unconditionally, never offer tools on the final turn. The last turn is always an answer turn.

The second rule isn't redundant with the first, and the reason is worth understanding before you read the code. The empty-search counter has a blind spot: it only advances when a search comes back *empty*, and in a topically coherent corpus, reformulated searches rarely do. A model hunting for the unanswerable half of a two-part benefits question keeps producing benefits-flavored queries, each one matches some semi-relevant chunk in that generically-related score band Chapter 7 mapped, each non-empty result resets the counter, and the model burns every turn searching while the actual answer sits retrieved in the transcript from turn 0. No component misbehaved. The counter counted correctly. The failure dodged the condition. The final-turn rule is the countermeasure: a trigger based on the turn number alone, which no pattern of search results can dodge, converting every would-be "Hit max_turns" into a real answer.

In `run_agent`, alongside the Chapter 6 tracker:

```python
    retrieved_pages = set()  # pages actually returned by search_knowledge_base this run
    consecutive_empty_searches = 0  # tracks the search-discipline guardrail below
```

The counter updates where search results land, one line extending the Chapter 6 block:

```python
            if name == "search_knowledge_base":
                found = result.get("results", [])
                for r in found:
                    retrieved_pages |= _extract_cited_pages(r.get("source", ""))
                consecutive_empty_searches = 0 if found else consecutive_empty_searches + 1
```

And the top of the turn loop makes the decision. Three small placements here: the two backstop lines go right after the compaction check, a trace line joins the existing verbose block, and the API call's `tools=` argument changes.

```python
# WHERE: in run_agent, insert directly after the
#   messages = _compact_history(messages, verbose=verbose)
# check at the top of the turn loop
        # Search-discipline structural backstop: the system prompt *asks* the
        # model to stop after two fruitless searches, but a prompt asking
        # nicely is a request, not a guarantee. If the last two searches both
        # came back empty, withhold tools entirely — the model must answer
        # in text, guaranteeing termination.
        # The final turn never gets tools either: the last turn is always
        # an answer turn, so the loop can't end on "hit max_turns" with
        # the answer stranded in the transcript. The empty-search counter
        # alone has a blind spot: in a topically coherent corpus,
        # reformulated searches keep matching *something* semi-relevant,
        # resetting the counter while the turns burn.
        force_text_only = consecutive_empty_searches >= 2 or turn == max_turns - 1
        tools_for_this_turn = None if force_text_only else TOOL_SCHEMAS
```

```python
# WHERE: in run_agent's existing `if verbose:` block, insert after the
#   print("   ", json.dumps(_dump_message(m)))
# line — then change the API call's tools=TOOL_SCHEMAS to tools=tools_for_this_turn
            if force_text_only:
                reason = "final turn" if turn == max_turns - 1 else "2 consecutive empty searches"
                print(f"  (tools disabled this turn — {reason}, forcing a final answer)")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_for_this_turn,
        )
```

One more small piece of structure while you're in the exit path. A small local model forced into a text-only turn occasionally complies in the most literal way possible: it returns an empty string. The turn ends, the loop exits, and the user sees a blank answer, which reads as a crash even though every mechanism worked. Cheap fix at the return:

```python
# WHERE: in run_agent's final-answer exit, replace the line
#   return message.content
            # A small model forced into a text-only turn occasionally
            # returns an empty string. Never hand the user a blank answer.
            return message.content or "(the model returned an empty answer)"
```

Read what changed philosophically, not just mechanically. The prompt tried to persuade the model not to choose the bad option. The backstop removes the option from the menu. There's nothing to persuade, nothing to comply with, no reliability question to hedge: the model cannot call a tool that wasn't offered, the same way `calculate` cannot `eval` arbitrary code. When you need a guarantee, restructure the choice space; don't escalate the rhetoric.

Keep the prompt rule anyway. When the model honors it, the agent concludes a turn earlier and the trace reads better. The prompt improves the common case; the counter bounds the known worst case a turn early; the final-turn rule bounds *every* worst case, including the ones nobody has imagined yet. Layers, each doing the job it's actually capable of. The "Hit max_turns without a final answer." return at the bottom of the loop survives as a last-resort sentinel, but with the final turn always tool-free, you should never see it again; if you do, something upstream changed.

## The Second Composition Bug: Junk Chunks

While validating the backstop, a subtler interaction surfaces, and it starts in a place nobody was looking: Chapter 4's chunker keeps everything, including fragments like a page that contributes only the two words "EMPLOYEE HANDBOOK."

Here's the mechanism, and it's worth understanding because it's invisible at every layer boundary. Very short chunks are prone to an embedding-normalization artifact: with so little content to dilute the match, a two-or-three-word chunk can score inflated similarity against almost any query sharing common vocabulary. In this system's testing, a bare "EMPLOYEE HANDBOOK" chunk scored 0.827 against an unrelated query, comfortably above chunks with real substantive content, and above any plausible calibrated threshold.

Now trace the blast radius through the stack you've built. The junk chunk clears the threshold, so a should-be-empty search comes back *non-empty*, so `consecutive_empty_searches` resets, so the backstop never fires, so the model keeps searching with garbage in hand. One malformed input at the bottom of the pipeline quietly disarmed a guardrail three layers up. And worse: casual testing at this point would have shown the backstop "working," because most queries don't happen to collide with the junk chunk's vocabulary. A guardrail can look validated by luck. The composition lesson again, from a direction nobody predicted.

The fix is a floor on chunk length:

```python
MIN_CHUNK_CHARS = 60              # drop chunks shorter than this — see _chunk_text
```

And `_chunk_text`'s return line becomes a filter:

```python
    # Drop very short chunks (title pages, running headers that survived,
    # stray section labels) before they're ever embedded. This isn't about
    # saving embedding calls — it's a correctness fix: short chunks are
    # prone to an embedding-normalization artifact where a 2-3 word chunk
    # scores an inflated similarity against almost any query that shares
    # common vocabulary, since there's so little content to dilute the
    # match. Filtering these out removes a class of false-positive "match"
    # the threshold guardrail can't distinguish from a real one.
    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
```

Chunk-quality filtering is not cosmetic cleanup. It's load-bearing for every guardrail whose input is a similarity score.

## The Stale-Cache Trap

One more composition, this time between the fix you just made and Chapter 4's caching. Change `_chunk_text` and rerun: nothing happens. The PDF's bytes didn't change, the content hash matches, ingestion skips, and the junk chunks sit exactly where they were, in the persisted collection on disk. The cache is behaving as designed, and its design just silently discarded your fix. You'd have every reason to believe the filter was live; it isn't.

The cache key was wrong from the start, it just didn't matter until now: the collection's contents are a function of the PDF *and* the chunking logic, and the key only covered the PDF. So the logic gets a version number, folded into the hash:

```python
# Bumped whenever chunking/ingestion logic changes in a way that should
# invalidate the on-disk cache, even though the source PDF's bytes haven't
# changed. Folded into the content hash below. Without this, changing
# MIN_CHUNK_CHARS (or chunk size, overlap, etc.) would silently do nothing —
# ingest_pdf() would see the same PDF hash as before and skip re-ingestion,
# leaving the stale, un-filtered chunks in place.
CHUNK_LOGIC_VERSION = 2
```

And in `ingest_pdf`, the hash line becomes:

```python
    content_hash = hashlib.sha256(pdf_bytes + str(CHUNK_LOGIC_VERSION).encode()).hexdigest()[:16]
```

Bump the constant whenever chunking behavior changes, and the cache invalidates itself. This is the same bug class you've met twice this chapter wearing different clothes: a component (the cache) doing its job correctly while the *system* around it changed what correct means. Derived-data caches must key on everything the derivation depends on. File bytes are the obvious dependency; the code that transforms them is the one everyone forgets.

## Run It

The `__main__` prompts gain a designed probe for the backstop:

```python
        # Deliberate test of the search-discipline structural backstop.
        # This topic has zero real content in a retail-boutique handbook
        # AND, now that MIN_CHUNK_CHARS filters out short junk chunks,
        # no leftover title/header chunk to spuriously inflate a
        # similarity score either — so two genuinely-empty (or
        # below-threshold) search_knowledge_base calls in a row is the
        # expected outcome, not a fluke.
        "Does the company offer a cryptocurrency signing bonus or stock buyback program?",
```

Watch the trace for the exact sequence that proves the layers are working: two "no chunk met the calibrated similarity threshold" notes, then "(tools disabled this turn — 2 consecutive empty searches, forcing a final answer)", then a real text answer saying the handbook doesn't cover it, with the citation guardrail passing on `retrieved: none`. Termination isn't hoped for anymore. It's enforced.

## The Code Changed This Chapter

Nine edits, in order.

**Edit 1: new constants.** Insert at module level, directly after the `CHUNK_OVERLAP = 150` line:

```python
MIN_CHUNK_CHARS = 60              # drop chunks shorter than this — see _chunk_text
CHUNK_LOGIC_VERSION = 2           # bump when chunking/ingestion logic changes
```

**Edit 2: the chunk filter.** In `_chunk_text()`, replace the final line:

```python
    return chunks
```

with the length filter (and the comment explaining it, from "The Second Composition Bug" above):

```python
    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
```

**Edit 3: the cache key.** In `ingest_pdf()`, replace the line:

```python
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
```

with:

```python
    content_hash = hashlib.sha256(pdf_bytes + str(CHUNK_LOGIC_VERSION).encode()).hexdigest()[:16]
```

**Edit 4: system prompt.** Append the "Search discipline: ..." block (shown in full in "The Prompt Request" above) to the system prompt string in `run_agent`, directly after the sentence ending "...even alongside a section number."

**Edit 5: the counter.** In `run_agent`, insert directly after the `retrieved_pages = set()` line:

```python
    consecutive_empty_searches = 0  # tracks the search-discipline guardrail below
```

**Edit 6: updating the counter.** In the `if name == "search_knowledge_base":` block from Chapter 6's Edit 4, insert one line after the `for r in found:` loop:

```python
                consecutive_empty_searches = 0 if found else consecutive_empty_searches + 1
```

**Edit 7: the backstop.** Three placements, all shown with WHERE comments in "The Structural Backstop" above: insert the `force_text_only` / `tools_for_this_turn` lines (including the final-turn condition) directly after the compaction check, add the `if force_text_only:` trace print with its `reason` line inside the existing `if verbose:` block, and change the API call's `tools=TOOL_SCHEMAS` to `tools=tools_for_this_turn`.

**Edit 8: never return a blank answer.** In the final-answer exit, replace `return message.content` with the fallback version shown above.

**Edit 9: the probe prompt.** Add the cryptocurrency question (from "Run It" above) to the `prompts` list in `__main__`, after the parental leave prompt. Every layer you build from here on gets a designed probe in the standard run; this is the first.

Next: the last guardrail layer, and the only one that runs before the model sees anything at all. Input screening, in two passes, including the moment a regex meets a pirate.
