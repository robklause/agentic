# Chapter 7: Embedding Guardrails, Calibrating a Similarity Threshold

By the end of this chapter your search tool rejects weak retrieval matches before the model ever sees them, using a threshold measured from a labeled eval set instead of guessed from a couple of lucky examples. You'll also learn what "no match" actually looks like in embedding space, which is messier than you'd hope and worth knowing before you trust any retrieval system, including ones you buy.

## The Loose Thread From Chapter 4

`search_knowledge_base` returns the top 3 nearest chunks *unconditionally*. Nearest is relative: when the handbook has no parental leave policy, the nearest chunks to "parental leave" are whatever leave-adjacent passages exist, bereavement, jury duty, sick time. The model receives them formatted exactly like a real answer, three chunks, sources attached, and every downstream protection now depends on the model noticing they don't actually answer the question. Chapter 5 told you what that dependence is worth.

The structural fix: filter by the `similarity` score each result already carries, and turn "nothing cleared the bar" into an explicit signal instead of three mediocre chunks.

## Why You Don't Guess the Threshold

The tempting shortcut: run one good query and one bad query, eyeball the scores, split the difference. A real match came back 0.86, garbage came back around 0.55, so 0.7, ship it.

That's overfitting to two data points, and embedding-space geometry punishes it specifically. When a document genuinely answers a query, the gap is often clean, 0.86 against a 0.5-to-0.6 background. But when a document has *no* answer, there is no "unrelated" score to fall back to, because within a topically coherent corpus, everything is somewhat related. Every chunk of an employee handbook is about employment. A no-answer query about benefits still lands in a band of generically benefits-flavored similarity, and that band sits uncomfortably close to where weak-but-real matches live.

Real calibration numbers make it concrete. In one run of this system's calibration, "loss prevention," a topic with an entire real section in the handbook, scored 0.598 on its best chunk. "Parental leave," a topic with *zero* real content, scored 0.639. The absent topic outscored the present one. No single cutoff separates those two, and any threshold you eyeball from two examples will be wrong about cases like them in whichever direction your examples happened to lean. (Your exact scores will differ, they're a property of document, chunking, and embedding model together, but the overlap phenomenon is general.)

The response isn't despair; it's measurement. If a threshold is a classifier, calibrate it like one: labeled examples, in both classes, scored by the real pipeline.

## The Eval Set

Ten queries, five with confirmed answers in the handbook, five confirmed absent. The labels were verified by grepping the extracted PDF text before writing them down, and that step is not optional: label an eval set from what you *assume* your document covers and you calibrate to your assumptions, not your corpus.

```python
EVAL_SET = [
    # Confirmed present in the handbook (grep'd the extracted PDF text first)
    ("How many PTO days do employees get?", True),
    ("What is the dress code policy?", True),
    ("What are the sales expectations for team members?", True),
    ("What is the loss prevention policy?", True),
    ("What are the store hours?", True),
    # Confirmed absent from the handbook (zero matches on grep)
    ("What is the company's policy on parental leave?", False),
    ("Does the company offer remote work?", False),
    ("What is the tuition reimbursement policy?", False),
    ("Does the company offer stock options?", False),
    ("Is there a gym membership benefit?", False),
]
```

Note what the negative examples are *not*: they're not random off-topic strings like "how do I fly a helicopter." Those would score low against any handbook and teach the threshold nothing. These negatives are plausible HR questions an employee would actually ask, the hard cases, the ones that produce that generically-related score band. Calibrate on the boundary you'll actually be defending.

## The Calibration Function

```python
SIMILARITY_THRESHOLD = None  # set by calibrate_similarity_threshold(); None = guardrail off


def calibrate_similarity_threshold(eval_set: list = EVAL_SET, verbose: bool = True) -> float:
    """Run each labeled (query, expect_match) pair through real retrieval,
    take the top-1 similarity score for each, and set SIMILARITY_THRESHOLD to
    the midpoint between the lowest true-match score and the highest
    true-non-match score — the biggest margin this eval set actually
    supports, not a number pulled from one lucky/unlucky example.
    """
    global SIMILARITY_THRESHOLD, _kb_collection
    if _kb_collection is None:
        _kb_collection = ingest_pdf()

    pos_scores, neg_scores, rows = [], [], []
    for query, expect_match in eval_set:
        result = _kb_collection.query(query_texts=[query], n_results=1, where={"type": "chunk"})
        top_sim = round(1 - result["distances"][0][0], 3)
        rows.append((query, expect_match, top_sim))
        (pos_scores if expect_match else neg_scores).append(top_sim)

    if pos_scores and neg_scores and min(pos_scores) > max(neg_scores):
        threshold = round((min(pos_scores) + max(neg_scores)) / 2, 3)
    else:
        # No clean separation in this eval set — the classes overlap, so any
        # single threshold will misclassify something. Fall back to the
        # midpoint of the overall score range rather than pretending a clean
        # cutoff exists.
        all_scores = pos_scores + neg_scores
        threshold = round((min(all_scores) + max(all_scores)) / 2, 3) if all_scores else 0.6

    SIMILARITY_THRESHOLD = threshold

    if verbose:
        print(f"\n--- calibrating similarity threshold from {len(eval_set)} labeled queries ---")
        for query, expect_match, score in rows:
            label = "match   " if expect_match else "no-match"
            predicted = "match" if score >= threshold else "no-match"
            flag = "" if (score >= threshold) == expect_match else "  <-- MISCLASSIFIED at this threshold"
            print(f"    expected={label}  sim={score:.3f}  predicted={predicted}{flag}  {query!r}")
        if pos_scores:
            print(f"    lowest true-match score:      {min(pos_scores):.3f}")
        if neg_scores:
            print(f"    highest true-non-match score: {max(neg_scores):.3f}")
        print(f"    calibrated threshold: {threshold:.3f}")

    return threshold
```

Three design decisions to read closely.

**It scores through the real pipeline.** Real collection, real embedding function, top-1 similarity per query. Calibrating against anything else, a different embedding model, hand-picked chunks, measures a system you don't run.

**The happy path takes the midpoint of the actual margin.** Lowest true-match score on one side, highest true-non-match on the other; if they separate, the midpoint centers the threshold in the gap the data demonstrated. Not a round number that feels right. The gap your corpus actually has.

**The unhappy path refuses to lie.** If the classes overlap, and the 0.598/0.639 example shows they can, there is no threshold without misclassification, and the code says so instead of manufacturing false precision: fall back to the overall midpoint and, critically, *print the full classification table with every misclassification flagged*. That verbose table is not decoration. It's the difference between "the guardrail is calibrated" as a claim you can inspect and as a vibe. When a labeled query lands on the wrong side, you see exactly which one and by how much, and you can decide whether that residual risk is acceptable, or whether the eval set needs to grow. Ten queries is a starting point, not a destination; the docstring's own advice is to keep recalibrating as real queries accumulate.

## Enforcing the Threshold

`search_knowledge_base` gets its filter. The changed portion:

```python
    if SIMILARITY_THRESHOLD is not None:
        kept = [r for r in results if r["similarity"] >= SIMILARITY_THRESHOLD]
        if not kept:
            return {
                "results": [],
                "note": (
                    f"no chunk met the calibrated similarity threshold "
                    f"({SIMILARITY_THRESHOLD:.3f}); best score was "
                    f"{results[0]['similarity']:.3f} — likely not covered in this document"
                ),
            }
        results = kept

    return {"results": results}
```

The rejection message is engineered, not incidental. It tells the model three things: nothing cleared the bar, what the bar was, and how close the best miss came. "Best score was 0.541 against a 0.62 threshold" reads very differently from "best score was 0.615," and giving the model that texture, plus the explicit hypothesis "likely not covered in this document," is what a bare empty list wouldn't do. A tool that can say *no, and here's why* is doing work that otherwise lands on the model's judgment, which is the whole program of this book.

Guarding with `None` keeps the guardrail switchable: comment out the calibration call and you're back to Chapter 4 behavior, which is genuinely useful for demonstrating to a skeptic what the threshold buys.

And `main` grows its calibration step, before any agent runs:

```python
if __name__ == "__main__":
    ingest_pdf()

    # Embedding guardrail: calibrate the similarity threshold against the
    # labeled eval set before running anything through the model. Comment
    # this out to see the earlier, unguarded behavior for comparison.
    calibrate_similarity_threshold()

    prompts = [
        "How many PTO days do new employees get, and does it change over time?",
        "What's our company's policy on parental leave?",  # confirmed not in this handbook — should say so, not guess
    ]
    for p in prompts:
        print(f"\nUSER: {p}")
        print(f"AGENT: {run_agent(p)}")
```

## Run It

The calibration table prints first, ten rows, the margin, the threshold. Then the parental leave question does something new: `search_knowledge_base` comes back with an empty result and the rejection note, and the model, holding an explicit "likely not covered" instead of three tempting almost-answers, says the handbook doesn't cover it. The citation guardrail from Chapter 6 prints its pass with `retrieved: none`. The layers are starting to work as a system.

Mostly. Run enough no-answer questions and you'll notice something off in the traces: sometimes the model takes that clear rejection note and, instead of concluding, *searches again*. Different phrasing, same rejection. Then again. A guardrail built to stop weak answers is, in a fraction of runs, manufacturing an agent that can't stop searching. That interaction, two correct guardrails composing into a new failure, is Chapter 8, and it's the most instructive thing in Part 2.

## The Code Changed This Chapter

Four edits, in order. The rest of the file matches the Chapter 4 listing plus Chapters 5 and 6.

**Edit 1: eval set, threshold global, and calibration function.** Insert at module level, directly after `ingest_pdf()` and before `search_knowledge_base()`: the `EVAL_SET` list, the `SIMILARITY_THRESHOLD = None` line, and the full `calibrate_similarity_threshold()` function, all exactly as shown above.

**Edit 2: enforcing the threshold.** In `search_knowledge_base()`, replace the final line:

```python
    return {"results": results}
```

with the threshold-filter block from "Enforcing the Threshold" above (the `if SIMILARITY_THRESHOLD is not None:` block, ending in the same `return {"results": results}`).

**Edit 3: calibrate at startup.** In the `__main__` block, insert one line directly after `ingest_pdf()`:

```python
    calibrate_similarity_threshold()
```

**Edit 4: sharpen the demo prompts.** In the `prompts` list, make the PTO question two-part so it exercises the grounding rules, and update the parental leave comment to reflect its new role as a threshold probe:

```python
    prompts = [
        "How many PTO days do new employees get, and does it change over time?",
        "What's our company's policy on parental leave?",  # confirmed not in this handbook — should say so, not guess
    ]
```

(The next full-file checkpoint lands at the end of Chapter 9, once Part 2's remaining two guardrails are in.)
