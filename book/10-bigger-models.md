# Chapter 10: What a Bigger Model Changes (and What It Doesn't)

By the end of this short chapter you'll be able to answer the question every stakeholder asks about systems like this one: "wouldn't a better model make all that guardrail work unnecessary?" The answer is no, and the reasons are worth having crisp, because you'll be giving them in meetings for years. No code changes in this chapter, which is itself the point.

## What Actually Improves

Be fair to the question first, because a frontier-scale cloud model genuinely does improve things. Raw error rates drop across the board: fewer ignored instructions, fewer hallucinated qualifiers, better search phrasing, more reliable tool-call formatting, better adherence to the citation format. The 9B model on your laptop ignores the search-discipline instruction in some fraction of runs; a frontier model ignores it in a smaller fraction. Every failure *rate* this book has discussed moves in the right direction.

Latency and context capacity improve too. And the swap itself, as Chapter 15 shows concretely, is three lines: base URL, API key, model name. Everything you've built rides along unchanged, which is exactly why this question is worth asking. The upgrade is cheap. So why keep the guardrails?

## The Rates Move. The Modes Don't.

Because a bigger model reduces every failure *rate* and eliminates no failure *mode*. Frontier models still hallucinate, still convert silence into stated negatives, still garnish grounded answers with helpful unsupported qualifiers. Less often. Not never. And your guardrails aren't priced in "often": the citation check exists for the one answer in fifty with an invented page, and one-in-fifty at production volume is a daily event.

There's a second effect that's less obvious and more important: **a better model's failures are harder to catch, not easier.** The 9B model's hallucinations are often clumsy, odd phrasing, formatting drift, non sequiturs, the kind of thing a human reviewer's eye snags on. A frontier model's rare hallucination arrives polished: fluent, well-structured, confident, wearing the same style as its hundred correct neighbors. The signal your reviewers used to catch errors *degrades* as the model improves, at exactly the moment rising trust makes everyone review less carefully. Structural checks don't have that problem. `_check_citation_grounding` verifies set membership identically whether the answer came from a 9B model or the largest thing money rents, and that indifference to fluency is precisely what makes it valuable when fluency stops being a tell.

So the honest engineering statement, the one to give the stakeholder: model quality and guardrails aren't substitutes; they're multiplicative. The model sets the error rate, the guardrails determine what escaping errors can *do*. Upgrade the model to make flags rarer. Keep the guardrails so that "rarer" never silently becomes "invisible."

It also cuts the other way, and it's worth saying in the same breath: the guardrails are what make the *small* model usable. An unguarded 9B agent is a liability; the guarded system you built is trustworthy enough to demo, precisely because its checkable failures get checked. Guardrails don't just protect you from a model's weaknesses. They lower the model quality you need, which on this book's economics means the difference between free-on-your-laptop and metered.

## The Gap This System Knowingly Ships With

Part 2 closes with a disclosure rather than a triumph, because the discipline of writing down what your guardrails don't catch is worth more than any individual guardrail.

Here it is: ask this agent a numeric question it decides to answer from pure training memory, no tool calls, no retrievals, no citations, and a fabricated number sails through everything. Walk the layers to see why. Input guardrails screen for injection, not fabrication. The similarity threshold never fires because search was never called. The backstop counts empty searches; there were none. The citation check finds no unverified citations, and its zero-citation clause only triggers when `retrieved_pages` is non-empty, which it isn't. Every layer reports clean. The answer is invented.

The system prompt does address this, "if you haven't called a tool for a number, leave it out," and by now you can name that instrument on sight: a request. The structural version would be a check that numeric claims in the final answer trace to tool results, and the source system deliberately declined to build it. That's a defensible judgment call, not an oversight: Chapter 6 already walked the reasoning, a model paraphrasing "10 days per year" as "ten days annually" false-positives any naive text-match, so a trustworthy version means either fragile matching that cries wolf or another model-as-judge layer with its own cost and its own failure modes. For this system's threat profile, a documented gap beat a noisy guardrail.

The principle to carry out of Part 2 is the ledger itself. Every guardrail chapter in this book ended by naming what the new layer doesn't catch. That habit, an explicit, current list of known gaps, with reasoning, is what separates a system whose risks are *chosen* from one whose risks are discovered by users. When the next stakeholder asks "is it safe?", the strong answer isn't "yes." It's "here's what's checked, here's what isn't, and here's why we drew the line there." You now have that answer for this system, and the template for writing it about any other.

## The Code Changed This Chapter

Nothing. The complete file stands as it did at the end of Chapter 9, and that's the chapter's thesis in one line: the argument for guardrails doesn't change when the model does.

Part 3 changes something else entirely: where tools come from. The weather tool is about to leave the building.
