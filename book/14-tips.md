# Chapter 14: Tips and Lessons Learned

This chapter is the book compressed for re-reading. Every entry is a standalone takeaway with the chapter that earned it, written to be useful a year from now when you're building something that looks nothing like a handbook agent. They're grouped, loosely, from model behavior outward to engineering practice.

## On Model Behavior

**"Not mentioned" is not "confirmed no."** The most dangerous RAG failure isn't invention from nowhere; it's the model converting a document's silence into a stated negative. "The chunk doesn't mention tenure changes" becomes "the rate doesn't change." It looks grounded, the retrieval happened, the citation is real, and the claim is manufactured. Screen for it in review, write prompt rules against it, and assume some of it survives anyway. (Chapter 5)

**A prompt fix built from one example teaches the example, not the rule.** Ban one specific unstated negative and the model finds a neighboring one you didn't enumerate. If you must teach a general rule by prompt, state it *as* a general rule, say explicitly that it isn't limited to the example, and sketch the category space it covers. Expect reduction, not elimination. (Chapter 5)

**Audit what judgment remains after you delegate.** A model given a calculator will compute 3 × 68 with the tool and then eyeball whether 101 exceeds 204 in its head. Every delegation has a residue: the comparison after the calculation, the qualifier after the citation. Finding and removing that residue, one judgment at a time, is what this book's guardrails have in common. (Chapters 3, 5, 6)

**Bigger models lower failure rates and eliminate no failure modes.** And their rarer failures arrive more fluent, which makes them harder to catch by eye exactly as rising trust makes everyone look less carefully. Structural checks don't care about fluency; that indifference is their value. Model quality and guardrails are multiplicative, not substitutes. (Chapter 10)

## On Guardrails

**Prompts are requests. Code is a guarantee.** The design test for every reliability rule you write: if the model ignored this sentence, what catches it? If the answer is nothing, you have a request where you need a structure. The strongest structures don't detect the bad choice, they remove it from the menu, the way `tools=None` makes another search call impossible rather than discouraged. (Chapters 5, 8)

**What you can check syntactically, fix with code; what requires judging meaning, a prompt can only make less frequent.** Cited page in the retrieved set: code. Similarity above threshold: code. Two empty searches in a row: code. "Is this a reasonable inference or a fabrication": prompt, and honesty about the residual rate. This one principle sorts nearly every reliability decision you'll face. (Chapter 5)

**A guardrail that prints "passed" may have verified nothing.** A citation checker passes an answer with zero citations; nothing cited, nothing to flag. Green lights that had nothing to check are false confidence with good UX. For every check you build, ask what makes it *vacuously* pass, and either close that path or flag it separately. (Chapter 6)

**Prefer a guardrail that's precise about something narrow over one that's mushy about something broad.** Citation verification checks set membership exactly and says nothing about paraphrase faithfulness, deliberately, because a faithfulness checker would false-positive on honest restatements and teach you to ignore it. You can trust a narrow check's output. Trustworthiness is the entire product of a guardrail. (Chapter 6)

**Test guardrails together, not just alone.** Two individually correct layers composed into an infinite search loop: the threshold changed what "no answer" looked like, and the model responded by retrying forever. Component tests validate components. The failures live in the interactions, so your test cases must exercise the layers as a system, including designed probes for each layer in your standard run. (Chapter 8)

**A guardrail can look validated by luck.** The termination backstop appeared to work while a junk chunk was quietly preventing it from ever firing; most test queries just didn't collide with the junk. When a protection "works," ask whether you've seen it fire for the reason you think it fires. Designed probe inputs, built to trigger each specific layer, are the difference. (Chapter 8)

**A backstop keyed to a specific signal has a blind spot the failure can dodge; pair it with an unconditional bound.** The empty-search counter never fires when reformulated queries keep matching semi-relevant chunks, and in a topically coherent corpus they usually do, so the counter can count correctly forever while the turns burn. The countermeasure triggers on the turn number alone: the final turn never offers tools, so the last turn is always an answer turn. When termination matters, at least one layer should fire on count or time, not on a condition. (Chapter 8)

**Calibrate thresholds from labeled data scored by the real pipeline; never eyeball them.** Embedding similarity has no natural "no match" score: in a topically coherent corpus, everything is somewhat related, and a topic absent from the document can outscore one that's present (0.639 vs 0.598 in one real calibration). Build a small eval set, verify the labels against the actual corpus, make the negatives *plausible* questions rather than nonsense, and print the classification table with misclassifications flagged. Recalibrate as real queries accumulate. (Chapter 7)

**Keep a written ledger of what your guardrails don't catch.** This system knowingly ships with one: a fabricated number with zero tool calls and zero citations passes every layer. The gap is documented, reasoned, and chosen, which is what separates a system whose risks are decided from one whose risks are discovered by users. "Here's what's checked, what isn't, and why" is a stronger answer than "yes, it's safe." (Chapters 6, 10)

**LLM-as-judge is a different layer, not a better one.** A guard model understands intent a regex can't, and costs latency, nondeterminism, and a new attack surface, since the judge itself reads attacker text. The trust boundary moves; it doesn't disappear. Layer them by economics: free deterministic checks first, the model judge only on what clears them. And decide fail-open versus fail-closed deliberately, in writing; the right answer differs between a laptop demo and a public endpoint. (Chapter 9)

## On Retrieval

**Embed documents and queries with the same function, structurally.** Two embedding models produce two vector spaces, distances between them are meaningless, and nothing errors: scores still look like numbers. Route both sides through one piece of code so the mistake is hard to write. (Chapter 4)

**Very short chunks are a correctness bug, not clutter.** A two-or-three-word chunk scores inflated similarity against almost anything sharing common vocabulary (a bare "EMPLOYEE HANDBOOK" fragment hit 0.827 against an unrelated query). One junk chunk clearing your threshold can disarm guardrails layered far above it. Filter by minimum length before embedding, and strip boilerplate that repeats on every page. (Chapters 4, 8)

**Key derived-data caches on the logic that derives, not just the source bytes.** Change your chunking rules without changing the PDF and a bytes-keyed cache will silently serve stale chunks while you believe your fix is live. Fold a version constant into the hash and bump it with the logic. (Chapter 8)

## On Engineering Practice

**Tools return errors as data, never as exceptions.** A raised exception kills your loop; an `{"error": ...}` dict goes back to the model, which can read it and recover. The tool boundary talks to a counterparty that can't catch. (Chapter 3)

**Schema descriptions are prompt engineering.** The description field is not documentation; it's behavioral instruction, and the difference between a tool used correctly and ignored can live entirely in that string ("Do not judge comparisons yourself; always call this tool instead"). This holds double for MCP, where the server author writes the description that steers *your* model. (Chapters 3, 11, 12)

**The docs describe the happy path; the running system defines the contract.** An MCP tool returning a plain dict leaves `structured_content` as `None`, with the JSON hiding in a text block, which you only learn by calling the live server. Same lesson as confirming a guard model's output tag before trusting your parser. Ten minutes against a live endpoint beats an afternoon debugging an assumption, and it applies at every protocol boundary you don't own. (Chapters 9, 12)

**Open one connection for the life of the run, not per call.** Wrapping each MCP dispatch in its own `asyncio.run()` works, and silently respawns the server subprocess on every single tool call. Expensive handles (vector store clients, MCP connections) are process-lifetime resources, threaded through, exactly like a lazily-initialized global. (Chapter 13)

**Message lists have referential structure; edit them like it.** Every tool result is paired by id to the assistant message that requested it, and a naive slice can strand one half of the pair on the wrong side of a cut. Any history surgery, compaction, truncation, summarization, must walk to a clean seam first. (Chapter 4)

**Version mismatches surface as lies.** `pip install mcp` on Python 3.9 doesn't say "Python too old"; it says no matching distribution, which reads like a network problem. When an install error makes no sense, check the interpreter version before checking the network. (Chapter 1)

**Normalize at boundaries so the core stays ignorant.** `_call_mcp_tool` converts every wire outcome, success, error, unparsed text, into the same plain dict local tools return, which is why the loop never learned MCP exists. The pattern generalizes: adapters absorb difference at the edge; anything that leaks past the adapter becomes a permanent tax on the core. (Chapter 13)

**The trace is the product while you're building.** Every design decision in this book was made by reading raw request/response traces, and every guardrail announces itself in them. Verbose mode that prints whole messages costs nothing and is the only way to know what your agent actually did, as opposed to what its final answer implies it did. (Chapters 2, 3, and everywhere)

## The One-Sentence Version

An agent is a loop; tools are its hands; guardrails are code holding every judgment the model shouldn't hold alone; MCP is a socket where the hands plug in. Everything else in this book was the reasoning that makes those clauses safe to act on.
