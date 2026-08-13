# Agentic AI from Scratch

### Build a Local Agent with Ollama and Python, from the Loop to Guardrails to MCP

<br>

**Rob Klause**

<br>

*First edition — August 2026*

Copyright © 2026 Rob Klause. All rights reserved.

Written and edited in collaboration with Claude (Sonnet 5 taught the material; Fable 5 helped write it down). All code in this book was built and run against local models via Ollama, on hardware the author actually owns.

The Marlow & Sage Boutique employee handbook used throughout is a fictional document created for this book. Any resemblance to a real boutique, its policies, or its opinions on rubber flip-flops is coincidental.

---

# Contents

**Front Matter**

- [Introduction](00-front-matter.md)
- [About This Book](00-front-matter.md)
- [Who This Is For](00-front-matter.md)
- [Why Local-First](00-front-matter.md)
- [What You'll Build](00-front-matter.md)
- [How to Read This Book](00-front-matter.md)
- [How Code Edits Are Specified](00-front-matter.md)

**Part 0 — Setup**

- [Chapter 1: Tools and Installation](01-setup.md)
  - Python 3.10 or Newer. This Is Not Optional. · VS Code · Ollama · The Three Models · Hardware, and What to Expect · The Python Packages · The Knowledge Base PDF · Project Layout · Smoke Test

**Part 1 — The Loop and Tool Calling**

- [Chapter 2: What Is an Agentic Loop](02-the-loop.md)
  - One Model Call, No Tools · What a Tool Call Looks Like on the Wire · The Loop · Why "Agentic" Is Just This
- [Chapter 3: Your First Tools](03-first-tools.md)
  - A Tool Is Two Things · The First Tool: calculate · The Second Tool: compare, and Why It Exists · The Schemas · Dispatch, and the Reinforcement in the System Prompt · Run It · **Checkpoint: the complete file**
- [Chapter 4: Giving the Agent a Knowledge Base](04-knowledge-base.md)
  - The Problem Worth Solving · Why Keyword Search Isn't Enough · Embeddings in One Paragraph · The Ingestion Pipeline · The Search Tool · The Bill Comes Due: Context Bloat · Run It · **Checkpoint: the complete file**

**Part 2 — Guardrails**

- [Chapter 5: Why Guardrails, Not Just Better Prompts](05-why-guardrails.md)
  - The Failure: Silence Becomes a Stated Negative · Why "Just Fix the Prompt" Underdelivers · The Principle That Organizes Part 2
- [Chapter 6: Output Guardrails, Verifying Citations](06-output-guardrails.md)
  - The Gap Being Closed · Tracking What Was Actually Retrieved · The Check, and the Trap Inside the Obvious Version · Design Decisions Worth Defending · Run It
- [Chapter 7: Embedding Guardrails, Calibrating a Similarity Threshold](07-embedding-guardrails.md)
  - The Loose Thread From Chapter 4 · Why You Don't Guess the Threshold · The Eval Set · The Calibration Function · Enforcing the Threshold · Run It
- [Chapter 8: Designing Guardrails That Work Together](08-guardrails-together.md)
  - Two Correct Guardrails, One New Failure · The Prompt Request, and Why It's Not Enough Alone · The Structural Backstop · The Second Composition Bug: Junk Chunks · The Stale-Cache Trap · Run It
- [Chapter 9: Input Guardrails, Screening What Comes In](09-input-guardrails.md)
  - Two Different Problems Wearing One Name · Why Position Beats Cleverness · Pass 1: The Pattern List · Pass 2: A Model Whose Only Job Is Judging · Wiring Both Passes In · Run It · **Checkpoint: the complete file (Part 2)**
- [Chapter 10: What a Bigger Model Changes (and What It Doesn't)](10-bigger-models.md)
  - What Actually Improves · The Rates Move. The Modes Don't. · The Gap This System Knowingly Ships With

**Part 3 — MCP**

- [Chapter 11: Where MCP Actually Fits](11-where-mcp-fits.md)
  - The Problem MCP Standardizes · One Seam, The Same Seam As Always · What This Buys You in Practice · The Plan for Part 3
- [Chapter 12: Building a Local MCP Server](12-building-mcp-server.md)
  - First, the Five-Minute Local Version · The Server · Run It by Hand, Once · The Gotcha: Verify the Result Shape Against a Live Server
- [Chapter 13: Merging Local and Remote Tools](13-merging-local-and-mcp.md)
  - The One Real Wrinkle: Async, and Connection Lifetime · New Imports and the Server Handle · Discovery: Schemas From the Wire · Invocation: Normalizing What Comes Back · The Loop: Four Small Changes · Startup: One Connection, Everything Inside It · Run It · **Checkpoint: the complete final file**

**Part 4 — Wrap-Up**

- [Chapter 14: Tips and Lessons Learned](14-tips.md)
  - On Model Behavior · On Guardrails · On Retrieval · On Engineering Practice · The One-Sentence Version
- [Chapter 15: Going Further](15-going-further.md)
  - Swapping in a Cloud Model · More MCP Servers · Persisting Conversations · Streaming · Multi-Agent Architectures · Frameworks: When to Adopt One, and How to Read One · Evals: Growing the Ten-Query Habit · Where to Keep Reading

**Appendices**

- [Appendix A: The Complete Code](appendix-a-code.md)
  - agentic_demo.py, Top to Bottom · mcp_weather_server.py · The System Prompt, Rule by Rule
- [Appendix B: Troubleshooting](appendix-b-troubleshooting.md)
  - Installation and Environment · Ollama and Models · Retrieval and the Vector Store · The Loop and Guardrails · MCP · When Nothing Here Fits
- [Companion Code: Chapter Checkpoints](code/README.md)
  - Runnable checkpoint files for every code-changing chapter, one continuous edit lineage from Chapter 3 to the final system
