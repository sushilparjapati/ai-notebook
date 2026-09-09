# Agentic RAG

Status: complete — 7-step build (00 → 05, plus a final combined notebook).

**Core idea:** replace the fixed retrieve → generate (or retrieve → evaluate → correct → generate) *graph* with an LLM **agent** that plans its own steps. Instead of hardcoded routing (like CRAG's `route_after_eval`), the model itself decides, in a loop:

- whether to retrieve at all, and from which source (internal vector store vs. live web search)
- how many retrieval rounds it needs before it has enough to answer, and when to stop
- how to re-plan when retrieved documents turn out irrelevant (rewrite the query and retry)
- whether a question is actually several distinct questions that need separate retrieval passes

**How this differs from CRAG:** CRAG is a fixed graph with one branch point (the verdict); the "intelligence" lives in an external evaluator, not in open-ended planning. Agentic RAG's graph is a loop the model itself controls via real tool-calling — it can retrieve 0, 1, or N times, from either of two tools, in whatever order it decides, based on its own reasoning at each step.

- **`documents/`** — `evs_oil_price_shock.pdf`, a technical report on EV adoption and oil price dynamics — deliberately a single, dense domain document so the agent's tool-choice and multi-round retrieval behavior are easy to observe.
- **`code/`** — `00_base_rag.ipynb` → `05_query_decomposition.ipynb`, each adding exactly one new agentic capability, plus `agentic_rag.ipynb`, the final combined build.
- **`requirements.txt`** — extra deps beyond the shared root ones (`langchain-chroma`, `chromadb`, `tavily-python`).
- **`notes/process.md`** — Mermaid-diagram revision notes (renders on GitHub).
- **`notes/process.html`** — same notes, fully styled standalone page. Open directly in a browser.

One-line idea: instead of a fixed pipeline, give the model two tools (internal search, web search) and let it call either, both, or neither, as many rounds as it needs — with a retry budget, a relevance check, a query rewrite, and query decomposition for multi-part questions layered on top.
