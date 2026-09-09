# Corrective RAG (CRAG)

Status: complete — 6-step build, fully worked.

- **`code/`** — `1_basic_rag.ipynb` → `6_ambiguous.ipynb`, each notebook adds exactly one new piece (evaluator → web search → query rewrite → ambiguous merge) on top of the previous one.
- **`documents/`** — CRAG's own PDF corpus (ML/DL textbooks).
- **`notes/process.md`** — Mermaid-diagram revision notes (renders on GitHub).
- **`notes/process.html`** — same notes, fully styled standalone page. Open directly in a browser.

One-line idea: grade retrieved chunks, then route to *trust internal* / *replace with web search* / *merge both* depending on the grade — never answer from unchecked retrieval.
