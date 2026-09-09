# Self-RAG

Status: complete — 7-step build + a web-search branch.

**Core idea** (Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*): instead of a fixed graph deciding retrieval logic from the outside (CRAG's approach), a single model decides **for itself**, inline, via special reflection tokens:

- `Retrieve` — does this step even need retrieval, or can the model answer from what it already knows?
- `ISREL` — is each retrieved passage actually relevant?
- `ISSUP` — is this generated segment actually supported by the passage it cites?
- `ISUSE` — is the overall response useful / well-formed?

**How this differs from CRAG:** CRAG grades retrieval *after the fact* with an external evaluator and reroutes the pipeline. Self-RAG asks the generator to grade *itself*, before and during generation, and skip retrieval entirely when it isn't needed.

- **`documents/`** — a small internal-company corpus (`Company_Policies.pdf`, `Company_Profile.pdf`, `Product_and_Pricing.pdf`) — deliberately different from CRAG's ML textbooks, since Self-RAG's grounding checks are easiest to see against short, factual policy/pricing text.
- **`code/`** — `self_rag_step1.ipynb` → `self_rag_step7.ipynb`, each one adding exactly one reflection-token capability on top of the last, plus `self_rag_web.ipynb`, a side-branch that swaps the internal "no relevant docs" dead-end for a web-search fallback.
- **`notes/process.md`** — Mermaid-diagram revision notes (renders on GitHub).
- **`notes/process.html`** — same notes, fully styled standalone page. Open directly in a browser.

One-line idea: before answering, decide *whether* to retrieve at all; after answering, check whether the answer is actually **supported** by what was retrieved and whether it's actually **useful** — revise or re-retrieve until both pass (or a retry budget runs out).
