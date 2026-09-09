# Graph RAG

Status: complete — 2-notebook build (ingestion + retrieval).

**Core idea (as actually implemented here):** instead of indexing flat text chunks in a vector store, extract **entities and relationships** from each chunk with an LLM (`LLMGraphTransformer`), store them as a real property graph in **Neo4j**, and answer questions by having an LLM write **Cypher** queries that traverse that graph (`GraphCypherQAChain`) — rather than by nearest-neighbor similarity search over chunks.

> Note: this is a different (and more common/practical) flavor of "Graph RAG" than Microsoft's community-summary variant (entity graph + Leiden clustering + hierarchical summaries, aimed at *global* "what are the themes in this corpus" questions). This implementation is graph-**traversal** based, aimed at *multi-hop* factual questions — e.g. "what product did the organization whose board Person X left release?" — where the answer requires chaining several relationships that no single chunk states in one place.

**How this differs from CRAG:** CRAG corrects *which chunks* to trust from a normal vector index. Graph RAG changes the index itself — from flat chunks to an entity/relation graph — which changes what kinds of questions retrieval can even answer (multi-hop relationship chains vs. single-passage facts).

- **`documents/`** — `elon_musk.pdf`, a short 2-page biography, deliberately dense with named entities and relationships (people, companies, dates, acquisitions) so the extracted graph is rich enough to demonstrate multi-hop traversal.
- **`code/`** — `01_ingestion.ipynb` (load → chunk → `LLMGraphTransformer` → store in Neo4j → build a vector index over the `Document` nodes) → `02_retrieval.ipynb` (`GraphCypherQAChain` — natural language question → generated Cypher → graph traversal → answer).
- **`requirements.txt`** — extra deps beyond the shared root ones (`langchain-neo4j`, `langchain-experimental`).
- **`SETUP.md`** — step-by-step instructions for creating a free Neo4j AuraDB instance and wiring its credentials into `.env` (`NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`) — **required** before running `01_ingestion.ipynb`.
- **`notes/process.md`** — Mermaid-diagram revision notes (renders on GitHub).
- **`notes/process.html`** — same notes, fully styled standalone page. Open directly in a browser.

One-line idea: turn a document into a graph of entities and relationships, then answer questions by generating a graph query — not a similarity search — so multi-hop questions ("who is X's partner, and where are *they* a director?") that no single chunk answers become directly traversable.
