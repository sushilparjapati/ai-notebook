# Advanced RAG

A complete, hands-on walk through Retrieval-Augmented Generation — from document loading to
production deployment — plus four original implementations and an end-to-end deployed capstone.

## Curriculum

| Folder | Topic |
|---|---|
| `01_document_loaders` | Loading documents from various sources (PDF, Web, JSON etc.) |
| `02_text_splitters` | Chunking strategies to split documents into meaningful pieces |
| `03_embeddings` | Generating vector embeddings using different embedding models |
| `04_vector_stores` | Storing and querying embeddings using vector databases |
| `05_retrievers` | Core retrieval strategies to fetch relevant document chunks |
| `06_advanced_retrievers` | Parent-document, multi-query, self-query, and contextual compression retrievers |
| `07_rerankers` | Re-ranking retrieved results with cross-encoder models |
| `08_rag_fusion` | Combining multiple retrievers with fusion strategies like RRF |
| `09_rag_hyde` | HyDE — generating hypothetical answers to improve retrieval |
| **`10_crag`** | **Corrective RAG** — evaluates retrieved documents and corrects retrieval when they're not relevant |
| **`11_self-rag`** | **Self-RAG** — the model reflects on its own retrieval and generation |
| **`12_agentic_rag`** | **Agentic RAG** — an LLM autonomously decides when and how to retrieve |
| **`13_graph_rag`** | **Graph RAG** — knowledge graphs for structured information retrieval |
| `14_rag_multimodal` | Multimodal RAG handling text, images, and tables in a single document |
| `15_rag_evaluation` | Evaluation frameworks (RAGAS) and metrics to measure RAG pipeline performance |
| `16_rag_guardrails` | Input, context, and output guardrails over a RAG pipeline |
| `17_rag_paper_project` | Capstone — see below |

Folders are numbered as a recommended learning path; `_archive/` holds earlier versions of two
topics, kept for reference only.

**Modules 10–13 are original implementations**, not sourced from course material — Corrective
RAG, Self-RAG, Agentic RAG, and Graph RAG built from the ground up.

## Capstone: `17_rag_paper_project`

An end-to-end research-paper assistant with fact and claim verification, deployed rather than
just prototyped:

- **Stack:** Dockerized, deployed to AWS, Qdrant for vector storage, streaming responses
- **Evaluation:** measured with DeepEval across faithfulness, relevancy, and precision/recall
  metrics on a held-out golden set — not just "it runs," but scored

## Getting started

Each folder is self-contained. Navigate into any topic and follow its own README:

```bash
cd advanced-rag/05_retrievers
uv sync   # or: pip install -r requirements.txt
```

Set your LLM provider's API key as an environment variable before running any notebook or script.

## Credit

Built on top of [Advanced_Rag_Codes](https://github.com/Himanshu-1703/Advanced_Rag_Codes)
(MIT licensed) as a learning foundation, then extended with original implementations
(modules 10–13), restructured topic-by-topic revision notes and cheat-sheets, and the capstone
project above.
