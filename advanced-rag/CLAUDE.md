# rag/ — conventions

This directory holds:

- **`01_document_loaders/` … `17_rag_paper_project/`** — one numbered sequence, worked through in order, ending with the capstone project. Most came from an upstream course clone (`Advanced_Rag_Codes`, see `rag/README.md`) and started as `notebooks/` + `main.py` with no `notes/`.
- **`10_crag/`, `11_self-rag/`, `12_agentic_rag/`, `13_graph_rag/`** — Sushil's own implementations, promoted into the sequence from kebab-case names. These already use the target layout.
- **`_archive/agentic_rag/`, `_archive/graph_rag/`** — the original course versions of topics 12 and 13, kept out of the numbered sequence. Superseded; don't work on them unless asked.

## Target folder layout

Folders `10_crag/` through `13_graph_rag/` already use this shape. It's the convention to apply to any folder being actively worked on, numbered or not:

```
<topic>/
├── README.md          — folder structure only: a title and the file tree, nothing else
├── code/               — notebooks (renamed from notebooks/) + any .py scripts
├── documents/          — source files the notebooks load (renamed from knowledge-source/ etc.)
├── notes/
│   ├── <topic>-notes.md              — revision notes (see below)
│   └── <topic>-cheatsheet.drawio.png — optional compact reference image (drawio-skill)
├── requirements.txt / pyproject.toml / uv.lock  — whatever the folder already had
```

**Migration status:** `01_document_loaders` through `09_rag_hyde` are converted (each has `code/`, `notes/` with revision notes + a cheat-sheet, and a structure-only README). `10_crag` – `13_graph_rag` already arrived in this shape; `13_graph_rag` also has `notes/graph-rag-notes.md` + `notes/images/`. `14_rag_multimodal`, `15_rag_evaluation` and `16_rag_guardrails` are converted (`notebook(s)/` → `code/`, `data/` → `documents/`, root `.py` files moved into `code/`, revision notes added; no cheat-sheets yet). Only `17_rag_paper_project` — the capstone — is still in the original clone shape.

**Numbering note:** guardrails and the capstone project were swapped so the project ends the sequence — what was `17_rag_guardrails` is now `16_rag_guardrails`, and `16_rag_paper_project` is now `17_rag_paper_project`. Migrate a folder only when actively asked to work on it — don't do it speculatively.

When migrating, check for path references that the rename breaks: notebook constants like `../data/<file>` become `../documents/<file>`, and `.gitignore` entries like `notebook/figures/` become `code/figures/`. Where root-level `.py` scripts used **CWD-relative** paths (`data/foo.pdf`), moving them into `code/` breaks them silently — anchor to the file instead, as `15_rag_evaluation/code/rag_pipeline.py` does:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "documents" / "sustainable_development.pdf")
```

## Revision notes (`notes/<topic>-notes.md`)

Source material for these notes is `rag/Notes/*.pdf` — Sushil's own handwritten/slide notes covering the whole RAG curriculum in one PDF per broad topic (e.g. `Advanced RAG.pdf` covers document loaders through RAGAS). **Never duplicate a PDF's content into `Notes/` itself** — extract only the section relevant to the folder being worked on, into that folder's `notes/`. The PDFs are often scanned/handwritten slides with no extractable text — render the relevant page range to images and read them directly rather than relying on `pdftotext`.

Where the source PDF has gaps (e.g. a "Code" slide with no actual code), pull code and comparisons from the folder's own notebooks in `code/` instead of inventing content.

Style, established with `01_document_loaders/notes/document-loader-notes.md`:

- Optimized for revision the day before an interview: short headings, bullets over paragraphs, tables for anything comparison-shaped.
- A **TL;DR** near the top — wide, multi-column tables (not a single narrow label/value column) so it uses the page width.
- Mermaid diagrams for pipeline/flow/hierarchy content (GitHub renders these natively) instead of ASCII art.
- Code snippets in collapsible `<details>` blocks, one per loader/technique, so the page doesn't turn into a long scroll.
- No meta/provenance text in the doc itself (no "extracted from page X of PDF Y" headers) — the notes should read as standalone reference material.
- Careful with bold/code spans inside table cells: `**word**` and `` `word` `` must hug their content tightly — a stray space glues the marker to adjacent text on GitHub (`the**bold**` renders wrong). Verify after any bulk find-and-replace across a table-heavy file.
- Close with a **Quick recap** section (can restate the TL;DR/diagrams more tersely — repetition is fine here, it's a revision doc).

## Cheat-sheet images (optional)

For a single-page, at-a-glance visual, use the `drawio-skill` to produce `<topic>-cheatsheet.drawio.png` in the `notes/` folder, alongside `<topic>-notes.md`. Keep it dense and table/grid-based — a compact reference card, not a sprawling flowchart.

**Keep only the `.png`, not the `.drawio`.** The draw.io export embeds the diagram source in the PNG itself (a `zTXt` chunk keyed `mxGraphModel`), so the image can be dragged back into draw.io and edited — the separate `.drawio` file is redundant. Delete it after exporting. The one exception is a multi-page diagram, which the embedded chunk can't carry; keep the `.drawio` in that case.

**Do not embed the cheat-sheet in `<topic>-notes.md` or in `README.md`.** They are separate deliverables that live side by side in `notes/`: the markdown is the readable revision doc, the PNG is the standalone glance-card. The README stays a bare structure tree.

This applies to the cheat-sheet only. **Diagrams from the source PDF may be embedded in `<topic>-notes.md`** — see below.

## Source diagrams from the notes PDFs

The `rag/Notes/*.pdf` slides usually have clean, non-handwritten diagrams underneath the ink. Extract them rather than rendering the whole page: `pdfimages -list -f <p> -l <p> <pdf>` to see what's embedded, `pdfimages -png -f <p> -l <p> <pdf> <prefix>` to pull them out. This gives the original screenshot **without** the handwritten annotations on top.

Store them in `notes/images/` with descriptive names and embed with a relative path (`![alt](images/<name>.png)`).

Only embed genuine **diagrams**. Slide screenshots that are really code, schema dumps or tables should be transcribed into markdown/code blocks instead — text is selectable, theme-aware and reflows; a fixed-width screenshot of a table is worse on every count. Where a Mermaid redraw already covers the same diagram, keep the Mermaid as primary (it renders in both light and dark) and put the original slide image in a collapsed `<details>` beneath it.
