---
tags: [learning, ai, rag, project]
created: 2026-11-20
status: not-started
---

# DocPilot: plan and architecture

A chat-with-your-documents app you build in 10 phases to learn LLMs, embeddings, RAG, LiteLLM, Langfuse and evals.

You drop in PDFs, CSVs and Markdown, ask questions in a React chat, and get answers with citations (file and page). CSV questions are answered with SQL, not guesses. Every model call goes through LiteLLM, and every step is traced in Langfuse.

**Time:** about 10–12 weeks at 6–8 h/week. Each phase ends with a working app.

---

## Progress

| Phase | Topic | Weeks | Status |
|---|---|---|---|
| 0 | Foundations | 1 | ✅ |
| 1 | Embeddings from scratch | 1–2 | ✅ |
| 2 | Ingestion pipeline | 2–3 | ⬜ |
| 3 | Naive RAG + React chat | 3–4 | ⬜ |
| 4 | Observability with Langfuse | 4–5 | ⬜ |
| 5 | Evaluation | 5–6 | ⬜ |
| 6 | Better retrieval | 6–8 | ⬜ |
| 7 | CSVs and structured data | 8–9 | ⬜ |
| 8 | Agentic RAG + MCP | 9–10 | ⬜ |
| 9 | LiteLLM proxy + production | 10–12 | ⬜ |

Change ⬜ → 🟨 (in progress) → ✅ (done) as you go.

---

## Before you start

- [ ] Pick a corpus you know well, so you can tell when an answer is wrong:
  - your own files (bank CSVs, contracts, manuals, payslips, invoices)
  - papers or docs on a topic you're learning
  - a public set, e.g. a company's annual reports (PDFs with tables) plus a public CSV
- [ ] Install: Python 3.12+, `uv`, Node 20+, Docker Desktop, Ollama
- [ ] Get an Anthropic API key (optional: OpenAI key for hosted embeddings)

---

## Architecture

![[docpilot-architecture.png]]

*(Dark version: [[docpilot-architecture-dark.png]])*

This is the **end state** after phase 9. Early phases use a subset: phases 0–2 call the LiteLLM Python SDK directly, and the LiteLLM proxy only appears in phase 9.

### Components

| Component | Role | Introduced in |
|---|---|---|
| **React** (Vite + TS, Tailwind, shadcn/ui, TanStack Query) | Chat UI with streaming, citations panel, document upload | Phase 3 |
| **FastAPI** | The only backend. RAG pipeline, routing, agent loop, upload endpoint | Phase 3 |
| **Postgres + pgvector** | Chunks, embeddings, metadata, full-text index | Phase 2 |
| **DuckDB** | CSVs loaded as SQL tables | Phase 7 |
| **Ingest worker** (Docling) | Parse → chunk → embed → store | Phase 2 |
| **LiteLLM** | One API for all models: SDK first, proxy later (keys, budgets, fallbacks, cache) | Phase 0 / 9 |
| **Langfuse** (self-hosted) | Traces, cost, prompt versions, datasets, experiment scores | Phase 4 |
| **eval.py** | Runs the golden question set, scores retrieval and answers | Phase 5 |
| **Claude** | Main chat model (`claude-sonnet-5-5`, `claude-haiku-5-5` for cheap steps) | Phase 0 |
| **Ollama** | Local chat + embedding models (`nomic-embed-text`) | Phase 0 / 1 |
| **OpenAI** (optional) | Hosted embeddings (`text-embedding-3-small`) for comparison | Phase 1 |

### Flow 1: answering a question (blue path)

1. React sends `POST /api/chat` with the question and conversation history.
2. FastAPI rewrites a follow-up question into a standalone one (cheap model).
3. Routing: documents, data, or both (phase 7), or the agent picks tools (phase 8).
4. **Documents:** embed the query → hybrid search in Postgres → rerank → top chunks.
5. **Data:** LLM writes SQL from the table schema → run read-only in DuckDB.
6. Build the prompt with numbered sources → call the model through LiteLLM.
7. Stream tokens back to React over SSE, then send the citations.
8. Every step is a span in one Langfuse trace.

### Flow 2: ingesting files

1. React uploads to `POST /api/documents` (or you drop files into `data/`).
2. Ingest worker hashes the file and skips it if unchanged.
3. **PDF / Markdown:** Docling parses → chunk by section → embed via LiteLLM → upsert into Postgres.
4. **CSV:** load into a DuckDB table, store schema + a short description.
5. Status is updated so the UI can show progress.

### Flow 3: evaluating

1. `eval.py` loads the golden dataset from Langfuse.
2. Runs every question through the same pipeline code.
3. Computes hit@k and MRR (retrieval) and LLM-judge scores (faithfulness, correctness).
4. Logs it as a Langfuse experiment run, so you can compare runs side by side.

### API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Ask a question; streams the answer (SSE), then citations |
| `POST` | `/api/documents` | Upload a file; returns a document id |
| `GET` | `/api/documents` | List documents and their ingest status |
| `GET` | `/api/documents/{id}` | One document's status and metadata |
| `GET` | `/api/chunks/{id}` | One chunk's text, file and page (for citation clicks) |
| `POST` | `/api/feedback` | 👍/👎 on an answer → Langfuse score |

### Key decisions

- **Postgres does vector *and* keyword search.** No separate vector DB; hybrid search is just SQL.
- **The app only talks to LiteLLM.** Switching between Claude, Ollama and OpenAI, or adding fallbacks and budgets, is config, not code.
- **CSVs go to SQL, not embeddings.** Embeddings can't sum or count.
- **Ingestion is separate from the API.** Slow parsing never blocks a chat request, and you can re-run it on its own.
- **React is a plain SPA.** FastAPI is the only backend, so no Next.js server.
- **Types come from the API.** TypeScript types are generated from FastAPI's OpenAPI schema (`openapi-typescript`), so frontend and backend can't drift.
- **Everything except model providers runs in `docker compose`.**

### Repo layout

```
docpilot/
├── docker-compose.yml        # postgres, langfuse, litellm, api, web
├── litellm_config.yaml
├── pyproject.toml
├── src/docpilot/
│   ├── llm.py                # LiteLLM wrapper
│   ├── ingest/               # parsers, chunkers, embedder
│   ├── retrieval/            # vector, hybrid, rerank
│   ├── data/                 # duckdb + text-to-sql
│   ├── agent/                # tools + loop
│   └── api.py                # FastAPI (+ OpenAPI schema for the web app)
├── web/                      # React + Vite + TS
│   ├── src/api/              # generated types + fetch/stream helpers
│   ├── src/features/chat/    # streaming chat + citations
│   └── src/features/docs/    # upload + document list
├── evals/
│   ├── golden.jsonl
│   └── eval.py
├── data/                     # your corpus (gitignored)
└── RESULTS.md                # experiment table from phases 6–8
```

---

## Phase 0: Foundations (week 1)

**Learn**
- [ ] Messages, system prompt, tokens, context window, temperature, max tokens
- [ ] Streaming responses
- [ ] Cost: input vs output tokens, why prompt size matters
- [ ] Structured output (JSON) and why pipelines need it

**Build**
- [ ] `uv init docpilot`, `.env` with API keys (gitignored)
- [ ] CLI chat loop with `litellm.completion()` and streaming
- [ ] Same question against Claude and an Ollama model, switched by a `--model` flag
- [ ] Print tokens and cost per turn (`litellm.completion_cost`)

**✅ Done when:** you can chat in the terminal, switch models with a flag, and see the cost of every turn.

---

## Phase 1: Embeddings from scratch (week 1–2)

**Learn**
- [x] Embedding = text → vector; similar meaning → nearby vectors
- [x] Cosine similarity vs dot product, normalization
- [x] Why queries and documents must use the **same** embedding model
- [x] Limits: fuzzy, bad at exact IDs, numbers, negation

**Build**
- [x] Embed ~50 sentences with `litellm.embedding()`, store in a NumPy array
- [x] Write `search(query, k)` yourself: cosine similarity + `argsort` (no vector DB)
- [x] Try queries that work ("how do I get a refund") and ones that fail ("invoice #A-1043")
- [x] Compare 2 embedding models on the same 10 queries

**✅ Done when:** you can explain why "invoice #A-1043" fails, and your semantic search is under 40 lines.

---

## Phase 2: Ingestion pipeline (week 2–3)

**Learn**
- [ ] PDF parsing: text layer vs scanned (OCR), multi-column, tables, headers/footers
- [ ] Chunking: fixed + overlap, recursive, by heading/section
- [ ] Metadata: file, page, section, doc type, date
- [ ] Idempotent ingestion: file hashes, skip unchanged, delete stale chunks

**Build**
- [ ] `docker-compose.yml` with Postgres + pgvector
- [ ] Table `chunks(id, doc_id, content, embedding vector(N), page, section, metadata jsonb)` + HNSW index
- [ ] `ingest.py <folder>`: parse → chunk → embed in batches → upsert
- [ ] Parse with PyMuPDF first, then Docling; compare both on a PDF with tables
- [ ] Markdown: chunk by heading
- [ ] Print 20 random chunks and read them

**✅ Done when:** `python ingest.py ./data` is safe to run twice, and a SQL query returns the top 5 chunks for any question.

---

## Phase 3: Naive RAG + React chat (week 3–4)

**Learn**
- [ ] The RAG loop: query → embed → retrieve top-k → prompt → generate
- [ ] Grounded prompting: answer only from context, say "I don't know", cite sources
- [ ] Context ordering, number of chunks, token budget
- [ ] Server-Sent Events (SSE) and reading a stream in the browser

**Build: backend**
- [ ] `answer(question)`: retrieve top-k, format chunks as numbered sources, ask the model to cite `[1]`, `[2]`
- [ ] `POST /api/chat` streaming SSE; `POST/GET /api/documents`; `GET /api/chunks/{id}`
- [ ] Rewrite follow-up questions into standalone queries before retrieval
- [ ] CORS for the Vite dev server (or a Vite proxy to `/api`)

**Build: frontend (`web/`)**
- [ ] `npm create vite@latest web -- --template react-ts`, add Tailwind + shadcn/ui + TanStack Query
- [ ] Generate TS types from FastAPI's OpenAPI schema with `openapi-typescript`
- [ ] **Chat view:** stream with `fetch()` + `response.body.getReader()` (not `EventSource`, which is GET-only). Hand-write it once; later you can try the Vercel AI SDK's `useChat`.
- [ ] **Citations panel:** clicking `[1]` shows the chunk text, file and page
- [ ] **Documents page:** drag-and-drop upload + list with status

**✅ Done when:** you upload a PDF in the browser, ask about it, see the answer stream in, and click through to the source chunk.

---

## Phase 4: Observability with Langfuse (week 4–5)

Instrument before you optimize; otherwise you're tuning blind.

**Learn**
- [ ] Traces, spans, generations: one trace per question
- [ ] Latency and cost per step
- [ ] Prompt management: versioned prompts fetched at runtime
- [ ] User feedback as scores

**Build**
- [ ] Self-host Langfuse with its official Docker Compose
- [ ] Turn on LiteLLM's Langfuse callback (all LLM calls logged)
- [ ] `@observe` on `rewrite()`, `retrieve()`, `answer()`; log chunk IDs and similarity scores
- [ ] Move the system prompt into Langfuse prompt management
- [ ] 👍/👎 buttons in React → `POST /api/feedback` → Langfuse score on the trace

**✅ Done when:** you open a bad answer in Langfuse and see which chunks were retrieved, which prompt version was used, and what it cost.

---

## Phase 5: Evaluation (week 5–6)

The most important phase. Without evals, every later "improvement" is a guess.

**Learn**
- [ ] Retrieval metrics: hit rate@k, MRR
- [ ] Generation metrics: faithfulness, correctness vs a reference answer
- [ ] LLM-as-judge prompts; spot-check the judge against your own ratings
- [ ] Synthetic question generation and its pitfalls

**Build**
- [ ] Golden dataset of 40–60 questions: 20 by hand, the rest LLM-generated and reviewed by you; include questions with no answer in the corpus
- [ ] Upload it as a Langfuse dataset
- [ ] `eval.py`: runs every question as a Langfuse experiment run; computes hit@5, MRR, faithfulness, correctness
- [ ] Rate 15 answers yourself and check that the judge agrees
- [ ] Optional: compare with Ragas or DeepEval
- [ ] Record the baseline scores in `RESULTS.md`

**✅ Done when:** one command prints a scorecard, and you can compare two runs side by side in Langfuse.

---

## Phase 6: Better retrieval (week 6–8)

Add one technique at a time, behind a config flag. Keep it only if the eval improves.

| Done | Technique | What it fixes | hit@5 | faithfulness |
|---|---|---|---|---|
| [ ] | Baseline (from phase 5) | | | |
| [ ] | **Hybrid search**: Postgres full-text + vector, merged with Reciprocal Rank Fusion | Exact terms, IDs, names, numbers | | |
| [ ] | **Reranking**: retrieve 30 → rerank to 5 (`bge-reranker`, Cohere, Voyage) | Best chunks not at the top | | |
| [ ] | **Contextual chunk headers**: prepend doc title + section before embedding | Chunks that lose meaning alone | | |
| [ ] | **Parent–child chunks**: search small, send the parent section | Precision without losing context | | |
| [ ] | **Query rewriting / multi-query / HyDE** | Vague questions | | |
| [ ] | **Metadata filters** (e.g. year, doc type) | Scoped questions | | |
| [ ] | **Chunk size**: 256 vs 512 vs 1024 tokens | Your corpus's sweet spot | | |

**✅ Done when:** this table is filled in with your numbers, and the best combination is the default.

---

## Phase 7: CSVs and structured data (week 8–9)

**Learn**
- [ ] Why embedding CSV rows fails for "total spent on food in March?"
- [ ] Text-to-SQL: schema + sample rows → model writes SQL → you run it
- [ ] Safety: read-only connection, timeouts, row limits
- [ ] Routing between documents, data, or both

**Build**
- [ ] Ingest loads CSVs into DuckDB tables; store schema + description
- [ ] `query_data(question)`: LLM writes SQL → validate → run → LLM summarizes
- [ ] Router with `claude-haiku-5-5` + structured output: `docs` / `data` / `both`
- [ ] Show the SQL and result table in the React answer (collapsible)
- [ ] Add 15 data questions to the eval set, scored on the exact numbers

**✅ Done when:** "how much did I spend per category last quarter?" returns correct numbers, and the SQL appears in the Langfuse trace.

---

## Phase 8: Agentic RAG + MCP (week 9–10)

**Learn**
- [ ] Tool calling: the model picks a tool, your code runs it, the result goes back
- [ ] Agent loop failure modes: infinite loops, wrong tools, cost blow-ups
- [ ] MCP (Model Context Protocol)

**Build**
- [ ] Tools: `search_docs(query, filters)`, `query_data(sql)`, `list_documents()`, `get_table_schema(name)`
- [ ] Agent loop through LiteLLM with a max-step limit
- [ ] Show tool calls as steps in the React chat
- [ ] Compare agent vs phase 7 router on the eval set: quality, latency, cost
- [ ] Bonus: expose the tools as an MCP server and use them from Claude Code or Claude Desktop

**✅ Done when:** the agent answers "compare what the contract says about late fees with how many late payments are in my CSV", and you know from the evals whether the agent beats the router.

---

## Phase 9: LiteLLM proxy + production (week 10–12)

**Learn**
- [ ] LiteLLM proxy: virtual keys, budgets, rate limits, fallbacks, caching
- [ ] Prompt injection hidden inside documents
- [ ] PII in documents and logs
- [ ] Cost and latency budgets

**Build**
- [ ] LiteLLM proxy in compose; the app calls `http://litellm:4000` with a virtual key
- [ ] Fallback model if Claude errors or times out; monthly budget
- [ ] Caching for repeated questions
- [ ] Background ingestion queue with progress in React (poll or SSE)
- [ ] Build React to static files, serve with nginx in compose
- [ ] Add 5 malicious documents + an eval check that the app ignores their instructions
- [ ] One `docker compose up` for Postgres, Langfuse, LiteLLM, API and web

**✅ Done when:** a friend can clone the repo, add keys, run one command and use it, and you can see their usage and cost per key.

---

## Reading list (in order of need)

- [ ] LiteLLM docs: Getting started, Embeddings, Proxy
- [ ] pgvector README: HNSW vs IVFFlat, distance operators
- [ ] Docling docs: converting PDFs, table export
- [ ] Langfuse docs: Tracing, Prompt Management, Datasets & Experiments, LiteLLM integration
- [ ] Hamel Husain, "Your AI Product Needs Evals"
- [ ] Anthropic, "Introducing Contextual Retrieval"
- [ ] Jason Liu, posts on systematically improving RAG
- [ ] Anthropic tool use docs + MCP docs

---

## Rules of thumb

- **Look at your data.** Read chunks, traces and failures.
- **Measure before and after** every change, and write it in `RESULTS.md`.
- **Fix retrieval before blaming the model.** If the right chunk isn't in the context, no model can answer.
- **Don't embed tables for math.** Use SQL.
- **Start simple.** Good parsing + hybrid search + reranking beats most fancy setups.
