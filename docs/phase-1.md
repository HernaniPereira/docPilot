# Phase 1 — Embeddings from scratch (plan)

**Status:** ✅ done (see [results](../experiments/phase1/RESULTS.md)) · **Time:** weeks 1–2, about 6–8 h/week · **Plan source:** [DocPilot Plan, Phase 1](../DocPilot%20Plan.md#phase-1-embeddings-from-scratch-week-12)

## Goal

Build a tiny semantic search engine by hand: turn sentences into vectors, compare a query vector against all of them, and return the closest matches. No vector database, no framework, just LiteLLM and NumPy.

By the end you should understand *why* it finds "You can request a refund within 30 days" for "how do I get my money back?", and *why* it can fail on "invoice #A-1043". Phase 2 replaces the NumPy array with Postgres + pgvector, and Phase 6 fixes the exact-match weakness with hybrid search. This phase is where you learn what those fixes are for.

## What you'll have at the end

| Deliverable | Path |
| --- | --- |
| About 50 test sentences | `experiments/phase1/sentences.txt` |
| 10 test queries with the expected answer | `experiments/phase1/queries.jsonl` |
| Semantic search (under 40 lines) | `src/docpilot/retrieval/semantic.py` |
| Interactive search command | `uv run docpilot-search` |
| Model comparison script | `experiments/phase1/compare.py` |
| Results and notes | `experiments/phase1/RESULTS.md` |

`experiments/` is committed (unlike `data/`), because the sentences are made-up test data, not your private documents.

## Before you start

- [ ] Ollama running, then pull two local embedding models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull mxbai-embed-large
  ```
- [ ] Optional: `OPENAI_API_KEY` in `.env` to also try `text-embedding-3-small` (hosted, very cheap).
- [ ] Add NumPy: `uv add numpy`

> **Note:** Anthropic does not offer an embeddings API. Claude stays the chat model; embeddings come from Ollama (local) or OpenAI (hosted). LiteLLM hides that difference, so the code only changes the model string.

---

## Step 1 — Learn the concepts (≈2 h)

Read and write a short note in your own words for each:

- [ ] **Embedding:** text → fixed-length vector of floats (e.g. 768 numbers for `nomic-embed-text`). Similar meaning → vectors pointing in similar directions.
- [ ] **Cosine similarity:** `dot(a, b) / (|a| · |b|)`, ranges from -1 to 1 and only cares about direction. **Dot product** also cares about length. If you **normalize** every vector to length 1 first, the two are identical, and dot product is cheaper.
- [ ] **Same model for queries and documents:** each model has its own coordinate system. A vector from model A compared with one from model B is meaningless, even if both have the same dimension.
- [ ] **Limits:** embeddings are fuzzy. They are weak on exact IDs and codes (`A-1043` vs `A-1034`), numbers (`€50` vs `€500`), and negation ("refunds are allowed" vs "refunds are not allowed" land very close).

**Check:** you can explain, without notes, why normalizing makes cosine similarity equal to dot product.

## Step 2 — Write the test corpus (≈1 h)

Create `experiments/phase1/sentences.txt`, one sentence per line, about 50 lines. Mix them on purpose so the experiments are interesting:

| Group | ~Count | Example | Tests |
| --- | --- | --- | --- |
| Paraphrase targets | 15 | "You can request a refund within 30 days of purchase." | Meaning match with different words |
| Unrelated topics | 15 | "The office closes at 6 pm on Fridays." | Distractors |
| Near-duplicate IDs | 8 | "Invoice #A-1043 was paid on 3 March." / "Invoice #A-1034 was paid on 12 May." | Exact ID matching |
| Numbers | 6 | "The monthly fee is €50." / "The annual fee is €500." | Numeric precision |
| Negation pairs | 6 | "Pets are allowed in the apartment." / "Pets are not allowed in the apartment." | Negation |

Tip: use a theme you know (bank, rental contract, a product manual), so you can judge results instantly. It can be the same domain as the corpus you picked for the whole project.

## Step 3 — Embed the sentences (≈1 h)

In `src/docpilot/retrieval/semantic.py`:

- [ ] `embed(texts, model) -> np.ndarray`: call `litellm.embedding(model=model, input=texts)`, take `item["embedding"]` from `response.data`, and return an array of shape `(n, dim)`.
- [ ] Normalize each row to length 1 (`v / np.linalg.norm(v, axis=1, keepdims=True)`).
- [ ] Embed all sentences **in one batch call**, not one call per sentence.
- [ ] Print the shape and one row's first 5 values, and confirm each row's norm is 1.0.

Model strings: `ollama/nomic-embed-text`, `ollama/mxbai-embed-large`, `text-embedding-3-small` (OpenAI).

> **Gotcha:** `nomic-embed-text` was trained with task prefixes. It works better if documents are sent as `"search_document: <text>"` and queries as `"search_query: <text>"`. Try it both ways in Step 6 and note the difference.

## Step 4 — Write `search(query, k)` (≈1 h)

- [ ] Embed the query with the **same model**, normalize it.
- [ ] Scores: `doc_vectors @ query_vector` (dot product = cosine, because both are normalized).
- [ ] Top k: `np.argsort(scores)[::-1][:k]`.
- [ ] Return a list of `(score, sentence)`.
- [ ] Keep the whole file (embed + search) **under 40 lines**.

Then add a small interactive loop and register it in `pyproject.toml`:

```toml
[project.scripts]
docpilot = "docpilot:main"
docpilot-search = "docpilot.retrieval.semantic:main"
```

```text
$ uv run docpilot-search --model ollama/nomic-embed-text
Pergunta: how do I get my money back?
 0.71  You can request a refund within 30 days of purchase.
 0.52  Refunds are paid to the original card within 5 working days.
 0.38  The monthly fee is €50.
```

**Check:** the refund question returns a refund sentence at rank 1.

## Step 5 — Probe what works and what breaks (≈1 h)

Write `experiments/phase1/queries.jsonl`, 10 lines, each with the query and the sentence that *should* be first:

```json
{"query": "how do I get my money back?", "expected": "You can request a refund within 30 days of purchase.", "type": "paraphrase"}
{"query": "invoice #A-1043", "expected": "Invoice #A-1043 was paid on 3 March.", "type": "exact_id"}
```

Aim for: 4 paraphrase, 2 exact ID, 2 number, 2 negation. Run each in `docpilot-search` and write down what you see:

- [ ] Do paraphrases land at rank 1?
- [ ] Does `invoice #A-1043` rank `#A-1034` above or close to the right one? How small is the score gap?
- [ ] Does "Are pets forbidden?" return the "allowed" or the "not allowed" sentence first?

## Step 6 — Compare two embedding models (≈2 h)

`experiments/phase1/compare.py` runs all 10 queries against each model and prints:

| Model | Dim | hit@1 | hit@3 | Avg score of correct | Gap to next wrong | Time to embed 50 |
| --- | --- | --- | --- | --- | --- | --- |
| nomic-embed-text | 768 | | | | | |
| nomic-embed-text + prefixes | 768 | | | | | |
| mxbai-embed-large | 1024 | | | | | |
| text-embedding-3-small (optional) | 1536 | | | | | |

- **hit@k** = fraction of queries where the expected sentence is in the top k. You'll use this exact metric again in Phase 5.
- Also break down hit@1 by query `type`, which is where the differences show.
- [ ] Sanity check: embed the corpus with `nomic-embed-text` (768) and the query with `mxbai-embed-large` (1024). NumPy refuses because the shapes differ. Note that even two models with the *same* dimension would give meaningless scores, because their coordinate systems differ.

Write the table and 3–5 sentences of conclusions in `experiments/phase1/RESULTS.md`.

## Step 7 — Write it up (≈30 min)

- [ ] Fill in `RESULTS.md`: the table, the failing queries, and your explanation of each failure.
- [ ] Update the Progress table in the plan: Phase 1 ⬜ → ✅.
- [ ] Update `README.md` with how to run `docpilot-search`.
- [ ] Commit.

---

## ✅ Done when

- [ ] `semantic.py` (embed + search) is under 40 lines and uses no vector DB.
- [ ] `uv run docpilot-search` returns ranked results with scores.
- [ ] `compare.py` prints hit@1 / hit@3 for at least two models on the same 10 queries.
- [ ] You can explain in your own words why "invoice #A-1043" fails: the ID is a few tokens inside an otherwise identical sentence, so the vectors of `#A-1043` and `#A-1034` are almost the same; embeddings capture meaning, not exact strings.

## Out of scope (comes later)

| Not now | Where |
| --- | --- |
| Storing vectors in a database, indexes (HNSW) | Phase 2 |
| Chunking real PDFs | Phase 2 |
| Generating answers from the results | Phase 3 |
| Keyword / hybrid search to fix exact IDs | Phase 6 |
| Reranking | Phase 6 |

## Questions to answer in RESULTS.md

1. Which model won on paraphrases, and which on exact IDs? Are they the same?
2. Did the `nomic` prefixes change anything?
3. What is a "good" score? Is 0.6 high for one model and low for another? (Hint: scores are not comparable across models, only rankings are.)
4. If you had to find `invoice #A-1043` reliably, what would you add? (This is the motivation for Phase 6.)
