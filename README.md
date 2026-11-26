# DocPilot

DocPilot is a learning project about building an AI assistant that can answer questions about your documents and cite its sources. Phases 0 and 1 are complete. You can chat in the terminal with Claude or a local Ollama model, watch replies stream in, and see token usage and estimated cost after each turn.

## Try the terminal chat

With [uv](https://docs.astral.sh/uv/) installed and [Ollama](https://ollama.com/) running, run these commands from the project directory:

```bash
ollama pull llama3.2
uv run docpilot --model ollama/llama3.2
```

Try asking `What is an embedding?`, then `Give me an example.` The chat sends the conversation history with each question. Watch the input token count on the second turn. Type `sair` to exit.

To use Claude, set `ANTHROPIC_API_KEY` in a local `.env` file using [.env.example](.env.example) as a template, then run `uv run docpilot`.

## Try semantic search

Phase 1 added a small search engine that finds sentences by meaning, using embeddings and NumPy. Pull the embedding model once, then start the search:

```bash
ollama pull bge-m3
uv run docpilot-search
```

Try `perdi o cartão` or `posso gastar mais do que tenho na conta?`. Each result shows its similarity score. Add `--model ollama/nomic-embed-text` to try the comparison model, or `-k 5` for more results. Type `sair` to exit.

To compare the models on the test queries, run `uv run python experiments/phase1/compare.py`. The findings are in [experiments/phase1/RESULTS.md](experiments/phase1/RESULTS.md).

The next phase builds the ingestion pipeline: parsing real PDFs, splitting them into chunks and storing the vectors in Postgres. See the [project plan](DocPilot%20Plan.md) for the exercises and milestones.
