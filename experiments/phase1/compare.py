"""Fase 1, passo 6: compara modelos de embeddings nas mesmas perguntas.

Uso:
    uv run python experiments/phase1/compare.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from docpilot.retrieval.semantic import embed

HERE = Path(__file__).parent

# (nome na tabela, modelo, prefixo das frases, prefixo das perguntas)
MODELS = [
    ("bge-m3", "ollama/bge-m3", "", ""),
    ("nomic", "ollama/nomic-embed-text", "", ""),
    ("nomic + prefixos", "ollama/nomic-embed-text", "search_document: ", "search_query: "),
]


def evaluate(model: str, doc_prefix: str, query_prefix: str, sentences: list[str], queries: list[dict]) -> dict:
    start = time.perf_counter()
    docs = embed([doc_prefix + s for s in sentences], model)
    embed_seconds = time.perf_counter() - start
    qs = embed([query_prefix + q["query"] for q in queries], model)
    scores = qs @ docs.T                          # (perguntas, frases)

    rows = []
    for q, row in zip(queries, scores):
        ranking = list(np.argsort(row)[::-1])
        top = sentences[ranking[0]]
        if q["expected"] is None:                 # pergunta sem resposta
            rows.append({**q, "rank": None, "top": top, "top_score": float(row[ranking[0]])})
            continue
        i = sentences.index(q["expected"])
        best_wrong = max(float(row[j]) for j in range(len(sentences)) if j != i)
        rows.append({**q, "rank": ranking.index(i) + 1, "top": top,
                     "score": float(row[i]), "gap": float(row[i]) - best_wrong,
                     "top_score": float(row[ranking[0]])})
    return {"dim": docs.shape[1], "seconds": embed_seconds, "rows": rows}


def main() -> None:
    load_dotenv(override=True)
    sentences = [l.strip() for l in (HERE / "sentences.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    results = {name: evaluate(m, dp, qp, sentences, queries) for name, m, dp, qp in MODELS}

    print("\n## Resumo (10 perguntas com resposta)\n")
    print("| Modelo | Dim | hit@1 | hit@3 | Score médio correto | Margem média | Tempo 50 frases |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        a = [x for x in r["rows"] if x["rank"] is not None]
        print(f"| {name} | {r['dim']} | {sum(x['rank'] == 1 for x in a)}/{len(a)} "
              f"| {sum(x['rank'] <= 3 for x in a)}/{len(a)} | {np.mean([x['score'] for x in a]):.2f} "
              f"| {np.mean([x['gap'] for x in a]):+.2f} | {r['seconds']:.2f}s |")

    print("\n## hit@1 por tipo\n")
    types = list(dict.fromkeys(q["type"] for q in queries if q["expected"] is not None))
    print("| Modelo | " + " | ".join(types) + " |")
    print("| --- |" + " --- |" * len(types))
    for name, r in results.items():
        by_type = defaultdict(list)
        for x in r["rows"]:
            if x["rank"] is not None:
                by_type[x["type"]].append(x["rank"] == 1)
        print(f"| {name} | " + " | ".join(f"{sum(by_type[t])}/{len(by_type[t])}" for t in types) + " |")

    print("\n## Detalhe por pergunta (posição da frase certa · margem para a melhor errada)\n")
    print("| Pergunta | Tipo | " + " | ".join(results) + " |")
    print("| --- | --- |" + " --- |" * len(results))
    for i, q in enumerate(queries):
        cells = []
        for r in results.values():
            x = r["rows"][i]
            cells.append(f"top {x['top_score']:.2f}" if x["rank"] is None
                         else f"{'✅' if x['rank'] == 1 else '❌'} #{x['rank']} {x['gap']:+.2f}")
        print(f"| {q['query']} | {q['type']} | " + " | ".join(cells) + " |")

    print("\n## Erros (o que ficou em 1.º lugar)\n")
    for name, r in results.items():
        for x in r["rows"]:
            if x["rank"] is not None and x["rank"] != 1:
                print(f"- **{name}** · {x['query']}\n  → {x['top']}")


if __name__ == "__main__":
    main()
