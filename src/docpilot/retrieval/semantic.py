"""Fase 1: pesquisa semântica com embeddings (sem base de dados vetorial).

Uso:
    uv run docpilot-search                              # bge-m3 (por defeito)
    uv run docpilot-search --model ollama/nomic-embed-text
"""

import argparse

import litellm
import numpy as np
from dotenv import load_dotenv

litellm.suppress_debug_info = True

SENTENCES_FILE = "experiments/phase1/sentences.txt"


def embed(texts: list[str], model: str) -> np.ndarray:
    """Transforma textos em vetores de comprimento 1, numa só chamada."""
    response = litellm.embedding(model=model, input=texts)
    vectors = np.array([item["embedding"] for item in response.data])
    # Normalizar: com vetores de comprimento 1, produto escalar = cosseno.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def search(query: str, sentences: list[str], vectors: np.ndarray, model: str, k: int = 3):
    """Devolve as k frases mais parecidas com a pergunta, com a pontuação."""
    query_vector = embed([query], model)[0]      # o MESMO modelo das frases
    scores = vectors @ query_vector              # 50 cossenos de uma vez
    top = np.argsort(scores)[::-1][:k]           # índices, do maior para o menor
    return [(float(scores[i]), sentences[i]) for i in top]


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Pesquisa semântica")
    parser.add_argument("--model", default="ollama/bge-m3")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    with open(SENTENCES_FILE, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    vectors = embed(sentences, args.model)       # só uma vez, no arranque
    print(f"{len(sentences)} frases, modelo {args.model}  (escreve 'sair' para terminar)\n")

    while True:
        try:
            query = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"sair", "exit", "quit"}:
            break
        if query:
            for score, sentence in search(query, sentences, vectors, args.model, args.k):
                print(f"  {score:.2f}  {sentence}")
            print()


if __name__ == "__main__":
    main()
