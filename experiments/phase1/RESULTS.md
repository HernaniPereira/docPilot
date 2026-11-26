# Phase 1 results — embeddings from scratch

Semantic search over 50 Portuguese banking sentences ([sentences.txt](sentences.txt)) with 10 test queries plus one query that has no answer ([queries.jsonl](queries.jsonl)). The search code is in [semantic.py](../../src/docpilot/retrieval/semantic.py): embed, normalize, score with a dot product, rank with `argsort`. No vector database.

Reproduce with:

```bash
uv run python experiments/phase1/compare.py
```

## Models compared

| Model | Parameters | Dim | Max input | Languages |
| --- | --- | --- | --- | --- |
| `ollama/bge-m3` | 568M | 1024 | 8192 tokens | Multilingual |
| `ollama/nomic-embed-text` (v1.5) | 137M | 768 | 8192 tokens | Mostly English |
| `nomic-embed-text` + `search_document:` / `search_query:` prefixes | same | 768 | same | same |

## Summary

| Model | hit@1 | hit@3 | Avg score of correct | Avg margin |
| --- | --- | --- | --- | --- |
| **bge-m3** | **10/10** | **10/10** | 0.72 | **+0.09** |
| nomic | 6/10 | 7/10 | 0.71 | −0.00 |
| nomic + prefixes | 6/10 | 7/10 | 0.67 | −0.01 |

- **hit@k:** share of queries whose expected sentence is in the top k.
- **Margin:** score of the expected sentence minus the best wrong sentence. Positive means the right sentence won.

Timing was left out: the first `nomic` run included loading the model into memory, so the times weren't comparable.

## hit@1 by query type

| Model | paraphrase | exact_id | number | negation |
| --- | --- | --- | --- | --- |
| bge-m3 | 4/4 | 2/2 | 2/2 | 2/2 |
| nomic | 2/4 | 2/2 | 2/2 | 0/2 |
| nomic + prefixes | 2/4 | 2/2 | 2/2 | 0/2 |

## Per query (rank of the expected sentence · margin)

| Query | Type | bge-m3 | nomic | nomic + prefixes |
| --- | --- | --- | --- | --- |
| posso gastar mais do que tenho na conta? | paraphrase | ✅ #1 +0.14 | ✅ #1 +0.07 | ✅ #1 +0.06 |
| como sei logo quando alguém me tira dinheiro? | paraphrase | ✅ #1 +0.06 | ✅ #1 +0.01 | ✅ #1 +0.03 |
| mudei de casa, onde ponho o novo endereço? | paraphrase | ✅ #1 +0.10 | ❌ #15 −0.08 | ❌ #22 −0.12 |
| preciso de um documento que mostre que sou o dono da conta | paraphrase | ✅ #1 +0.19 | ❌ #13 −0.08 | ❌ #10 −0.04 |
| estado da autorização DD-5912 | exact_id | ✅ #1 +0.07 | ✅ #1 +0.02 | ✅ #1 +0.03 |
| o pedido CR-7268 já foi aceite? | exact_id | ✅ #1 +0.05 | ✅ #1 +0.00 | ✅ #1 +0.01 |
| quanto custa a manutenção da conta? pago 5 euros por mês? | number | ✅ #1 +0.04 | ✅ #1 +0.08 | ✅ #1 +0.01 |
| posso tirar 2000 euros por dia numa caixa? | number | ✅ #1 +0.03 | ✅ #1 +0.03 | ✅ #1 +0.02 |
| o quiosque recusa notas e moedas? | negation | ✅ #1 +0.12 | ❌ #9 −0.05 | ❌ #7 −0.04 |
| as contas jovens bloqueiam transferências para o estrangeiro? | negation | ✅ #1 +0.07 | ❌ #2 −0.01 | ❌ #2 −0.00 |
| como recebo o meu dinheiro de volta? | no_answer | top 0.63 | top 0.64 | top 0.63 |

## Failures explained

| Model | Query | Ranked first instead | Why |
| --- | --- | --- | --- |
| nomic | mudei de casa, onde ponho o novo endereço? | "Podes **mudar** o nome dos objetivos de poupança…" | Matched a shared word ("mudar") instead of the meaning. An English-trained model doesn't link "endereço" to "morada" |
| nomic | preciso de um documento que mostre que sou o dono da conta | "Os extratos mensais estão disponíveis para descarregar em PDF." | Picked the general "account document" topic, missing "titularidade" = ownership |
| nomic | o quiosque recusa notas e moedas? | "Ativa as notificações push… sair dinheiro…" | Didn't connect "notas e moedas" with "numerário" or "recusa" with "não são aceites" |
| nomic | as contas jovens bloqueiam transferências para o estrangeiro? | "As transferências internacionais **estão** disponíveis nas contas jovens." | Negation: the opposite sentence wins because it shares almost all the words |

## Why "invoice #A-1043" is hard

Both exact-ID queries passed, but only just. `bge-m3` beat the twin sentence by +0.05 and +0.07; `nomic` by +0.00 and +0.02. In Step 4, `fatura TR-1043` scored 0.65 for the right sentence and 0.57 for TR-1034.

The twin sentences are identical except for a code that is a few tokens long. An embedding compresses the whole sentence's *meaning* into one vector, and `TR-1043` and `TR-1034` mean the same thing to the model: "some transfer reference". So the two vectors are almost the same, and which one wins depends on noise. Embeddings capture meaning, not exact strings. Exact matching needs keyword search (Phase 6, hybrid search).

The same applies to numbers (€5 vs €50, margins +0.01 to +0.08) and negation ("são aceites" vs "não são aceites": 0.73 vs 0.70 in Step 4).

## The no-answer query

`como recebo o meu dinheiro de volta?` has no answer in the file, yet every model returned something with a score of about 0.63, close to the scores of correct answers (0.65 for `fatura TR-1043`). Search always returns the top k, and a score threshold can't separate "nothing relevant" from "right answer" here. The fix belongs later: the chat model must say "I don't know" when the sources don't answer the question (Phase 3), a reranker gives more reliable scores (Phase 6), and no-answer questions belong in the eval set (Phase 5).

## Answers to the plan's questions

1. **Which model won on paraphrases, and which on exact IDs?** `bge-m3` won both. On exact IDs, both models ranked the right sentence first, but `bge-m3` had larger margins.
2. **Did the `nomic` prefixes change anything?** No. hit@1 stayed at 6/10 and the margins moved by ±0.01–0.04 either way. The language mismatch matters far more than the prefixes.
3. **What is a "good" score?** There isn't a fixed one. Correct answers scored 0.65–0.80 and the no-answer query scored 0.63. Scores aren't comparable across models; only rankings and margins are.
4. **How would you find `invoice #A-1043` reliably?** Add keyword search (Postgres full-text) next to vector search and combine them (hybrid search, Phase 6), or filter on extracted IDs as metadata.

## Decision

Use **`bge-m3`** as DocPilot's embedding model from Phase 2 on. It handles Portuguese, reads up to 8192 tokens per chunk, and won every query type. Keep `nomic-embed-text` as a baseline for later comparisons.

Caveat: 10 queries is a small test. Phase 5 builds a proper golden set to confirm this.
