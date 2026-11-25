"""Fase 0: chat no terminal com LiteLLM.

Uso:
    uv run docpilot                                  # Claude Sonnet (por defeito)
    uv run docpilot --model anthropic/claude-haiku-5-5
    uv run docpilot --model ollama/llama3.2          # modelo local (precisa do Ollama)
"""

import argparse
import os

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True  # esconde avisos do LiteLLM (ex.: "Provider List")

DEFAULT_MODEL ="anthropic/claude-sonnet-5-5"
SYSTEM_PROMPT = "És um assistente útil. Responde de forma clara e concisa."


def main() -> None:
    # 1. Lê as chaves de API do ficheiro .env para variáveis de ambiente.
    #    override=True: o .env ganha a qualquer valor que já exista no terminal.
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Chat no terminal")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ex.: anthropic/claude-haiku-5-5")
    args = parser.parse_args()

    if args.model.startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Falta a ANTHROPIC_API_KEY no ficheiro .env")
        return

    # 2. O histórico da conversa. O modelo não tem memória: em cada pedido
    #    enviamos a conversa inteira. Por isso cada turno fica mais caro.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_cost = 0.0

    print(f"Modelo: {args.model}  (escreve 'sair' para terminar)\n")

    while True:
        try:
            question = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D ou Ctrl+C
            print()
            break
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})

        # 3. stream=True: a resposta chega aos bocados (tokens), em vez de
        #    esperarmos pela resposta completa.
        response = litellm.completion(
            model=args.model,
            messages=messages,
            stream=True,
            max_tokens=1024,
            stream_options={"include_usage": True},  # pede a contagem de tokens no fim
        )

        print("IA: ", end="", flush=True)
        chunks = []
        for chunk in response:
            chunks.append(chunk)
            if not chunk.choices:  # o último bocado só traz a contagem de tokens
                continue
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
        print()

        # 4. Junta os bocados numa resposta completa para ler tokens e custo.
        full = litellm.stream_chunk_builder(chunks, messages=messages)
        answer = full.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})

        usage = full.usage
        try:
            cost = litellm.completion_cost(completion_response=full)
        except Exception:
            cost = 0.0  # modelos locais (Ollama) são grátis / sem preço conhecido
        total_cost += cost
        print(
            f"   [tokens: {usage.prompt_tokens} entrada + {usage.completion_tokens} saída"
            f" | custo: ${cost:.5f} | total: ${total_cost:.5f}]\n"
        )
        


if __name__ == "__main__":
    main()
