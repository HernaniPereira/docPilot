# Phase 0 — Terminal chat foundations

Phase 0 delivered a terminal chat for comparing a hosted Claude model with a local Llama model. Both use the same Python chat loop. The `--model` option selects which model receives the conversation.

## Running the chat

Run these commands from the project root with uv installed. The project requires Python 3.13 or newer.

| Model | Setup | Start the chat |
| --- | --- | --- |
| Claude (default) | Set `ANTHROPIC_API_KEY` in your local `.env`. The repository includes an [.env.example](../.env.example) template and Git ignores `.env`. | `uv run docpilot` |
| Llama 3.2 | Start Ollama and download the model once with `ollama pull llama3.2`. | `uv run docpilot --model ollama/llama3.2` |

Type a question at the `Tu:` prompt. The reply appears after `IA:`. Type `sair` to end the session.

## How one turn works

1. At startup, the program reads the model option with Python’s `argparse` module and starts a message list with a system prompt asking for clear, concise answers.
2. When you enter a question, the program adds it to the list and sends the whole conversation to the selected model through `litellm.completion()`. This lets a follow-up such as “Now in Portuguese?” refer to the previous reply.
3. The terminal prints each piece of the reply as it arrives. The program then joins the pieces and adds the complete reply to the conversation. Starting a new session starts a fresh history.
4. After each reply, the program prints input tokens (`entrada`), output tokens (`saída`), estimated cost for that turn (`custo`), and cumulative session cost (`total`). If LiteLLM cannot calculate a price, the program displays zero for that turn.

The [chat implementation](../src/docpilot/chat.py) contains this loop. The [project configuration](../pyproject.toml) defines the dependencies and registers the `docpilot` command.

## What was verified

Both Claude and local Llama answered questions in the terminal and displayed token counts. In the two-turn Claude check, the first request used 40 input tokens. The follow-up used 63 because the request also included the earlier exchange. The displayed session cost after both turns was $0.00114. The local Llama run displayed $0.00000.

These runs met the [Phase 0 milestone](../DocPilot%20Plan.md#phase-0-foundations-week-1) of chatting in the terminal, switching models with a flag, and seeing the cost of each turn.
