# Getting Started

Install, configure, and run Claude Code Proxy in minutes.

## Prerequisites

- **Python 3.13+** (see `pyproject.toml` for the exact constraint)
- **uv** package manager (recommended) or pip
- An API key for your chosen provider (OpenAI, Azure, Ollama, etc.)
- **Claude Code CLI** installed (`npm install -g @anthropic-ai/claude-code`)

## Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Configure Your Provider

Copy the example environment file and set your API key and model names:

```bash
cp .env.example .env
```

Edit `.env` with your OpenAI credentials:

```bash
OPENAI_API_KEY="sk-your-openai-key"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

For Azure OpenAI, Ollama, or other providers, see the
[Configuration Reference](./configuration.md).

## Start the Proxy

```bash
python start_proxy.py
```

The proxy starts on `http://localhost:8082` by default.

## Connect Claude Code

In a separate terminal, point Claude Code at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it -- Claude Code now sends requests through the proxy to your
chosen provider.

## Verify the Setup

Check that the proxy is running:

```bash
curl http://localhost:8082/health
```

Run the built-in test script to confirm end-to-end connectivity:

```bash
python src/test_claude_to_openai.py
```
