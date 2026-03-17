# Configuration Reference

Environment variables, provider examples, and custom headers.

All configuration is done through environment variables, typically set in a
`.env` file at the project root. Copy `.env.example` as a starting point:

```bash
cp .env.example .env
```

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | -- | API key for your upstream provider |
| `ANTHROPIC_API_KEY` | No | -- | If set, clients must present this exact key to access the proxy |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Base URL of the upstream OpenAI-compatible API |
| `AZURE_API_VERSION` | No | -- | API version string for Azure OpenAI deployments |
| `BIG_MODEL` | No | `gpt-4o` | Model used for Claude Opus requests |
| `MIDDLE_MODEL` | No | Value of `BIG_MODEL` | Model used for Claude Sonnet requests |
| `SMALL_MODEL` | No | `gpt-4o-mini` | Model used for Claude Haiku requests |
| `HOST` | No | `0.0.0.0` | Bind address for the proxy server |
| `PORT` | No | `8082` | Listen port |
| `LOG_LEVEL` | No | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `MAX_TOKENS_LIMIT` | No | `4096` | Maximum token cap sent to the upstream provider |
| `MIN_TOKENS_LIMIT` | No | `100` | Minimum token floor (avoids errors with thinking models) |
| `REQUEST_TIMEOUT` | No | `90` | Upstream request timeout in seconds |
| `MAX_RETRIES` | No | `2` | Number of retries on upstream failures |
| `SEARCH_PROVIDER` | No | -- | Search provider plugin for web search interception (`tavily`) |
| `TAVILY_API_KEY` | No | -- | API key for [Tavily](https://tavily.com/) web search |

> **Note:** `MAX_TOKENS_LIMIT` is a proxy-side cap on the value sent upstream.
> Modern Claude models support 64k--128k output tokens, but your upstream
> provider may have lower limits.

## Web Search

To enable web search interception, set `SEARCH_PROVIDER` to a supported
provider name and provide its API key:

```bash
SEARCH_PROVIDER="tavily"
TAVILY_API_KEY="tvly-your-api-key"
```

When no `SEARCH_PROVIDER` is set, the proxy silently strips `web_search`
tools from requests so the upstream provider does not receive unsupported
tool definitions.

## Provider Examples

### OpenAI

```bash
OPENAI_API_KEY="sk-your-openai-key"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

### Azure OpenAI

```bash
OPENAI_API_KEY="your-azure-key"
OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
AZURE_API_VERSION="2024-03-01-preview"
BIG_MODEL="gpt-4"
MIDDLE_MODEL="gpt-4"
SMALL_MODEL="gpt-35-turbo"
```

> **Tip:** If you encounter `unsupported_country_region_territory` errors with
> OpenAI direct, Azure OpenAI is a good alternative.

### Local Models (Ollama)

```bash
OPENAI_API_KEY="dummy-key"
OPENAI_BASE_URL="http://localhost:11434/v1"
BIG_MODEL="llama3.1:70b"
MIDDLE_MODEL="llama3.1:70b"
SMALL_MODEL="llama3.1:8b"
```

Ollama does not require a real API key -- any non-empty value works.

## Custom Headers

You can inject arbitrary HTTP headers into every upstream request by setting
environment variables with the `CUSTOM_HEADER_` prefix. Underscores in the
suffix are converted to hyphens.

```bash
# Sends "Accept: application/jsonstream" on every upstream request
CUSTOM_HEADER_ACCEPT=application/jsonstream

# Sends "X-Api-Key: your-api-key"
CUSTOM_HEADER_X_API_KEY=your-api-key

# Sends "X-Client-Id: your-client-id"
CUSTOM_HEADER_X_CLIENT_ID=your-client-id
```

This is useful for providers that require extra authentication or routing
headers beyond the standard `Authorization` bearer token.

## Client API Key Validation

When `ANTHROPIC_API_KEY` is set, every request from Claude Code must include
a matching key in the `x-api-key` header or `Authorization: Bearer` header.
If the key does not match, the proxy returns HTTP 401.

When `ANTHROPIC_API_KEY` is **not** set, client-side validation is disabled
and any request is accepted. This is the default for local development.
