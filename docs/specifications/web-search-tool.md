---
title: Web Search Tool Specification
description: Anthropic server_tool_use / web_search_tool_result content block specification
sidebar:
  label: Web Search Spec
  order: 1
---

> Reverse-engineered from the Anthropic API documentation,
> Python SDK (`anthropic-sdk-python`) type definitions,
> and TypeScript SDK (`anthropic-sdk-node`) type definitions.
> Last updated: 2026-03-16.

## 1. Overview

Web Search is a **server-side tool** — Anthropic executes
searches internally in a sampling loop (up to 10 iterations)
with no client round-trip needed. The `stop_reason` is
typically `"end_turn"` (not `"tool_use"`). If the internal
loop limit is hit, `stop_reason` is `"pause_turn"`.

Two versions exist:

| Version | Identifier | Notes |
| --------- | ----------- | ------- |
| v1 | `web_search_20250305` | Basic web search |
| v2 | `web_search_20260209` | Adds dynamic filtering and `allowed_callers` |

## 2. Tool Definition (Request)

The tool definition sent in the `tools` array:

```json
{
  "type": "web_search_20250305",
  "name": "web_search",
  "max_uses": 5,
  "allowed_domains": ["example.com"],
  "blocked_domains": ["spam.com"],
  "user_location": {
    "type": "approximate",
    "city": "San Francisco",
    "region": "California",
    "country": "US",
    "timezone": "America/Los_Angeles"
  }
}
```

### Tool fields

| Field | Type | Required | Description |
| ------- | ------ | ---------- | ------------- |
| `type` | `"web_search_20250305"` \| `"web_search_20260209"` | Yes | Version identifier |
| `name` | `"web_search"` (literal) | Yes | Must be exactly `"web_search"` |
| `max_uses` | `int` \| `null` | No | Max search invocations per request |
| `allowed_domains` | `string[]` | No | Restrict searches to these domains |
| `blocked_domains` | `string[]` | No | Exclude these domains from results |
| `user_location` | `UserLocation` \| `null` | No | Approximate user location for geo-relevance |
| `cache_control` | `CacheControlEphemeral` \| `null` | No | Prompt caching control |
| `allowed_callers` | `("direct" \| "code_execution_20250825" \| "code_execution_20260120")[]` | No | v2 only — which callers can invoke |

### UserLocation

| Field | Type | Required | Description |
| ------- | ------ | ---------- | ------------- |
| `type` | `"approximate"` (literal) | Yes | Always `"approximate"` |
| `city` | `string` \| `null` | No | City name |
| `region` | `string` \| `null` | No | Region/state name |
| `country` | `string` \| `null` | No | ISO 3166-1 alpha-2 country code |
| `timezone` | `string` \| `null` | No | IANA timezone identifier |

## 3. Content Block Types (Response)

### 3a. `server_tool_use` — Claude's search invocation

```json
{
  "type": "server_tool_use",
  "id": "srvtoolu_01A2B3C4D5E6F7G8H9I0J1K2",
  "name": "web_search",
  "input": {
    "query": "current weather in San Francisco"
  }
}
```

Key differences from `tool_use`:

- `type` is `"server_tool_use"` (not `"tool_use"`)
- `id` prefix is `"srvtoolu_"` (not `"toolu_"`)
- The client does NOT execute this — the server already did
- `name` is one of: `"web_search"`, `"web_fetch"`,
  `"code_execution"`, `"bash_code_execution"`,
  `"text_editor_code_execution"`,
  `"tool_search_tool_regex"`, `"tool_search_tool_bm25"`

| Field | Type | Description |
| ------- | ------ | ------------- |
| `type` | `"server_tool_use"` | Discriminator |
| `id` | `string` | Unique ID, prefix `srvtoolu_` |
| `name` | `string` | Server tool name (e.g., `"web_search"`) |
| `input` | `object` | Tool input (for web_search: `{"query": "..."}`) |
| `caller` | `Caller` \| `null` | Optional — which tool invoked this (for nested calls) |

#### Caller union (discriminated on `type`)

- `{"type": "direct"}` — invoked directly by Claude
- `{"type": "code_execution_20250825", "tool_id": "..."}` — invoked from code execution
- `{"type": "code_execution_20260120", "tool_id": "..."}` — invoked from v2 code execution

### 3b. `web_search_tool_result` — Search results

```json
{
  "type": "web_search_tool_result",
  "tool_use_id": "srvtoolu_01A2B3C4D5E6F7G8H9I0J1K2",
  "content": [
    {
      "type": "web_search_result",
      "url": "https://example.com/page",
      "title": "Example Page Title",
      "encrypted_content": "<encrypted string>",
      "page_age": "2 days ago"
    }
  ]
}
```

| Field | Type | Description |
| ------- | ------ | ------------- |
| `type` | `"web_search_tool_result"` | Discriminator |
| `tool_use_id` | `string` | References the `server_tool_use` block's `id` |
| `content` | `WebSearchResult[]` \| `WebSearchToolResultError` | Results array or error object |
| `caller` | `Caller` \| `null` | Optional caller context |

### 3c. `web_search_result` — Individual result

```json
{
  "type": "web_search_result",
  "url": "https://example.com/page",
  "title": "Example Page Title",
  "encrypted_content": "<encrypted string>",
  "page_age": "2 days ago"
}
```

| Field | Type | Description |
| ------- | ------ | ------------- |
| `type` | `"web_search_result"` | Discriminator |
| `url` | `string` | Source URL |
| `title` | `string` | Page title |
| `encrypted_content` | `string` | Encrypted page content (opaque to client) |
| `page_age` | `string` \| `null` | Human-readable age (e.g., "2 days ago") |

`encrypted_content` is an opaque encrypted blob that
Anthropic's servers use to enable citations. It is not
intended to be decoded or interpreted by clients.

### 3D. `web_search_tool_result_error` — Error case

```json
{
  "type": "web_search_tool_result_error",
  "error_code": "unavailable"
}
```

| Field | Type | Description |
| ------- | ------ | ------------- |
| `type` | `"web_search_tool_result_error"` | Discriminator |
| `error_code` | `WebSearchToolResultErrorCode` | Error classification |

#### Error codes

| Code | Description |
| ------ | ------------- |
| `invalid_tool_input` | Malformed tool input |
| `unavailable` | Search service temporarily unavailable |
| `max_uses_exceeded` | Exceeded `max_uses` limit |
| `too_many_requests` | Rate limited |
| `query_too_long` | Search query exceeds length limit |
| `request_too_large` | Overall request too large |

## 4. Citations in Text Blocks

When Claude cites search results, the text block includes
a `citations` array:

```json
{
  "type": "text",
  "text": "According to recent reports, the weather is sunny.",
  "citations": [
    {
      "type": "web_search_result_location",
      "url": "https://example.com/weather",
      "title": "Weather Report",
      "encrypted_index": "<encrypted string>",
      "cited_text": "The weather in San Francisco is sunny today with..."
    }
  ]
}
```

### Citation fields (`web_search_result_location`)

| Field | Type | Description |
| ------- | ------ | ------------- |
| `type` | `"web_search_result_location"` | Discriminator |
| `url` | `string` | URL of the cited source |
| `title` | `string` | Title of the cited page |
| `encrypted_index` | `string` | Encrypted reference to the result |
| `cited_text` | `string` | Exact text being cited (up to ~150 chars) |

Citations are generated by Anthropic's servers using the
`encrypted_content` from search results. The `encrypted_index`
field is an opaque reference back to the source result.

## 5. Complete Non-Streaming Response Example

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-20250514",
  "content": [
    {
      "type": "server_tool_use",
      "id": "srvtoolu_01A2B3C4D5E6F7G8H9I0J1K2",
      "name": "web_search",
      "input": {
        "query": "current weather San Francisco March 2026"
      }
    },
    {
      "type": "web_search_tool_result",
      "tool_use_id": "srvtoolu_01A2B3C4D5E6F7G8H9I0J1K2",
      "content": [
        {
          "type": "web_search_result",
          "url": "https://weather.example.com/sf",
          "title": "San Francisco Weather - Current Conditions",
          "encrypted_content": "<encrypted>",
          "page_age": "1 hour ago"
        },
        {
          "type": "web_search_result",
          "url": "https://news.example.com/weather-sf",
          "title": "SF Weather Update",
          "encrypted_content": "<encrypted>",
          "page_age": "3 hours ago"
        }
      ]
    },
    {
      "type": "text",
      "text": "Based on current reports, San Francisco is experiencing sunny weather today."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1024,
    "output_tokens": 256,
    "server_tool_use": {
      "web_search_requests": 1
    }
  }
}
```

## 6. Streaming Event Sequence

The SSE event sequence for a web search interaction:

```
1.  event: message_start
    data: {"type":"message_start","message":{
      "id":"msg_...","type":"message","role":"assistant",
      "model":"...","content":[],"stop_reason":null,
      "stop_sequence":null,"usage":{"input_tokens":N,"output_tokens":0}}}

2.  event: content_block_start
    data: {"type":"content_block_start","index":0,
      "content_block":{"type":"server_tool_use",
      "id":"srvtoolu_...","name":"web_search","input":{}}}

3.  event: content_block_delta
    data: {"type":"content_block_delta","index":0,
      "delta":{"type":"input_json_delta",
      "partial_json":"{\"query\": \"search terms\"}"}}

4.  event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    [Server executes search — pause here while results are fetched]

5.  event: content_block_start
    data: {"type":"content_block_start","index":1,
      "content_block":{"type":"web_search_tool_result",
      "tool_use_id":"srvtoolu_...","content":[
        {"type":"web_search_result","url":"...","title":"...",
         "encrypted_content":"...","page_age":"..."}]}}

6.  event: content_block_stop
    data: {"type":"content_block_stop","index":1}

7.  event: content_block_start
    data: {"type":"content_block_start","index":2,
      "content_block":{"type":"text","text":""}}

8.  event: content_block_delta
    data: {"type":"content_block_delta","index":2,
      "delta":{"type":"text_delta","text":"Based on..."}}
    [... more text deltas ...]

9.  event: content_block_stop
    data: {"type":"content_block_stop","index":2}

10. event: message_delta
    data: {"type":"message_delta",
      "delta":{"stop_reason":"end_turn","stop_sequence":null},
      "usage":{"output_tokens":N,
        "server_tool_use":{"web_search_requests":1}}}

11. event: message_stop
    data: {"type":"message_stop"}
```

Key observations:

- `server_tool_use` block starts with `input: {}` and the
  query arrives via `input_json_delta`
- `web_search_tool_result` arrives fully formed in
  `content_block_start` (no deltas for results)
- There is a pause between steps 4 and 5 while the server
  fetches results
- Text blocks with the synthesized answer come after the
  search results
- The `server_tool_use` / `web_search_tool_result` pair
  can repeat if Claude searches multiple times

## 7. Multi-Turn Conversation Handling

When search results appear in conversation history
(subsequent turns), the client must pass back the
`server_tool_use` and `web_search_tool_result` blocks
verbatim in the `messages` array:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in SF?"},
    {
      "role": "assistant",
      "content": [
        {
          "type": "server_tool_use",
          "id": "srvtoolu_...",
          "name": "web_search",
          "input": {"query": "..."}
        },
        {
          "type": "web_search_tool_result",
          "tool_use_id": "srvtoolu_...",
          "content": [...]
        },
        {
          "type": "text",
          "text": "The weather is..."
        }
      ]
    },
    {"role": "user", "content": "What about tomorrow?"}
  ]
}
```

The `encrypted_content` and `encrypted_index` fields must
be passed back verbatim for citations to work in subsequent
turns.

## 8. Usage Tracking

The `usage` object in the response includes:

```json
{
  "input_tokens": 1024,
  "output_tokens": 256,
  "server_tool_use": {
    "web_search_requests": 1
  }
}
```

- `server_tool_use.web_search_requests` counts the number
  of search invocations
- Pricing: $10 per 1,000 searches (in addition to standard
  token costs)
- `encrypted_content` fields do NOT count toward token usage

## 9. Server Tools vs Client Tools

| Aspect | Client Tool (`tool_use`) | Server Tool (`server_tool_use`) |
| -------- | -------------------------- | -------------------------------- |
| Block type | `"tool_use"` | `"server_tool_use"` |
| ID prefix | `"toolu_"` | `"srvtoolu_"` |
| Who executes | Client | Anthropic's server |
| Result block | `"tool_result"` (in next user message) | `"web_search_tool_result"` (in same assistant message) |
| `stop_reason` | `"tool_use"` | `"end_turn"` (or `"pause_turn"`) |
| Client action | Execute tool, send result | None — results are inline |
| Tool definition | `name` + `input_schema` | `type` (versioned) + config |
