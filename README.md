# Gemini Live for Home Assistant

A Home Assistant custom integration for real-time voice conversations with
**Google's Gemini Live API**, including Home Assistant device control,
Google Search and MCP (Model Context Protocol) server support.

## Features

- Real-time, bidirectional voice conversation (16 kHz in / 24 kHz out) with barge-in
- Home Assistant control via built-in tools (`get_entity_state`, `call_service`,
  `get_entities_by_domain`, `get_area_entities`)
- MCP servers (SSE/HTTP and stdio), exposed to Gemini as function calls
- Optional Google Search grounding
- Input and output transcription
- Long sessions: context window compression plus transparent reconnect via
  session resumption when the server ends a connection (~10 min)
- Affective dialog and proactive audio (not on 3.1 Flash Live)
- Lovelace card (`gemini-live-card`) with microphone streaming and playback
- Conversation agent, media player and binary sensors (connected / listening / speaking)

## Supported models

| Model | Notes |
|---|---|
| `gemini-3.8-live` | Default. Low-latency voice agent model. No thinking level. |
| `gemini-3.8-live-extended-thinking` | Background reasoning; thinking level `low`/`medium`/`high`; functions run non-blocking. |
| `gemini-3.1-flash-live-preview` | Legacy preview; thinking level `minimal`–`high`; no affective dialog / proactive audio. |
| `gemini-2.5-flash-native-audio-preview-12-2025` | Legacy; only for projects that already used 2.5 models. |

## Requirements

- Home Assistant 2024.1 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Python package `google-genai>=2.0.0` (installed automatically)

## Installation

### HACS

1. HACS → ⋮ → **Custom repositories**
2. Add `https://github.com/BuehlerSimon/HA_Voice_Test_1` with category **Integration**
3. Install **Gemini Live for Home Assistant** and restart Home Assistant

### Manual

Copy `custom_components/gemini_live` to `/config/custom_components/` and restart Home Assistant.

## Configuration

1. **Settings → Devices & Services → Add Integration → Gemini Live**
2. Enter your API key and choose model, voice (30 prebuilt voices), instructions,
   temperature, thinking level and optional features
3. Optionally add MCP servers in the next step

All settings can be changed later via **Configure** on the integration.

## Lovelace card

The card is registered automatically:

```yaml
type: custom:gemini-live-card
title: Gemini Live
show_transcript: true
```

Microphone access requires Home Assistant to be served over HTTPS (or localhost).

## MCP servers

Gemini Live cannot call MCP servers itself, so the integration connects to them
from Home Assistant and exposes each tool as a function named `server__tool`.

- **SSE / HTTP**: URL of a JSON-RPC MCP endpoint, optional bearer token
  (e.g. Home Assistant's own MCP Server integration)
- **stdio**: command + arguments, started by Home Assistant. On Home Assistant OS
  only commands available in the container work; prefer SSE servers there.

## Services

| Service | Description |
|---|---|
| `gemini_live.send_message` | Send a text message |
| `gemini_live.send_audio` | Send base64 PCM audio |
| `gemini_live.start_listening` / `stop_listening` | Start / stop a voice session |
| `gemini_live.add_mcp_server` / `remove_mcp_server` | Manage MCP servers |
| `gemini_live.list_mcp_servers` / `connect_mcp_servers` | Inspect / connect MCP servers |
| `gemini_live.clear_conversation` | Reset conversation state |

## Limits (Gemini Live API)

- Audio-only sessions: 15 min without compression (compression is enabled by default)
- Audio + video sessions: 2 min
- Resumption handles are valid for 2 hours

## Troubleshooting

Enable debug logging:

```yaml
logger:
  logs:
    custom_components.gemini_live: debug
```

Useful log lines: `Connected to Gemini Live API (model=...)`, `Received GoAway`,
`Resumed Gemini Live API session`.

## Credits

Based on [SJang1/ha-ai-audio](https://github.com/SJang1/ha-ai-audio).
