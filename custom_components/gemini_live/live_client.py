"""Gemini Live API client using google-genai library.

This module provides a client for Google's Gemini Live API,
supporting real-time audio, video (camera/screen), function calling,
Google Search and transparent reconnects via session resumption.

Based on official documentation:
- https://ai.google.dev/gemini-api/docs/live-guide
- https://ai.google.dev/gemini-api/docs/live-tools
- https://ai.google.dev/gemini-api/docs/live-session
"""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from google import genai
from google.genai import types

from .const import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_VOICE,
    EVENT_AUDIO_DELTA,
    EVENT_ERROR,
    EVENT_FUNCTION_CALL,
    EVENT_GENERATION_COMPLETE,
    EVENT_GO_AWAY,
    EVENT_INPUT_TRANSCRIPTION,
    EVENT_INTERRUPTED,
    EVENT_OUTPUT_TRANSCRIPTION,
    EVENT_RECONNECTING,
    EVENT_SESSION_RESUMED,
    EVENT_SESSION_RESUMPTION_UPDATE,
    EVENT_SESSION_STARTED,
    EVENT_TEXT_DELTA,
    EVENT_TOOL_CALL_CANCELLATION,
    EVENT_TURN_COMPLETE,
    MODEL_THINKING_LEVELS,
    MODELS_NON_BLOCKING_ONLY,
    MODELS_WITHOUT_NATIVE_AUDIO_FEATURES,
)

_LOGGER = logging.getLogger(__name__)

# Set google_genai loggers to DEBUG to avoid warning spam about non-data parts
logging.getLogger("google_genai.types").setLevel(logging.DEBUG)
logging.getLogger("google_genai").setLevel(logging.DEBUG)

# Live API features (affective dialog, proactive audio) are served on v1beta
API_VERSION = "v1beta"
# google-genai still creates/uses ephemeral tokens via v1alpha only
EPHEMERAL_TOKEN_API_VERSION = "v1alpha"

# Audio constants
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000

# Reconnect settings (a single Live connection lasts ~10 minutes)
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_BACKOFF_SECONDS = 1.0

# JSON schema keys Gemini does not accept in function declarations
_UNSUPPORTED_SCHEMA_KEYS = ("$schema", "$id", "$comment")


@dataclass
class SessionConfig:
    """Configuration for a Gemini Live session."""

    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    instructions: str = DEFAULT_INSTRUCTIONS
    temperature: float = DEFAULT_TEMPERATURE
    tools: list[dict[str, Any]] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    enable_google_search: bool = False
    enable_affective_dialog: bool = False
    enable_proactive_audio: bool = False
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM"
    # Only sent to models that support thinking_level (see MODEL_THINKING_LEVELS)
    thinking_level: str = DEFAULT_THINKING_LEVEL
    context_window_compression: bool = True
    # None = server defaults for the sliding window
    trigger_tokens: int | None = None
    target_tokens: int | None = None
    input_audio_transcription: bool = True
    output_audio_transcription: bool = True
    # Session resumption - handle from previous session to resume
    session_resumption_handle: str | None = None
    # Enable session resumption for long-running sessions
    enable_session_resumption: bool = True


@dataclass
class ConversationItem:
    """Represents a conversation item."""

    id: str
    role: str
    parts: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"


@dataclass
class LiveResponse:
    """Represents a response from the API."""

    id: str
    status: str
    output: list[ConversationItem] = field(default_factory=list)
    text: str = ""
    audio_transcript: str = ""
    audio_data: bytes = b""


def _clean_json_schema(schema: Any) -> Any:
    """Recursively strip JSON schema keys that Gemini rejects."""
    if isinstance(schema, dict):
        return {
            key: _clean_json_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_clean_json_schema(item) for item in schema]
    return schema


class GeminiLiveClient:
    """Client for Gemini Live API using google-genai library."""

    def __init__(
        self,
        api_key: str,
        session_config: SessionConfig | None = None,
    ) -> None:
        """Initialize the client."""
        self._api_key = api_key
        self._session_config = session_config or SessionConfig()
        self._client: genai.Client | None = None
        self._session = None
        self._session_context = None  # Keep context manager reference for session
        self._connected = False
        # True while disconnect() tears the session down on purpose
        self._closing = False
        # True while the session is swapped for a resumed one
        self._reconnecting = False
        # Set by GoAway: reconnect at the next turn boundary
        self._reconnect_requested = False

        # Event handlers
        self._event_handlers: dict[str, list[Callable]] = {}

        # Queue for received audio
        self._audio_in_queue: asyncio.Queue | None = None

        # Background receive task
        self._receive_task: asyncio.Task | None = None

        # Function call handling
        self._pending_function_calls: dict[str, dict[str, Any]] = {}
        self._tool_tasks: set[asyncio.Task] = set()

        # Current response tracking
        self._current_response: LiveResponse | None = None
        self._response_futures: dict[str, asyncio.Future] = {}

        # Session resumption tracking
        self._session_resumption_handle: str | None = None
        self._session_resumable: bool = False

        # GoAway tracking
        self._go_away_time_left: int | None = None
        # Number of frontend/clients currently using this live session
        self._connection_count: int = 0

    @property
    def connected(self) -> bool:
        """Return connection status."""
        return self._connected

    @property
    def _model(self) -> str:
        """Return the configured model name."""
        return self._session_config.model

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to registered handlers."""
        for handler in list(self._event_handlers.get(event_type, [])):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                _LOGGER.error("Error in event handler for %s: %s", event_type, e)

    def _function_behavior(self) -> types.Behavior | None:
        """Return the function behavior the configured model requires.

        gemini-3.8-live-extended-thinking only supports NON_BLOCKING calls.
        Other models keep the default (blocking), so Home Assistant state
        queries are answered before the model continues speaking.
        """
        if self._model in MODELS_NON_BLOCKING_ONLY:
            return types.Behavior.NON_BLOCKING
        return None

    def _build_tools(self) -> list[types.Tool]:
        """Build tools list for the session."""
        tools_list = []

        # Add Google Search if enabled
        if self._session_config.enable_google_search:
            tools_list.append(types.Tool(google_search=types.GoogleSearch()))

        # Add function declarations from config. The full JSON schema is
        # passed through so nested objects, arrays, enums and required
        # fields from Home Assistant and MCP tools are preserved.
        if self._session_config.tools:
            behavior = self._function_behavior()
            function_declarations = []
            for tool in self._session_config.tools:
                parameters = tool.get("parameters") or {}
                decl_kwargs: dict[str, Any] = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                if parameters.get("properties"):
                    decl_kwargs["parameters_json_schema"] = _clean_json_schema(parameters)
                if behavior is not None:
                    decl_kwargs["behavior"] = behavior
                function_declarations.append(types.FunctionDeclaration(**decl_kwargs))

            if function_declarations:
                tools_list.append(types.Tool(function_declarations=function_declarations))

        return tools_list

    def _build_thinking_config(self) -> types.ThinkingConfig | None:
        """Build thinking config for models that accept a thinking level.

        gemini-3.8-live rejects thinking_level, so it is omitted there.
        """
        supported_levels = MODEL_THINKING_LEVELS.get(self._model)
        if not supported_levels:
            return None

        level = self._session_config.thinking_level
        if level not in supported_levels:
            _LOGGER.debug(
                "Thinking level %s not supported by %s, using %s",
                level,
                self._model,
                supported_levels[0],
            )
            level = supported_levels[0]

        return types.ThinkingConfig(thinking_level=types.ThinkingLevel(level.upper()))

    def _build_config(self) -> types.LiveConnectConfig:
        """Build LiveConnectConfig for the session."""
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self._session_config.voice
                )
            )
        )

        config_kwargs: dict[str, Any] = {
            # Native audio models only support the AUDIO response modality
            "response_modalities": [types.Modality.AUDIO],
            "speech_config": speech_config,
            "realtime_input_config": types.RealtimeInputConfig(
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
            ),
            "temperature": self._session_config.temperature,
        }

        if self._session_config.instructions:
            config_kwargs["system_instruction"] = self._session_config.instructions

        if self._session_config.media_resolution:
            config_kwargs["media_resolution"] = self._session_config.media_resolution

        if self._session_config.context_window_compression:
            config_kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                trigger_tokens=self._session_config.trigger_tokens,
                sliding_window=types.SlidingWindow(
                    target_tokens=self._session_config.target_tokens
                ),
            )

        tools = self._build_tools()
        if tools:
            config_kwargs["tools"] = tools

        thinking_config = self._build_thinking_config()
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        if self._session_config.output_audio_transcription:
            config_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

        if self._session_config.input_audio_transcription:
            config_kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()

        # Affective dialog / proactive audio are not available on 3.1 Flash Live
        native_audio_features = self._model not in MODELS_WITHOUT_NATIVE_AUDIO_FEATURES
        if self._session_config.enable_affective_dialog and native_audio_features:
            config_kwargs["enable_affective_dialog"] = True

        if self._session_config.enable_proactive_audio and native_audio_features:
            config_kwargs["proactivity"] = types.ProactivityConfig(proactive_audio=True)

        if self._session_config.enable_session_resumption:
            config_kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=self._session_config.session_resumption_handle
            )

        return types.LiveConnectConfig(**config_kwargs)

    async def _open_session(self) -> None:
        """Open a Live API session with the current configuration."""
        if self._client is None:
            # Create client in executor to avoid blocking SSL operations
            def create_client() -> genai.Client:
                return genai.Client(
                    http_options={"api_version": API_VERSION},
                    api_key=self._api_key,
                )

            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(None, create_client)

        self._session_context = self._client.aio.live.connect(
            model=self._model,
            config=self._build_config(),
        )
        self._session = await self._session_context.__aenter__()

    async def _close_session(self) -> None:
        """Close the current Live API session, if any."""
        session_context = self._session_context
        self._session = None
        self._session_context = None
        if session_context:
            try:
                await session_context.__aexit__(None, None, None)
            except Exception as e:
                _LOGGER.debug("Error closing session context: %s", e)

    async def connect(self) -> bool:
        """Connect to Gemini Live API.

        Supports:
        - Session resumption via session_resumption_handle
        - Automatic reconnection with stored resumption handle
        """
        # If already connected, increment connection count and return
        if self._connected:
            self._connection_count += 1
            _LOGGER.info("Additional client connected (count=%d)", self._connection_count)
            await self._emit(EVENT_SESSION_STARTED, {"connected": True, "client_count": self._connection_count})
            return True

        try:
            is_resuming = bool(self._session_config.session_resumption_handle)

            self._closing = False
            self._reconnect_requested = False
            await self._open_session()

            self._connected = True
            self._connection_count = 1
            self._audio_in_queue = asyncio.Queue()
            self._receive_task = asyncio.create_task(self._receive_loop())

            if is_resuming:
                await self._emit(EVENT_SESSION_RESUMED, {
                    "connected": True,
                    "resumed_handle": self._session_config.session_resumption_handle,
                    "client_count": self._connection_count,
                })
                _LOGGER.info("Resumed Gemini Live API session (model=%s)", self._model)
            else:
                await self._emit(EVENT_SESSION_STARTED, {"connected": True, "client_count": self._connection_count})
                _LOGGER.info("Connected to Gemini Live API (model=%s)", self._model)

            return True

        except Exception as e:
            _LOGGER.error("Failed to connect to Gemini Live API: %s", e)
            await self._close_session()
            await self._emit(EVENT_ERROR, {"error": str(e)})
            return False

    async def disconnect(self) -> None:
        """Disconnect from Gemini Live API."""
        if not self._connected:
            return

        # If multiple clients are still using the session, only decrement the count
        if self._connection_count > 1:
            self._connection_count -= 1
            _LOGGER.info("Client disconnected, remaining clients=%d", self._connection_count)
            return

        # Last client disconnecting — tear down the session
        self._connection_count = 0
        self._closing = True
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        for task in list(self._tool_tasks):
            task.cancel()

        await self._close_session()

        # A deliberate disconnect starts the next connect with a fresh session
        self._session_config.session_resumption_handle = None
        self._pending_function_calls.clear()
        _LOGGER.info("Disconnected from Gemini Live API (all clients disconnected)")

    async def _reconnect(self) -> bool:
        """Reconnect transparently using the latest session resumption handle.

        Called when the server announced the end of the connection (GoAway)
        or the WebSocket was closed. Returns True if a resumed session is open.
        """
        if self._closing or not self._session_config.enable_session_resumption:
            return False
        if not self._session_resumption_handle:
            _LOGGER.warning("Connection lost and no resumption handle available")
            return False

        self._reconnecting = True
        self._reconnect_requested = False
        await self._emit(EVENT_RECONNECTING, {"time_left": self._go_away_time_left})
        try:
            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                await self._close_session()
                self._session_config.session_resumption_handle = self._session_resumption_handle
                try:
                    await self._open_session()
                except Exception as e:
                    _LOGGER.warning(
                        "Reconnect attempt %d/%d failed: %s", attempt, MAX_RECONNECT_ATTEMPTS, e
                    )
                    await asyncio.sleep(RECONNECT_BACKOFF_SECONDS * attempt)
                    continue

                self._go_away_time_left = None
                # Function calls of the old connection can no longer be answered
                self._pending_function_calls.clear()
                await self._emit(EVENT_SESSION_RESUMED, {
                    "connected": True,
                    "resumed_handle": self._session_resumption_handle,
                    "client_count": self._connection_count,
                })
                _LOGGER.info("Resumed Gemini Live API session after connection end")
                return True
        finally:
            self._reconnecting = False

        return False

    async def _receive_loop(self) -> None:
        """Background task to receive responses from the API.

        session.receive() yields messages until the end of the current turn,
        so it is called repeatedly. Connection ends (GoAway, closed socket)
        are handled by resuming the session with the latest handle.
        """
        while self._connected and self._session:
            try:
                if self._current_response is None:
                    self._current_response = LiveResponse(id=str(uuid.uuid4()), status="in_progress")

                async for response in self._session.receive():
                    await self._handle_server_message(response)

                # Turn boundary: good moment to move to a fresh connection
                if self._reconnect_requested and not await self._reconnect():
                    break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closing or not self._connected:
                    break
                _LOGGER.info("Gemini Live connection ended: %s", e)
                if await self._reconnect():
                    continue
                self._connected = False
                await self._emit(EVENT_ERROR, {"error": "Connection closed"})
                break

    async def _handle_server_message(self, response: types.LiveServerMessage) -> None:
        """Dispatch a single server message."""
        if response.server_content:
            await self._handle_server_content(response.server_content)

        if response.tool_call:
            await self._handle_tool_call(response.tool_call)

        if response.tool_call_cancellation:
            ids = list(response.tool_call_cancellation.ids or [])
            for call_id in ids:
                self._pending_function_calls.pop(call_id, None)
            _LOGGER.debug("Tool calls cancelled by server: %s", ids)
            await self._emit(EVENT_TOOL_CALL_CANCELLATION, {"ids": ids})

        if response.session_resumption_update:
            update = response.session_resumption_update
            self._session_resumable = bool(update.resumable)
            if update.resumable and update.new_handle:
                self._session_resumption_handle = update.new_handle
                await self._emit(EVENT_SESSION_RESUMPTION_UPDATE, {
                    "handle": update.new_handle,
                    "resumable": True,
                })
                _LOGGER.debug("Session resumption handle updated")

        if response.go_away:
            time_left = None
            if response.go_away.time_left is not None:
                time_left = self._parse_time_left(response.go_away.time_left)
            self._go_away_time_left = time_left
            self._reconnect_requested = True
            await self._emit(EVENT_GO_AWAY, {
                "time_left": time_left,
                "resumption_handle": self._session_resumption_handle,
            })
            _LOGGER.info("Received GoAway - connection ends in %s seconds, will resume", time_left)

    @staticmethod
    def _parse_time_left(time_left: Any) -> int | None:
        """Parse GoAway time_left (timedelta, number or '50s')."""
        if isinstance(time_left, timedelta):
            return int(time_left.total_seconds())
        if isinstance(time_left, (int, float)):
            return int(time_left)
        if isinstance(time_left, str):
            try:
                return int(float(time_left.rstrip("smSM").strip()))
            except ValueError:
                _LOGGER.debug("Could not parse time_left: %s", time_left)
        return None

    async def _handle_server_content(self, server_content: types.LiveServerContent) -> None:
        """Handle model output, transcriptions and turn signals."""
        if self._current_response is None:
            self._current_response = LiveResponse(id=str(uuid.uuid4()), status="in_progress")
        current = self._current_response

        # Interruption (barge-in) is normal, not an error
        if server_content.interrupted:
            if self._audio_in_queue:
                while not self._audio_in_queue.empty():
                    self._audio_in_queue.get_nowait()
            await self._emit(EVENT_INTERRUPTED, {"interrupted": True})
            _LOGGER.debug("Received interruption signal from API")
            return

        if server_content.model_turn and server_content.model_turn.parts:
            for part in server_content.model_turn.parts:
                # Thought summaries are not spoken output
                if part.thought:
                    continue

                inline_data = part.inline_data
                if inline_data and isinstance(inline_data.data, bytes) and inline_data.data:
                    audio_data = inline_data.data
                    if self._audio_in_queue:
                        self._audio_in_queue.put_nowait(audio_data)
                    current.audio_data += audio_data
                    await self._emit(EVENT_AUDIO_DELTA, {
                        "audio": base64.b64encode(audio_data).decode("utf-8")
                    })

                if part.text:
                    current.text += part.text
                    await self._emit(EVENT_TEXT_DELTA, {"text": part.text})

        if server_content.output_transcription and server_content.output_transcription.text:
            transcript = server_content.output_transcription.text
            current.audio_transcript += transcript
            await self._emit(EVENT_OUTPUT_TRANSCRIPTION, {"text": transcript})

        if server_content.input_transcription and server_content.input_transcription.text:
            await self._emit(EVENT_INPUT_TRANSCRIPTION, {
                "text": server_content.input_transcription.text
            })

        if server_content.generation_complete:
            self._session_resumable = True
            await self._emit(EVENT_GENERATION_COMPLETE, {
                "resumable": True,
                "has_handle": bool(self._session_resumption_handle),
            })

        if server_content.turn_complete:
            # With background reasoning (extended thinking), turn_complete does
            # not mean the model is idle; IN_PROGRESS keeps the response open.
            interaction_status = getattr(server_content, "interaction_status", None)
            still_working = interaction_status == types.InteractionStatus.IN_PROGRESS

            if not still_working:
                current.status = "completed"
                future = self._response_futures.pop(current.id, None)
                if future and not future.done():
                    future.set_result(current)
                self._current_response = None

            await self._emit(EVENT_TURN_COMPLETE, {
                "transcript": current.audio_transcript,
                "in_progress": still_working,
            })

    async def _handle_tool_call(self, tool_call: types.LiveServerToolCall) -> None:
        """Handle function/tool calls from the model."""
        for function_call in tool_call.function_calls or []:
            call_id = function_call.id or ""
            name = function_call.name or ""
            args = dict(function_call.args) if function_call.args else {}

            _LOGGER.info("Function call: %s with args: %s", name, args)

            self._pending_function_calls[call_id] = {"name": name, "args": args}

            # Run handlers (which execute the tool) in the background so the
            # receive loop keeps reading, e.g. tool_call_cancellation on barge-in
            task = asyncio.create_task(self._emit(EVENT_FUNCTION_CALL, {
                "call_id": call_id,
                "name": name,
                "arguments": args,
            }))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def send_text(self, text: str, turn_complete: bool = True) -> LiveResponse:
        """Send text message to the API and wait for response.

        Uses session.send_client_content(). Note: turn_complete=True
        interrupts any ongoing generation.
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to Gemini Live API")

        self._current_response = LiveResponse(id=str(uuid.uuid4()), status="in_progress")
        response_future: asyncio.Future[LiveResponse] = asyncio.get_running_loop().create_future()
        self._response_futures[self._current_response.id] = response_future

        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=turn_complete,
        )

        try:
            return await asyncio.wait_for(response_future, timeout=60.0)
        except asyncio.TimeoutError:
            _LOGGER.error("Timeout waiting for response")
            if self._current_response:
                self._current_response.status = "timeout"
                return self._current_response
            raise

    async def _send_realtime_input(self, **kwargs: Any) -> None:
        """Send realtime input, dropping it while the session is being resumed."""
        if not self._connected or not self._session or self._reconnecting:
            return
        try:
            await self._session.send_realtime_input(**kwargs)
        except Exception as e:
            # The receive loop notices the closed connection and resumes it
            _LOGGER.debug("Dropping realtime input, connection unavailable: %s", e)

    async def send_audio(self, audio_data: bytes, sample_rate: int = SEND_SAMPLE_RATE) -> None:
        """Send audio data to the API (16-bit PCM, mono, 16kHz native)."""
        await self._send_realtime_input(
            audio=types.Blob(data=audio_data, mime_type=f"audio/pcm;rate={sample_rate}")
        )

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = SEND_SAMPLE_RATE) -> None:
        """Send base64 encoded audio to the API."""
        await self.send_audio(base64.b64decode(audio_b64), sample_rate)

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream.

        When the audio stream is paused for more than a second, send this to
        flush cached audio. Also lets the server finalize the turn faster.
        """
        await self._send_realtime_input(audio_stream_end=True)

    async def send_image(self, image_data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send a video frame (max. 1 frame per second) to the API."""
        await self._send_realtime_input(video=types.Blob(data=image_data, mime_type=mime_type))

    async def send_image_base64(self, image_b64: str, mime_type: str = "image/jpeg") -> None:
        """Send base64 encoded image to the API."""
        await self.send_image(base64.b64decode(image_b64), mime_type)

    async def send_function_result(
        self,
        call_id: str,
        result: Any,
    ) -> None:
        """Send function call result back to the API.

        Results for calls the server cancelled (barge-in) are dropped.
        """
        if not self._connected or not self._session:
            return

        func_info = self._pending_function_calls.pop(call_id, None)
        if func_info is None:
            _LOGGER.debug("Dropping result for unknown or cancelled call_id=%s", call_id)
            return

        # FunctionResponse.response must be an object
        if not isinstance(result, dict):
            result = {"result": result}

        try:
            await self._session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(id=call_id, name=func_info["name"], response=result)
                ]
            )
            _LOGGER.info("Sent function result for call_id=%s", call_id)
        except Exception as e:
            _LOGGER.error("Error sending function result: %s", e)

    async def get_audio_chunk(self, timeout: float = 0.1) -> bytes | None:
        """Get an audio chunk from the receive queue."""
        if not self._audio_in_queue:
            return None

        try:
            return await asyncio.wait_for(self._audio_in_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def add_tool(self, tool: dict[str, Any]) -> None:
        """Add a function tool to the session."""
        self._session_config.tools.append(tool)

    def get_conversation_history(self) -> list[ConversationItem]:
        """Get the conversation history."""
        return []

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._current_response = None
        self._pending_function_calls.clear()

    def update_config(self, **kwargs) -> None:
        """Update session configuration."""
        for key, value in kwargs.items():
            if hasattr(self._session_config, key):
                setattr(self._session_config, key, value)

    # Session Management Methods

    def get_resumption_handle(self) -> str | None:
        """Get the current session resumption handle (valid for 2 hours)."""
        return self._session_resumption_handle

    def is_session_resumable(self) -> bool:
        """Check if the current session is resumable."""
        return self._session_resumable

    def set_resumption_handle(self, handle: str | None) -> None:
        """Set the resumption handle for the next connection.

        Call this before connect() to resume a previous session.
        """
        self._session_config.session_resumption_handle = handle

    def get_go_away_time_left(self) -> int | None:
        """Get the time left before connection termination (if GoAway received)."""
        return self._go_away_time_left

    @staticmethod
    async def create_ephemeral_token(
        api_key: str,
        model: str = DEFAULT_MODEL,
        expire_time_seconds: int = 1800,
        new_session_expire_seconds: int = 60,
    ) -> dict[str, Any] | None:
        """Create a single-use ephemeral token for client-side authentication.

        The token is locked to the given model so a browser cannot change it.
        """
        try:
            client = genai.Client(
                http_options={"api_version": EPHEMERAL_TOKEN_API_VERSION},
                api_key=api_key,
            )

            now = datetime.now(tz=timezone.utc)
            expire_time = now + timedelta(seconds=expire_time_seconds)

            token = await client.aio.auth_tokens.create(
                config=types.CreateAuthTokenConfig(
                    uses=1,
                    expire_time=expire_time,
                    new_session_expire_time=now + timedelta(seconds=new_session_expire_seconds),
                    live_connect_constraints=types.LiveConnectConstraints(model=model),
                )
            )

            return {
                "name": token.name or "",
                "expire_time": expire_time.isoformat(),
            }

        except Exception as e:
            _LOGGER.error("Failed to create ephemeral token: %s", e)
            return None
