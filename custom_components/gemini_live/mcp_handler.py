"""MCP (Model Context Protocol) server handler for Gemini Live integration."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant

from .const import (
    CONF_MCP_SERVER_ARGS,
    CONF_MCP_SERVER_COMMAND,
    CONF_MCP_SERVER_ENABLED,
    CONF_MCP_SERVER_ENV,
    CONF_MCP_SERVER_NAME,
    CONF_MCP_SERVER_TIMEOUT,
    CONF_MCP_SERVER_TOKEN,
    CONF_MCP_SERVER_TYPE,
    CONF_MCP_SERVER_URL,
    MCP_SERVER_TYPE_SSE,
    MCP_SERVER_TYPE_STDIO,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    server_type: str = MCP_SERVER_TYPE_SSE  # 'sse' or 'stdio'
    url: str = ""  # For SSE servers
    token: str | None = None  # For SSE servers
    command: str = ""  # For stdio servers
    args: list[str] = field(default_factory=list)  # For stdio servers
    env: dict[str, str] = field(default_factory=dict)  # For stdio servers
    timeout: int = 30  # Initialization timeout in seconds (for stdio servers)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        """Create config from dictionary."""
        return cls(
            name=data.get(CONF_MCP_SERVER_NAME, "mcp_server"),
            server_type=data.get(CONF_MCP_SERVER_TYPE, MCP_SERVER_TYPE_SSE),
            url=data.get(CONF_MCP_SERVER_URL, ""),
            token=data.get(CONF_MCP_SERVER_TOKEN),
            command=data.get(CONF_MCP_SERVER_COMMAND, ""),
            args=data.get(CONF_MCP_SERVER_ARGS, []),
            env=data.get(CONF_MCP_SERVER_ENV, {}),
            timeout=data.get(CONF_MCP_SERVER_TIMEOUT, 30),
            enabled=data.get(CONF_MCP_SERVER_ENABLED, True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            CONF_MCP_SERVER_NAME: self.name,
            CONF_MCP_SERVER_TYPE: self.server_type,
            CONF_MCP_SERVER_URL: self.url,
            CONF_MCP_SERVER_TOKEN: self.token,
            CONF_MCP_SERVER_COMMAND: self.command,
            CONF_MCP_SERVER_ARGS: self.args,
            CONF_MCP_SERVER_ENV: self.env,
            CONF_MCP_SERVER_TIMEOUT: self.timeout,
            CONF_MCP_SERVER_ENABLED: self.enabled,
        }


@dataclass
class MCPServer:
    """Represents an MCP server."""

    config: MCPServerConfig
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def server_type(self) -> str:
        """Get server type."""
        return self.config.server_type

    @property
    def url(self) -> str:
        """Get server URL (for SSE)."""
        return self.config.url

    @property
    def token(self) -> str | None:
        """Get server token (for SSE)."""
        return self.config.token

    @property
    def enabled(self) -> bool:
        """Check if server is enabled."""
        return self.config.enabled


class MCPTransport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result."""
        pass


class SSETransport(MCPTransport):
    """SSE (Server-Sent Events) transport for MCP."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        token: str | None = None,
    ) -> None:
        """Initialize SSE transport."""
        self._session = session
        self._url = url
        self._token = token
        self._connected = False
        self._request_id = 0

    async def connect(self) -> bool:
        """Connect to the SSE server."""
        try:
            headers = {"Content-Type": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"

            # Test connection with initialize
            async with self._session.post(
                self._url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": self._get_request_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "home-assistant-gemini-live",
                            "version": "1.0.0"
                        }
                    },
                },
            ) as response:
                if response.status == 200:
                    self._connected = True
                    return True
        except aiohttp.ClientError as e:
            _LOGGER.error("SSE connection error: %s", e)
        return False

    async def disconnect(self) -> None:
        """Disconnect from the SSE server."""
        self._connected = False

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._get_request_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        try:
            async with self._session.post(
                self._url,
                headers=headers,
                json=request,
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": {"code": response.status, "message": error_text}}
        except aiohttp.ClientError as e:
            return {"error": {"code": -1, "message": str(e)}}


class StdioTransport(MCPTransport):
    """Stdio transport for MCP servers."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize stdio transport."""
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._connected = False
        self._lock = asyncio.Lock()
        self._response_futures: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        """Start the stdio MCP server process."""
        try:
            # Prepare environment
            process_env = os.environ.copy()

            # Auto-add UV environment variables for uv/uvx commands in HASSIO
            # This avoids permission issues with /tmp noexec mount
            if self._command in ("uv", "uvx"):
                uv_defaults = {
                    "UV_TOOL_DIR": "/config/.uv/tools",
                    "UV_CACHE_DIR": "/config/.uv/cache",
                    "TMPDIR": "/config/.uv/tmp",
                }
                for key, default_value in uv_defaults.items():
                    if key not in self._env:
                        self._env[key] = default_value
                        _LOGGER.debug("Auto-set %s=%s for uvx/uv command", key, default_value)

            process_env.update(self._env)

            # Create directories for environment variables that look like paths
            # This helps with uvx/uv which need UV_TOOL_DIR, UV_CACHE_DIR, TMPDIR
            for key, value in self._env.items():
                if key in ("UV_TOOL_DIR", "UV_CACHE_DIR", "TMPDIR", "UV_TOOL_BIN_DIR"):
                    try:
                        os.makedirs(value, exist_ok=True)
                        _LOGGER.debug("Created directory for %s: %s", key, value)
                    except Exception as e:
                        _LOGGER.warning("Failed to create directory %s: %s", value, e)

            # Start the process
            cmd = [self._command] + self._args
            _LOGGER.info("Starting stdio MCP server: %s", " ".join(cmd))

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )

            # Start reading responses and stderr
            self._read_task = asyncio.create_task(self._read_responses())
            # Also read stderr to prevent buffer blocking
            asyncio.create_task(self._read_stderr())

            _LOGGER.debug("Sending MCP initialize request (timeout: %ds)", self._timeout)

            # Send initialize request with configurable timeout
            result = await self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "home-assistant-gemini-live",
                        "version": "1.0.0"
                    }
                },
                timeout=float(self._timeout),
            )

            _LOGGER.debug("MCP initialize response: %s", result)

            if "error" not in result:
                # Send initialized notification
                await self._send_notification("notifications/initialized", {})
                self._connected = True
                _LOGGER.info("Stdio MCP server connected successfully")
                return True
            else:
                _LOGGER.error("Stdio MCP server initialization failed: %s", result)
                return False

        except Exception as e:
            _LOGGER.error("Failed to start stdio MCP server: %s", e)
            return False

    async def disconnect(self) -> None:
        """Stop the stdio MCP server process."""
        self._connected = False

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception as e:
                _LOGGER.error("Error stopping stdio process: %s", e)
            finally:
                self._process = None

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _read_responses(self) -> None:
        """Read responses from the process stdout."""
        if not self._process or not self._process.stdout:
            _LOGGER.error("No process or stdout available for reading")
            return

        _LOGGER.debug("Started reading MCP server stdout")
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    _LOGGER.debug("MCP server stdout closed (EOF)")
                    break

                _LOGGER.debug("MCP server stdout raw: %s", line[:200])
                try:
                    response = json.loads(line.decode())
                    request_id = response.get("id")
                    _LOGGER.debug("MCP response received, id=%s, keys=%s", request_id, list(response.keys()))
                    if request_id and request_id in self._response_futures:
                        future = self._response_futures.pop(request_id)
                        if not future.done():
                            future.set_result(response)
                    else:
                        _LOGGER.debug("No pending future for response id=%s", request_id)
                except json.JSONDecodeError as e:
                    _LOGGER.debug("JSON decode error: %s, line: %s", e, line[:100])
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:
            _LOGGER.error("Error reading stdio responses: %s", e)

    async def _read_stderr(self) -> None:
        """Read and log stderr from the process to prevent buffer blocking."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                # Log stderr output for debugging
                stderr_text = line.decode().strip()
                if stderr_text:
                    _LOGGER.debug("MCP server stderr: %s", stderr_text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _LOGGER.debug("Error reading stderr: %s", e)

    async def _send_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        """Send a notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            message = json.dumps(notification) + "\n"
            self._process.stdin.write(message.encode())
            await self._process.stdin.drain()
        except Exception as e:
            _LOGGER.error("Error sending notification: %s", e)

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response.
        
        Args:
            method: The JSON-RPC method to call
            params: Optional parameters for the method
            timeout: Timeout in seconds (default 30s, use longer for init)
        """
        if not self._process or not self._process.stdin:
            return {"error": {"code": -1, "message": "Process not running"}}

        async with self._lock:
            request_id = self._get_request_id()
            request: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params:
                request["params"] = params

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._response_futures[request_id] = future

            try:
                message = json.dumps(request) + "\n"
                self._process.stdin.write(message.encode())
                await self._process.stdin.drain()

                # Wait for response with timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result

            except asyncio.TimeoutError:
                self._response_futures.pop(request_id, None)
                return {"error": {"code": -1, "message": f"Request timeout after {timeout}s"}}
            except Exception as e:
                self._response_futures.pop(request_id, None)
                return {"error": {"code": -1, "message": str(e)}}


class MCPServerHandler:
    """Handler for MCP server connections and tool management."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the MCP server handler."""
        self._session = session
        self._servers: dict[str, MCPServer] = {}
        self._transports: dict[str, MCPTransport] = {}
        self._tools: dict[str, MCPTool] = {}
        self._function_map: dict[str, tuple[str, str]] = {}

    def add_server(
        self,
        name: str,
        url: str = "",
        token: str | None = None,
        server_type: str = MCP_SERVER_TYPE_SSE,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> MCPServer:
        """Add an MCP server configuration."""
        config = MCPServerConfig(
            name=name,
            server_type=server_type,
            url=url,
            token=token,
            command=command,
            args=args or [],
            env=env or {},
            enabled=enabled,
        )
        server = MCPServer(config=config)
        self._servers[name] = server
        _LOGGER.info("Added MCP server: %s (type: %s)", name, server_type)
        return server

    def add_server_from_config(self, config: dict[str, Any]) -> MCPServer:
        """Add an MCP server from configuration dictionary."""
        server_config = MCPServerConfig.from_dict(config)
        server = MCPServer(config=server_config)
        self._servers[server_config.name] = server
        _LOGGER.info(
            "Added MCP server from config: %s (type: %s)",
            server_config.name,
            server_config.server_type,
        )
        return server

    def update_server(self, name: str, config: dict[str, Any]) -> MCPServer | None:
        """Update an existing MCP server configuration."""
        if name not in self._servers:
            return None

        # Disconnect if connected
        if name in self._transports:
            asyncio.create_task(self._transports[name].disconnect())
            del self._transports[name]

        # Update config
        server_config = MCPServerConfig.from_dict(config)
        server = MCPServer(config=server_config)
        self._servers[name] = server
        _LOGGER.info("Updated MCP server: %s", name)
        return server

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server."""
        if name in self._servers:
            # Remove associated tools
            server = self._servers[name]
            for tool in server.tools:
                tool_key = f"{name}:{tool.name}"
                if tool_key in self._tools:
                    del self._tools[tool_key]

            # Disconnect transport
            if name in self._transports:
                asyncio.create_task(self._transports[name].disconnect())
                del self._transports[name]

            del self._servers[name]
            _LOGGER.info("Removed MCP server: %s", name)
            return True
        return False

    def _create_transport(self, server: MCPServer) -> MCPTransport:
        """Create appropriate transport for server type."""
        if server.server_type == MCP_SERVER_TYPE_STDIO:
            return StdioTransport(
                command=server.config.command,
                args=server.config.args,
                env=server.config.env,
                timeout=server.config.timeout,
            )
        else:  # SSE is default
            return SSETransport(
                session=self._session,
                url=server.url,
                token=server.token,
            )

    async def connect_server(self, name: str) -> bool:
        """Connect to an MCP server and fetch its tools."""
        if name not in self._servers:
            _LOGGER.error("MCP server not found: %s", name)
            return False

        server = self._servers[name]

        if not server.enabled:
            _LOGGER.info("MCP server %s is disabled", name)
            return False

        try:
            # Create transport if needed
            if name not in self._transports:
                self._transports[name] = self._create_transport(server)

            transport = self._transports[name]

            # Connect
            if not await transport.connect():
                _LOGGER.error("Failed to connect to MCP server: %s", name)
                return False

            # List tools
            result = await transport.send_request("tools/list")

            if "error" in result:
                _LOGGER.error(
                    "Failed to list tools from MCP server %s: %s",
                    name,
                    result.get("error"),
                )
                return False

            tools_data = result.get("result", {}).get("tools", [])

            server.tools.clear()
            for tool_data in tools_data:
                tool = MCPTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=name,
                )
                server.tools.append(tool)
                tool_key = f"{name}:{tool.name}"
                self._tools[tool_key] = tool

            server.connected = True
            _LOGGER.info(
                "Connected to MCP server %s (type: %s), found %d tools",
                name,
                server.server_type,
                len(server.tools),
            )
            return True

        except Exception as e:
            _LOGGER.error("Error connecting to MCP server %s: %s", name, e)
            return False

    async def connect_all_servers(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}
        for name in self._servers:
            results[name] = await self.connect_server(name)
        return results

    async def disconnect_all_servers(self) -> None:
        """Disconnect from all MCP servers."""
        for name, transport in list(self._transports.items()):
            try:
                await transport.disconnect()
            except Exception as e:
                _LOGGER.error("Error disconnecting from %s: %s", name, e)
        self._transports.clear()
        for server in self._servers.values():
            server.connected = False

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an MCP server."""
        if server_name not in self._servers:
            return {"error": f"Server not found: {server_name}"}

        server = self._servers[server_name]

        if not server.connected:
            connected = await self.connect_server(server_name)
            if not connected:
                return {"error": f"Could not connect to server: {server_name}"}

        if server_name not in self._transports:
            return {"error": f"No transport for server: {server_name}"}

        transport = self._transports[server_name]

        try:
            result = await transport.send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments,
                },
            )

            if "error" in result:
                _LOGGER.error(
                    "MCP tool call failed: %s/%s - %s",
                    server_name,
                    tool_name,
                    result.get("error"),
                )
                return {"error": result.get("error")}

            _LOGGER.info(
                "MCP tool call successful: %s/%s",
                server_name,
                tool_name,
            )
            return result.get("result", {})

        except Exception as e:
            _LOGGER.error(
                "Error calling MCP tool %s/%s: %s",
                server_name,
                tool_name,
                e,
            )
            return {"error": str(e)}

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers."""
        return list(self._tools.values())

    def get_server_tools(self, server_name: str) -> list[MCPTool]:
        """Get tools from a specific server."""
        if server_name in self._servers:
            return self._servers[server_name].tools
        return []

    def get_tool(self, server_name: str, tool_name: str) -> MCPTool | None:
        """Get a specific tool."""
        tool_key = f"{server_name}:{tool_name}"
        return self._tools.get(tool_key)

    def get_enabled_servers(self) -> list[MCPServer]:
        """Get all enabled servers (SSE and stdio are both handled locally)."""
        return [server for server in self._servers.values() if server.enabled]

    def get_stdio_servers(self) -> list[MCPServer]:
        """Get list of enabled stdio servers."""
        return [
            server for server in self._servers.values()
            if server.server_type == MCP_SERVER_TYPE_STDIO and server.enabled
        ]

    def get_server_configs(self) -> list[dict[str, Any]]:
        """Get all server configurations as list of dictionaries."""
        return [server.config.to_dict() for server in self._servers.values()]

    def get_tools_as_functions(self) -> list[dict[str, Any]]:
        """Get all MCP tools as Gemini function declarations.

        Gemini Live cannot call MCP servers itself, so every tool (SSE and
        stdio) is exposed as a local function named ``server__tool``.
        Names are sanitized to Gemini's allowed charset; the original
        server/tool names are kept in a lookup table for dispatch.
        """
        functions = []
        self._function_map.clear()
        for tool in self._tools.values():
            func_name = self._make_function_name(tool.server_name, tool.name)
            self._function_map[func_name] = (tool.server_name, tool.name)
            functions.append({
                "name": func_name,
                "description": f"[{tool.server_name}] {tool.description}",
                "parameters": tool.input_schema or {
                    "type": "object",
                    "properties": {},
                },
            })
        return functions

    @staticmethod
    def _make_function_name(server_name: str, tool_name: str) -> str:
        """Create a Gemini-safe function name (max 64 chars)."""
        safe_server = re.sub(r"[^a-zA-Z0-9_]", "_", server_name)
        safe_tool = re.sub(r"[^a-zA-Z0-9_.:-]", "_", tool_name)
        return f"{safe_server}__{safe_tool}"[:64]

    def parse_function_name(self, function_name: str) -> tuple[str, str] | None:
        """Map a function name back to the original server and tool names."""
        if function_name in self._function_map:
            return self._function_map[function_name]
        if "__" in function_name:
            server_name, tool_name = function_name.split("__", 1)
            return server_name, tool_name
        return None

    @property
    def servers(self) -> dict[str, MCPServer]:
        """Get all configured servers."""
        return self._servers.copy()

    @property
    def connected_servers(self) -> list[str]:
        """Get names of connected servers."""
        return [name for name, server in self._servers.items() if server.connected]


class HomeAssistantMCPTools:
    """Built-in Home Assistant MCP-like tools."""

    TOOL_NAMES = frozenset({
        "get_entity_state",
        "call_service",
        "get_entities_by_domain",
        "get_area_entities",
        "list_areas",
    })

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize with Home Assistant instance."""
        self._hass = hass

    def get_builtin_tools(self) -> list[dict[str, Any]]:
        """Get built-in HA tools as function definitions."""
        return [
            {
                "name": "get_entity_state",
                "description": "Get the current state of a Home Assistant entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID (e.g., light.living_room)",
                        },
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "call_service",
                "description": (
                    "Call a Home Assistant service to control devices, e.g. "
                    "domain='light', service='turn_off', target={'entity_id': ['light.kitchen']}. "
                    "Always use exact entity_ids from get_entities_by_domain or "
                    "get_area_entities; never guess them. The result contains the "
                    "entity states after the call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The service domain, e.g. light, switch (optional if service is 'domain.service')",
                        },
                        "service": {
                            "type": "string",
                            "description": "The service, e.g. 'turn_off' (with domain) or 'light.turn_off'",
                        },
                        "target": {
                            "type": "object",
                            "description": "Target entities/areas for the service",
                            "properties": {
                                "entity_id": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Exact entity IDs, e.g. ['light.kitchen']",
                                },
                                "area_id": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Area IDs or area names",
                                },
                            },
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional service data, e.g. {'brightness_pct': 50}",
                        },
                    },
                    "required": ["service"],
                },
            },
            {
                "name": "get_entities_by_domain",
                "description": "List all entities in a specific domain",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain (e.g., light, switch, sensor)",
                        },
                    },
                    "required": ["domain"],
                },
            },
            {
                "name": "get_area_entities",
                "description": (
                    "Get all entities in a room/area. Use this to find the exact "
                    "entity_ids before controlling devices in a room."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area_id": {
                            "type": "string",
                            "description": "The area ID, name or alias (e.g. 'Simon', 'Kitchen')",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Optional domain filter, e.g. 'light'",
                        },
                    },
                    "required": ["area_id"],
                },
            },
            {
                "name": "list_areas",
                "description": "List all rooms/areas with their area_id, name and aliases",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a built-in tool and log failures so they show up in the HA log."""
        handlers = {
            "get_entity_state": self._get_entity_state,
            "call_service": self._call_service,
            "get_entities_by_domain": self._get_entities_by_domain,
            "get_area_entities": self._get_area_entities,
            "list_areas": self._list_areas,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            result: dict[str, Any] = {"error": f"Unknown tool: {tool_name}"}
        else:
            result = await handler(arguments)

        if "error" in result:
            _LOGGER.warning("Tool %s failed for %s: %s", tool_name, arguments, result)
        else:
            _LOGGER.debug("Tool %s(%s) -> %s", tool_name, arguments, result)
        return result

    async def _get_entity_state(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get entity state."""
        entity_id = arguments.get("entity_id", "")
        state = self._hass.states.get(entity_id)

        if state is None:
            return {"error": f"Entity {entity_id} not found"}

        return {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat(),
        }

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        """Normalize a string / list / comma-separated value to a list."""
        if not value:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(v) for v in value if v]

    def _resolve_area_id(self, area: str) -> str | None:
        """Resolve an area id, name or alias to an area id.

        Falls back to a word match so phrases like "Simons Zimmer" or
        "Küche Licht" find the areas "Simon" and "Küche". Returns None if
        nothing or more than one area matches.
        """
        from homeassistant.helpers import area_registry

        area_reg = area_registry.async_get(self._hass)
        if area_reg.async_get_area(area):
            return area
        wanted = area.strip().lower()
        areas = list(area_reg.async_list_areas())
        for entry in areas:
            names = {entry.name.lower(), entry.id.lower(), *(alias.lower() for alias in entry.aliases)}
            if wanted in names:
                return entry.id

        words = [w for w in re.split(r"[^\w]+", wanted) if w]
        matches = set()
        for entry in areas:
            for name in {entry.name.lower(), *(alias.lower() for alias in entry.aliases)}:
                if any(w == name or (len(name) >= 3 and w.startswith(name)) for w in words):
                    matches.add(entry.id)
        if len(matches) == 1:
            return matches.pop()
        return None

    def _area_entity_ids(self, area_id: str) -> list[str]:
        """Return entity_ids in an area, including those assigned via their device."""
        from homeassistant.helpers import device_registry, entity_registry

        device_reg = device_registry.async_get(self._hass)
        entity_reg = entity_registry.async_get(self._hass)
        entity_ids = []
        for entry in entity_reg.entities.values():
            entry_area = entry.area_id
            if entry_area is None and entry.device_id:
                device = device_reg.async_get(entry.device_id)
                entry_area = device.area_id if device else None
            if entry_area == area_id:
                entity_ids.append(entry.entity_id)
        return entity_ids

    async def _list_areas(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List all areas with their ids, names and aliases."""
        from homeassistant.helpers import area_registry

        area_reg = area_registry.async_get(self._hass)
        return {
            "areas": [
                {"area_id": a.id, "name": a.name, "aliases": sorted(a.aliases)}
                for a in area_reg.async_list_areas()
            ]
        }

    async def _call_service(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a Home Assistant service and report the resulting states.

        The model may put entity_id/area_id in ``target``, in ``data`` or at
        the top level; all are accepted. Unknown entities, unknown areas and
        missing targets are reported as errors instead of silently doing
        nothing, so the model cannot claim success for a no-op.
        """
        domain = str(arguments.get("domain", "")).strip().lower()
        service = str(arguments.get("service", "")).strip().lower()
        target = dict(arguments.get("target") or {})
        data = dict(arguments.get("data") or {})

        # Accept the common "light.turn_off" notation in service (with or without domain)
        if "." in service:
            service_domain, service = service.split(".", 1)
            domain = domain or service_domain

        if not self._hass.services.has_service(domain, service):
            return {
                "error": f"Service {domain}.{service} does not exist",
                "hint": "Use services like light.turn_off, switch.turn_on, climate.set_temperature.",
            }

        entity_ids = self._as_list(target.pop("entity_id", None))
        entity_ids += self._as_list(data.pop("entity_id", None))
        entity_ids += self._as_list(arguments.get("entity_id"))
        areas = self._as_list(target.pop("area_id", None))
        areas += self._as_list(data.pop("area_id", None))
        areas += self._as_list(arguments.get("area_id"))

        unknown_entities = [e for e in entity_ids if self._hass.states.get(e) is None]
        if unknown_entities:
            return {
                "error": f"Unknown entity_id(s): {', '.join(unknown_entities)}",
                "hint": "Look up exact entity_ids with get_entities_by_domain or get_area_entities first.",
            }

        area_ids = []
        for area in areas:
            area_id = self._resolve_area_id(area)
            if area_id is None:
                available = await self._list_areas({})
                return {
                    "error": f"Unknown area: {area}",
                    "available_areas": [a["name"] for a in available["areas"]],
                }
            area_ids.append(area_id)

        service_target: dict[str, Any] = {k: v for k, v in target.items() if v}
        if entity_ids:
            service_target["entity_id"] = list(dict.fromkeys(entity_ids))
        if area_ids:
            service_target["area_id"] = list(dict.fromkeys(area_ids))

        if not service_target and domain not in ("homeassistant", "notify", "script", "scene", "automation"):
            return {
                "error": f"No target given for {domain}.{service}",
                "hint": "Pass target.entity_id (e.g. ['light.kitchen']) or target.area_id.",
            }

        _LOGGER.info("Calling %s.%s target=%s data=%s", domain, service, service_target, data)
        try:
            await self._hass.services.async_call(
                domain,
                service,
                service_data=data,
                target=service_target or None,
                blocking=True,
            )
        except Exception as e:
            _LOGGER.warning("Service call %s.%s failed: %s", domain, service, e)
            return {"error": str(e)}

        # Report resulting states so the model can verify the action
        states = {}
        for entity_id in service_target.get("entity_id", []):
            state = self._hass.states.get(entity_id)
            if state:
                states[entity_id] = state.state
        return {
            "success": True,
            "message": f"Called {domain}.{service}",
            "target": service_target,
            "states_after": states,
        }

    async def _get_entities_by_domain(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Get entities by domain."""
        domain = arguments.get("domain", "")
        entities = [
            {
                "entity_id": state.entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
            }
            for state in self._hass.states.async_all()
            if state.entity_id.startswith(f"{domain}.")
        ]
        return {"domain": domain, "entities": entities}

    async def _get_area_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get entities in an area (area id, name, alias or a phrase containing it)."""
        from homeassistant.helpers import area_registry

        query = str(arguments.get("area_id", ""))
        area_id = self._resolve_area_id(query)
        if area_id is None:
            areas = await self._list_areas({})
            return {
                "error": f"Area {query} not found",
                "available_areas": [a["name"] for a in areas["areas"]],
            }

        area = area_registry.async_get(self._hass).async_get_area(area_id)
        domain = str(arguments.get("domain", "")).strip().lower()
        entities = []
        for entity_id in self._area_entity_ids(area_id):
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            state = self._hass.states.get(entity_id)
            if state:
                entities.append({
                    "entity_id": entity_id,
                    "state": state.state,
                    "friendly_name": state.attributes.get("friendly_name"),
                })

        return {"area": area.name if area else area_id, "area_id": area_id, "entities": entities}
