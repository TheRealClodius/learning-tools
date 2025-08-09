# Memory-Signal MCP Server Connection Issue Analysis

## Problem Summary
The Client Agent is stuck for more than 10 minutes when trying to connect to the Memory-Signal MCP server. The issue is caused by a protocol mismatch between the client and server.

## Root Cause Analysis

### 1. Protocol Mismatch
- **Error Message**: `"Client must accept text/event-stream"`
- **Cause**: The MCP server at `https://memory-signal-production.up.railway.app/mcp` expects clients to accept `text/event-stream` content type for Server-Sent Events (SSE)
- **Client Issue**: The `streamablehttp_client` from the MCP library is not sending the correct `Accept` header

### 2. Connection Details
- **Server URL**: `https://memory-signal-production.up.railway.app/mcp`
- **Server Response**: HTTP 406 Not Acceptable
- **Required Header**: `Accept: text/event-stream`

### 3. Client Configuration Issues
- The `memory_mcp.py` client uses `streamablehttp_client` which doesn't properly set the required headers
- The client has a 5-second timeout (`self._fast_timeout = 5.0`) but the connection attempt may hang before reaching this timeout
- Missing MCP module dependency (`ModuleNotFoundError: No module named 'mcp'`)

## Why The Agent Gets Stuck

1. **Connection Attempt**: The client tries to connect to the MCP server
2. **Header Mismatch**: Server rejects the connection with 406 error
3. **No Proper Error Handling**: The client doesn't properly handle the 406 error and may retry indefinitely
4. **Async Context**: The asyncio context may not properly propagate timeouts, causing the agent to hang

## Solutions

### Option 1: Fix the Client Headers (Recommended)
Modify the `streamablehttp_client` call to include proper headers:

```python
# In memory_mcp.py, line 87
headers = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache"
}
self._client_cm = streamablehttp_client(self.server_url, headers=headers)
```

### Option 2: Use Alternative MCP Server
The Memory-Signal repository includes a local MCP server implementation in `memoryos-mcp/server_new.py` that can be run locally instead of using the production Railway deployment.

### Option 3: Install Missing Dependencies
```bash
pip install mcp
# or
pip install mcp[client]
```

### Option 4: Implement Proper Timeout Handling
Add explicit timeout handling around the connection attempt:

```python
try:
    async with asyncio.timeout(10):  # 10 second hard timeout
        await self._ensure_session()
except asyncio.TimeoutError:
    logger.error("Failed to connect to MCP server within timeout")
    raise
```

## Immediate Workaround

1. **Disable Memory MCP**: Comment out or remove memory-related tools from the tool executor
2. **Use Local Server**: Run the MCP server locally from the Memory-Signal repository
3. **Set Environment Variables**: Configure to use a different MCP server:
   ```bash
   export MEMORYOS_MCP_HOST=localhost
   export MEMORYOS_MCP_PORT=8000
   export MEMORYOS_MCP_PATH=/
   ```

## Verification Steps

1. Test server connectivity:
   ```bash
   curl -H "Accept: text/event-stream" https://memory-signal-production.up.railway.app/mcp
   ```

2. Check if MCP module is installed:
   ```bash
   python3 -c "import mcp; print(mcp.__version__)"
   ```

3. Monitor connection attempts:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # This will show detailed connection logs
   ```

## Conclusion

The issue is a protocol mismatch where the MCP server expects Server-Sent Events (SSE) protocol with `text/event-stream` content type, but the client is not sending the correct headers. This causes the server to reject the connection with a 406 error, and the client's error handling doesn't properly recover from this, leading to the agent being stuck.