# Slack Chatter MCP Integration

## Overview

The Slack Chatter MCP client provides integration with the Slack Chatter Service, enabling semantic search of Slack messages through the Model Context Protocol (MCP). This integration allows the client agent to search through indexed Slack messages, retrieve channel information, and get search statistics.

## Features

### Available Tools

1. **slack_chatter.search_slack_messages**
   - Semantic search through Slack messages
   - Filter by channel, user, and date range
   - AI-enhanced query processing
   - Returns relevance-scored results

2. **slack_chatter.get_slack_channels**
   - List all indexed Slack channels
   - View available channels for search

3. **slack_chatter.get_slack_search_stats**
   - View index statistics
   - Check total indexed messages
   - Monitor index health and status

## Configuration

### Environment Variables

The following environment variables can be used to configure the MCP client:

```bash
# MCP Server Configuration (optional - defaults shown)
SLACK_CHATTER_MCP_HOST=slack-chatter-service.andreiclodius.repl.co
SLACK_CHATTER_MCP_PORT=5000  # Default Replit port
SLACK_CHATTER_API_KEY=<optional_api_key>  # If authentication is required
```

### Server Requirements

The Slack Chatter Service must be running and accessible. The service is hosted at:
- GitHub Repository: https://github.com/TheRealClodius/Slack-Chatter-Service
- Default Replit URL: https://slack-chatter-service.andreiclodius.repl.co:5000

The service provides the following endpoints:
- `/health` - Health check endpoint
- `/mcp/request` - Main MCP JSON-RPC endpoint
- `/oauth/authorize` - OAuth 2.1 authorization (if using OAuth)
- `/oauth/token` - OAuth 2.1 token exchange
- `/docs` - API documentation

**Note:** The Replit instance must be running for the integration to work. If the server is not accessible, all tool calls will return error responses with appropriate error messages.

## Usage

### Through the Client Agent

The tools are automatically available through the registry system. You can discover them using:

```
reg.search query="slack messages"
```

### Direct Tool Usage

#### Search Slack Messages
```python
result = await execute_tool("slack_chatter.search_slack_messages", {
    "query": "deployment issues",
    "top_k": 10,
    "channel_filter": "engineering",
    "date_from": "2024-01-01"
})
```

#### Get Slack Channels
```python
result = await execute_tool("slack_chatter.get_slack_channels", {})
```

#### Get Search Statistics
```python
result = await execute_tool("slack_chatter.get_slack_search_stats", {})
```

## Architecture

### Components

1. **MCP Client** (`tools/slack_chatter_mcp.py`)
   - Handles connection to the remote MCP server
   - Manages session lifecycle
   - Provides async methods for each tool

2. **Schema Files** (`schemas/services/slack_chatter/`)
   - Input/output schemas for each tool
   - Metadata for registry discovery
   - Validation rules

3. **Integration Points**
   - `runtime/tool_executor.py`: Tool loading and execution
   - `execute.py`: Pydantic models for validation
   - Registry system: Automatic discovery

### Connection Flow

```mermaid
graph LR
    A[Client Agent] --> B[Tool Executor]
    B --> C[Slack Chatter MCP Client]
    C --> D[Remote MCP Server]
    D --> E[Slack Data]
```

## Testing

Run the test script to verify the integration:

```bash
python3 test_slack_chatter.py
```

The test script will:
1. Test connection to the MCP server
2. Retrieve search statistics
3. List available channels
4. Perform sample searches
5. Test filtered searches

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify the MCP server is running
   - Check network connectivity
   - Ensure the Replit instance is active

2. **No Results Returned**
   - Verify Slack channels are indexed
   - Check if the bot has access to channels
   - Ensure search query is valid

3. **Timeout Errors**
   - The server may be slow to respond
   - Try increasing the timeout in the client
   - Check server logs for issues

### Debug Mode

Enable debug logging to see detailed connection information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Limitations

1. **Server Dependency**: Requires the remote MCP server to be running
2. **Rate Limits**: Subject to the server's rate limiting (60 requests/minute)
3. **Data Access**: Only searches indexed Slack messages
4. **Authentication**: Currently uses public endpoint (no auth required)

## Future Enhancements

- [ ] Add OAuth 2.1 authentication support
- [ ] Implement caching for frequently accessed data
- [ ] Add support for real-time message streaming
- [ ] Enable local fallback mode
- [ ] Add batch search capabilities

## Support

For issues or questions:
- Check the [Slack Chatter Service repository](https://github.com/TheRealClodius/Slack-Chatter-Service)
- Review the MCP specification at [modelcontextprotocol.io](https://modelcontextprotocol.io)
- Check the test script output for diagnostic information