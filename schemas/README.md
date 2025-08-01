# Memory Tool Schemas

This directory contains JSON schemas that define the dual memory system API for the MemoryOS integration.

## Architecture Overview

The schemas are organized into **input** and **output** definitions for each memory operation, following a clear client-server contract model.

## Schema Types

### Input Schemas (Client → Server)
These define what the client sends to the MCP server:

| Schema | Purpose |
|--------|---------|
| `add_conversation_input.json` | Store conversation pairs (user input + agent response) |
| `add_execution_input.json` | Store execution details (tools used, errors, reasoning) |
| `retrieve_conversation_input.json` | Query conversation memories with filters |
| `retrieve_execution_input.json` | Query execution patterns for learning |

### Output Schemas (Server → Client) 
These serve as **API contracts** defining what the MCP server should return:

| Schema | Purpose |
|--------|---------|
| `add_conversation_output.json` | Response format for storing conversations |
| `add_execution_output.json` | Response format for storing executions |
| `retrieve_conversation_output.json` | Response format for conversation retrieval |
| `retrieve_execution_output.json` | Response format for execution retrieval |

## Dual Memory System

### Conversation Memory
Stores user prompts and agent responses for building persistent dialogue history:
- **Input**: User input, agent response, metadata, timestamp
- **Output**: Success status, message ID, storage details
- **Linking**: Uses `message_id` to connect with execution memory

### Execution Memory  
Stores detailed execution information for learning and pattern analysis:
- **Input**: Tools used (chronological), errors, reasoning observations
- **Output**: Success status, execution details, performance metrics
- **Linking**: Uses `message_id` to connect with conversation memory

## Message ID Linking

The `message_id` field creates bidirectional links between conversation and execution memories:

```json
{
  "message_id": "msg_123",
  // ... conversation details
}
```

```json
{
  "message_id": "msg_123", 
  // ... execution details
}
```

This enables queries like:
- "What was the execution trail for this conversation?"
- "Show me the conversation that led to this execution pattern"

## User Identification

**Important**: User identification is handled at the MCP transport layer via environment variables (`MEMORY_USER_ID`), not explicitly in schemas. This enables:

- ✅ **Multi-user support** for web and Slack interfaces
- ✅ **Clean schema separation** from authentication concerns  
- ✅ **Infrastructure-level user isolation**

## Schema Usage

### Client Implementation
The client uses:
- ✅ **Input schemas** for request construction
- ✅ **Pydantic models** (generated from schemas) for validation
- ✅ **Dynamic parsing** of server responses

### Server Implementation  
The server uses:
- ✅ **Input schemas** for request validation
- ✅ **Output schemas** as response contracts
- ✅ **User ID injection** from MCP client context

## Timestamp Format

All timestamps use **ISO 8601 format** for consistency:
```json
{
  "timestamp": "2024-12-15T14:30:00Z",
  "retrieval_timestamp": "2024-12-15T14:30:45.123Z"
}
```

## Validation

- **Client-side**: Input validation using Pydantic models
- **Server-side**: Input validation + output contract compliance
- **Runtime**: Graceful error handling for malformed responses

---

**Note**: Output schemas are reference documentation for server implementers. The client parses responses dynamically without strict output validation.