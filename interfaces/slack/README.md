# Slack Interface Architecture

A modular, scalable Slack bot interface built with the Slack Bolt framework. This interface provides real-time message streaming, modal interactions, and intelligent mention handling for AI agent communication.

## 🏗️ Architecture Overview

The Slack interface is organized into focused modules that handle specific responsibilities:

```
interfaces/slack/
├── core_slack_orchestration.py    # Main orchestrator (465 lines)
├── formatters/                    # Content formatting & parsing
├── handlers/                      # Event & interaction processing  
├── services/                      # Business logic & utilities
├── features/                      # Future expansion modules
└── utils/                         # Shared utilities
```

## 📁 Core Components

### 🎯 Core Orchestration
**File**: `core_slack_orchestration.py` (465 lines)

The main orchestrator that:
- Initializes all modular components
- Sets up Slack event handlers (messages, mentions, buttons)
- Coordinates between different services
- Provides FastAPI integration for webhooks

**Key Classes**:
- `SlackInterface` - Main interface class
- `create_slack_app()` - FastAPI handler factory

### 📝 Formatters (`formatters/`)

#### Markdown Parser
**File**: `formatters/markdown_parser.py` (243 lines)

Converts markdown text into Slack Block Kit format:
```python
from interfaces.slack.formatters import MarkdownToSlackParser

parser = MarkdownToSlackParser()
blocks = parser.parse_to_blocks("# Heading\n**Bold text**")
```

**Features**:
- Headers → Slack section blocks
- Bold/italic → mrkdwn formatting
- Code blocks → preformatted text
- Lists → structured blocks
- Smart text splitting for Slack limits

#### Mention Resolver
**File**: `formatters/mention_resolver.py` (231 lines)

Handles @user and #channel mention parsing and resolution:
```python
from interfaces.slack.formatters import SlackMentionResolver

resolver = SlackMentionResolver(cache_service, user_service)
clean_text, context = await resolver.parse_slack_mentions(raw_text, slack_client)
formatted_text = await resolver.resolve_and_format_mentions(response_text, context, slack_client)
```

**Features**:
- Incoming: `<@U123|user>` → `@username`
- Outgoing: `@username` → `<@U123>`
- Channel mentions: `<#C123|general>` ↔ `#general`
- Caching for performance
- Special mentions (@everyone, @here, @channel)

### ⚡ Handlers (`handlers/`)

#### Streaming Handler
**File**: `handlers/streaming_handler.py` (335 lines)

Manages real-time message updates during agent execution:
```python
from interfaces.slack.handlers import SlackStreamingHandler

handler = SlackStreamingHandler(app_client, channel_id, thread_ts)
await handler.start_streaming()
await handler.update_thinking("Processing your request...")
await handler.start_tool("search", {"query": "example"})
await handler.complete_tool("Found 5 results")
await handler.finish_with_response("Here's what I found...")
```

**Features**:
- Progressive message updates
- Tool execution visualization
- Thinking state display
- Final response formatting
- Error handling and recovery

#### Modal Handler
**File**: `handlers/modal_handler.py` (295 lines)

Handles modal interactions for detailed execution views:
```python
from interfaces.slack.handlers import SlackModalHandler

handler = SlackModalHandler(cache_service)
await handler.show_execution_details_modal(body, client)
await handler.update_execution_modal_page(client, body, message_ts, page_index)
```

**Features**:
- Paginated execution details
- Modal size optimization
- Navigation controls
- Error handling for trigger_id timeouts
- Responsive design for mobile

### 🛠️ Services (`services/`)

#### Cache Service
**File**: `services/cache_service.py` (180 lines)

Centralized caching and persistent storage for performance optimization:
```python
from interfaces.slack.services import SlackCacheService

cache = SlackCacheService()
await cache.store_execution_details(message_ts, channel_id, user_id, execution_details)
details = await cache.get_execution_details(message_ts)
user_id = await cache.find_user_id(username, slack_client)
```

**Features**:
- **Persistent execution details storage** (PostgreSQL/SQLite database)
- 7-day automatic expiration with database cleanup
- User ID/name mapping cache (in-memory)
- Channel ID/name mapping cache (in-memory)
- Automatic database cleanup task (every 12 hours)
- Cache statistics and monitoring

#### User Service
**File**: `services/user_service.py` (205 lines)

User information management and timezone handling:
```python
from interfaces.slack.services import SlackUserService

user_service = SlackUserService(slack_client)
user_info = await user_service.get_user_info(user_id)
timezone = await user_service.get_user_timezone(user_id)
```

**Features**:
- User profile fetching
- Timezone resolution
- **Database integration for persistent caching**
- Graceful fallbacks for missing data

#### Execution Summarizer
**File**: `agents/execution_summarizer.py` (175 lines)

LLM-based summarization of tool execution results (moved to agents/ as general utility):
```python
from agents.execution_summarizer import ExecutionSummarizer

summarizer = ExecutionSummarizer()
summary = await summarizer.create_narrative_summary(
    tool_name, tool_args, result_data, initial_narrative
)
```

**Features**:
- **Configurable LLM integration** (Gemini 2.5 Flash by default)
- **Real-time streaming support** for progressive summaries
- Template-based narrative generation
- Fallback handling for API failures
- Optimized for speed and cost

## 🚀 Integration & Usage

### Basic Setup
```python
from interfaces.slack import SlackInterface, create_slack_app

# For direct usage
slack_interface = SlackInterface()

# For FastAPI integration
app_handler = create_slack_app()
```

### Environment Variables
Required environment variables:
```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
GEMINI_API_KEY=your-gemini-key  # Optional, for tool summarization
DATABASE_URL=postgresql://...   # Optional, for PostgreSQL (Railway deployment)
```

### Event Flow

1. **Message Received** → `core_slack_orchestration.py`
2. **Mention Parsing** → `formatters/mention_resolver.py`
3. **Agent Processing** → External agent system
4. **Streaming Updates** → `handlers/streaming_handler.py`
5. **Response Formatting** → `formatters/markdown_parser.py`
6. **Final Response** → Back to Slack

### Modal Interaction Flow

1. **Button Click** → `core_slack_orchestration.py`
2. **Database Lookup** → `services/cache_service.py` → **PostgreSQL/SQLite**
3. **Modal Creation** → `handlers/modal_handler.py`
4. **Page Navigation** → Modal update cycle

### Streaming Architecture

1. **Initial Message** → "🔄 Getting weather for London..."
2. **Progressive Updates** → Real-time Gemini summary chunks
3. **Final Display** → Complete formatted response with "view flow" button

## 🔧 Development & Extension

### Adding New Features

New features should go in the `features/` directory:
```
features/
├── app_home/          # Home tab functionality
├── shortcuts/         # Global shortcuts
├── slash_commands/    # /agent commands
├── workflows/         # Workflow Builder
└── scheduled/         # Time-based features
```

### Testing Individual Components

Each component can be tested independently:
```python
# Test cache service
from interfaces.slack.services import SlackCacheService
cache = SlackCacheService()
cache.store_execution_details("test", [("thinking", "test")])

# Test markdown parser
from interfaces.slack.formatters import MarkdownToSlackParser
parser = MarkdownToSlackParser()
blocks = parser.parse_to_blocks("# Test")
```

### Performance Considerations

- **Persistent Storage**: Execution details stored in database (never expire)
- **Caching**: All user/channel lookups are cached in-memory
- **Async Operations**: Non-blocking throughout
- **Parallel API Calls**: Independent operations run concurrently (saves 200-400ms)
- **Batch Operations**: Multiple Slack API calls batched when possible
- **Size Limits**: Automatic handling of Slack's size constraints
- **Rate Limiting**: Integrated with global rate limiting system
- **Database Cleanup**: Automatic expiration and cleanup every 12 hours

### ⚡ Performance Optimizations

The interface includes several performance optimizations:

#### Parallel API Operations
```python
# OPTIMIZED: Concurrent API calls (saves 200-400ms per message)
mention_task = asyncio.create_task(mention_resolver.parse_slack_mentions(...))
user_info_task = asyncio.create_task(get_user_info(...))

# Wait for both to complete in parallel
(message_text, mention_context), user_info = await asyncio.gather(
    mention_task, user_info_task
)
```

#### Timing Metrics
All operations are timed with millisecond precision:
```bash
SLACK-TIMING: parallel API calls took 150 ms user=U123
SLACK-TIMING: start_streaming took 89 ms user=U123  
SLACK-TIMING: agent.process_request took 3420 ms user=U123
SLACK-TIMING: finish_with_response took 156 ms user=U123
SLACK-TIMING: end-to-end for user=U123 ms=3815
```

#### Modal Optimizations
- **Fast retry logic**: 50ms → 75ms → 112ms (capped at 150ms)
- **Trigger ID efficiency**: No artificial delays
- **Size validation**: Pre-check before sending to avoid failures

## 📊 Component Dependencies

```mermaid
graph TD
    A[core_slack_orchestration.py] --> B[services/cache_service.py]
    A --> C[services/user_service.py]
    A --> D[handlers/streaming_handler.py]
    A --> E[handlers/modal_handler.py]
    A --> F[formatters/mention_resolver.py]
    
    D --> G[formatters/markdown_parser.py]
    D --> H[agents/execution_summarizer.py]
    
    E --> B
    F --> B
    F --> C
    
    B --> I[External: Slack API]
    B --> K[Database: PostgreSQL/SQLite]
    C --> I
    C --> K
    H --> J[External: Gemini API]
```

## 🔍 Debugging & Monitoring

### Logging
Each component uses structured logging:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Component operation", extra={"component": "cache_service"})
```

### Cache & Database Monitoring
```python
stats = await cache_service.get_cache_stats()
print(f"Storage: {stats['execution_details_storage']}")  # "database"
print(f"User cache: {stats['user_cache_count']}")
print(f"Channel cache: {stats['channel_cache_count']}")

# Database cleanup monitoring
deleted_count = await cache_service._cleanup_database_periodically()
print(f"Cleaned up {deleted_count} expired execution details")
```

### Performance Metrics
- Message processing time (logged with SLACK-TIMING prefix)
- Cache hit rates (DB-HIT for database, Cache-HIT for memory)
- Modal open success rates
- API call frequencies
- Database cleanup statistics (every 12 hours)

## 🎯 Recent Architecture Improvements

### Database Integration (v2.0)
- **Persistent Execution Details**: All "view flow" button data stored in PostgreSQL/SQLite
- **7-Day Auto-Expiration**: Automatic cleanup with database-level TTL
- **Railway Deployment Ready**: Full PostgreSQL compatibility for production
- **Development Flexibility**: SQLite fallback for local development

### Streaming Optimization (v2.0)  
- **Two-Phase Tool Execution**: Immediate "working" message + progressive Gemini summaries
- **Real-Time Chunk Streaming**: `tool_summary_chunk` content type for live updates
- **Centralized Summarization**: All tool operations get consistent LLM-based summaries
- **UI Simplification**: Slack interface focuses purely on formatting (no LLM logic)

### Performance Enhancements (v2.0)
- **Parallel API Calls**: Mention parsing + user info fetching (saves 200-400ms)
- **Optimized Modal Retries**: 50ms → 75ms → 112ms (was 100ms → 150ms → 300ms)
- **Database Cleanup**: Automatic background cleanup every 12 hours
- **Reduced Memory Usage**: In-memory caches only for user/channel lookups

## 🚀 Future Enhancements

The modular architecture supports easy addition of:
- **App Home Dashboard**: User-specific views
- **Slash Commands**: `/agent help`, `/agent status`
- **Shortcuts**: Quick actions from any message
- **Workflows**: Integration with Slack Workflow Builder
- **Scheduled Tasks**: Proactive notifications
- **Rich Interactions**: Forms, dropdowns, date pickers

This architecture ensures the Slack interface remains maintainable and scalable as new features are added.
