# Learning Tools

An AI agent system with dynamic tool discovery, conversation memory, and multi-interface support.

## Features

- 🤖 **Intelligent Agent**: Claude-powered agent with adaptive problem-solving
- 🔧 **Dynamic Tool Discovery**: Registry-based tool system with runtime discovery
- 🧠 **Dual Memory System**: Revolutionary conversation + execution memory with message linking
- 🌤️ **Weather Tools**: Current weather and forecasts via OpenWeather API
- 🔍 **Web Search**: Research and search capabilities via Perplexity API
- 💬 **Multiple Interfaces**: CLI, API, and Slack interfaces

## Architecture

### Core Components

- **`agents/`**: AI agent implementations with system prompts
- **`tools/`**: Individual tool implementations (weather, memory, perplexity, registry)
- **`schemas/`**: JSON schemas for tool inputs/outputs with LLM-optimized flat structure (see [`schemas/README.md`](./schemas/README.md))
- **`runtime/`**: Tool executor for dynamic loading and execution
- **`interfaces/`**: CLI, API, and Slack interface implementations

### Tool System

Tools are dynamically discovered and executed:
1. **Registry Tools**: Search, describe, list, and categorize available tools
2. **Memory Tools**: Dual memory system (conversation + execution) with MemoryOS MCP server
3. **External Tools**: Weather, web search, and other API integrations

## Setup

### 1. Environment Variables

Create `.env.local` with your API keys:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional
OPENWEATHER_API_KEY=your_openweather_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### 2. Install Dependencies

```bash
# Install MemoryOS dependencies (requires Python 3.10+)
pip install numpy==1.24.* sentence-transformers==5.0.0 transformers>=4.51.0
pip install FlagEmbedding>=1.2.9 faiss-cpu>=1.7.0 httpx openai
pip install flask>=2.0.0 python-dotenv>=0.19.0 typing-extensions>=4.0.0

# Install other dependencies
pip install anthropic pydantic python-dotenv pyyaml httpx asyncio
```

### 3. Set Python Path

```bash
export PYTHONPATH="/path/to/learning tools :$PYTHONPATH"
```

## Usage

### CLI Interface

**Interactive mode:**
```bash
python3 interfaces/cli_interface.py --interactive
```

**Single command:**
```bash
python3 interfaces/cli_interface.py "What's the weather in Paris?"
```

### API Interface

```bash
python3 interfaces/api_interface.py
```

### Slack Interface

Configure Slack tokens and run:
```bash
python3 interfaces/slack_interface.py
```

## Tool Usage Examples

### Weather Tools
```bash
# Current weather
"What's the current weather in London?"

# Weather forecast  
"Give me a 5-day forecast for Tokyo"
```

### Web Search Tools
```bash
# General search
"Search for the latest news about quantum computing"

# Research query
"Research sustainable energy technologies and their current adoption rates"
```

### Memory Tools
```bash
# Store conversation memory
"Remember this conversation about Python optimization techniques"

# Query conversation history  
"What did we discuss about API integration last week?"

# Learn from execution patterns
"Show me how I've handled database errors in the past"

# Cross-memory insights
"What was the execution trail for our discussion about machine learning?"
```

**Dual Memory Features:**
- **Conversation Memory**: Stores Q&A pairs with context and metadata
- **Execution Memory**: Tracks tools used, errors, and problem-solving approaches  
- **Message Linking**: Cross-reference conversations with their execution details
- **Pattern Learning**: Analyze past successes and failures for better problem-solving

## Project Structure

```
learning tools/
├── agents/
│   ├── client_agent.py              # Main Claude-powered agent
│   ├── system prompts/
│   │   └── client_agent_sys_prompt.yaml  # Agent configuration
│   └── specialized agents...
├── tools/
│   ├── registry.py                  # Tool discovery and metadata
│   ├── weather.py                   # OpenWeather API integration
│   ├── perplexity.py               # Perplexity API integration
│   └── memory.py                   # MemoryOS dual memory system (conversation + execution)
├── schemas/
│   ├── services/
│   │   ├── weather/                # Weather tool schemas
│   │   ├── perplexity/            # Perplexity tool schemas
│   │   ├── memory/                # Dual memory schemas (conversation + execution)
│   │   └── registry/              # Registry tool schemas
│   └── shared/                    # Common schemas and patterns
│   └── README.md                  # Complete schema architecture documentation
├── runtime/
│   └── tool_executor.py           # Dynamic tool loading and execution
├── interfaces/
│   ├── cli_interface.py           # Command-line interface
│   ├── api_interface.py           # REST API interface
│   └── slack_interface.py         # Slack bot interface
└── execute.py                     # Pydantic models for tool validation
```

## Memory System

The agent uses **MemoryOS MCP Server** with a revolutionary **dual memory architecture** for comprehensive conversation and execution tracking:

### 🎯 Dual Memory Architecture

**Two Memory Types for Complete Context:**

#### 1. **Conversation Memory**
- Stores user prompts and agent responses
- Builds persistent dialogue history
- Enables contextual follow-up questions
- Links to execution details via `message_id`

#### 2. **Execution Memory** 
- Tracks tools used, errors encountered, reasoning approaches
- Captures problem-solving strategies and observations
- Enables learning from past execution patterns
- Links to conversations for complete task understanding

### 🚀 Technical Architecture
- **MCP Client**: Lightweight HTTP client with flat parameter structure (this project)
- **MCP Server**: Remote MemoryOS server with GPU acceleration
- **Performance**: ~50ms response time vs 15+ seconds with direct integration
- **LLM-Optimized**: Flat parameter structure for better AI processing

### 🗂️ Memory Tiers
- **Short-term**: Recent conversation context (Redis-style)
- **Mid-term**: Important conversation segments with heat tracking  
- **Long-term**: User profiles and knowledge extraction
- **Execution Patterns**: Searchable tool usage and error patterns

### ⚙️ Setup
1. **Deploy MemoryOS Server**: [MemoryOS-Private-Server-AP-Expert](https://github.com/TheRealClodius/MemoryOS-Private-Server-AP-Expert)
2. **Configure Environment**:
   ```bash
   MEMORYOS_SERVER_URL=http://localhost:5000
   MEMORYOS_API_KEY=your_api_key_here
   MEMORY_USER_ID=your_user_id
   ```
3. **Test Connection**: Memory functions automatically validate on first use

### 💡 Dual Memory Usage

**Agent-Driven Storage**: The AI agent intelligently chooses when to store conversation and execution data using memory functions
```python
# Conversation Memory - stored via memory_conversation_add
{
  "message_id": "unique_identifier",
  "user_input": "User's question",
  "agent_response": "Agent's response",
  "meta_data": {"platform": "cli", "importance": "high"}
}

# Execution Memory - stored via memory_execution_add (Flat Structure for LLM)
{
  "message_id": "unique_identifier", 
  "execution_summary": "High-level task summary",
  "tools_used": ["weather_api", "search_tool"],
  "errors": [{"error_type": "timeout", "tool": "weather_api"}],
  "observations": "Reasoning and strategy insights",
  "success": true,
  "duration_ms": 250
}
```

**How It Works**:
- **Agent Decision**: The AI agent decides when memory storage would be valuable
- **Background AI Generation**: Gemini 2.5 Flash creates execution summaries and observations in the background
- **Agent Storage**: Once summaries are generated, agent calls memory_execution_add to store the data
- **Strategic Storage**: Agent stores important conversations and complex problem-solving sessions

**Available Memory Functions**:
- `memory_conversation_add`: Store Q&A pairs for future retrieval
- `memory_execution_add`: Store execution details (tools, reasoning, outcomes)
- `memory_conversation_retrieve`: Retrieve conversation history with semantic/recent search
- `memory_execution_retrieve`: Retrieve execution patterns and tool usage history
- `memory_get_profile`: Get user preferences and context from memory

**Cross-Memory Queries**:
- *"What was the execution trail for this conversation?"*
- *"Show me similar tool usage patterns"*
- *"What errors have I encountered with weather tools?"*

### 🎯 Schema Architecture

**Flat Structure Benefits:**
- ✅ **LLM-Friendly**: Flat parameters are easier for AI models to process
- ✅ **Performance**: Reduced nesting improves parsing speed
- ✅ **Validation**: Simpler field validation and error handling
- ✅ **Debugging**: Clear parameter visibility for troubleshooting

**Complete Schema Documentation**: See [`schemas/README.md`](./schemas/README.md)

### 🚀 Key Benefits
- ⚡ **300x faster response times** (15s → 50ms)
- 🧠 **Dual memory system** for comprehensive context
- 🎯 **Message linking** between conversation and execution
- 🚀 **GPU acceleration** for embeddings
- 📈 **Multi-user scalability** with user isolation
- 💾 **Persistent embedding cache**
- 🔍 **Execution pattern learning** and error analysis
- 🎨 **LLM-optimized flat structure** for better AI processing
- 🔗 **Cross-memory insights** linking conversation to execution details

### 🛠️ Advanced Features

**Smart Retrieval**: 
- Query conversation history: *"What did we discuss about Python?"*
- Learn from execution patterns: *"How have I handled API errors before?"*
- Time-filtered searches: *"Show conversations from last week"*

**Memory Management Features**:
- Agent-controlled storage of conversations and executions
- Heat-based promotion from short to mid-term memory (MemoryOS server-side)
- Intelligent memory processing and summarization (MemoryOS server-side)  
- Execution pattern analysis for improved problem-solving

## Tool Development

To add new tools:

1. **Create tool implementation** in `tools/your_tool.py`
2. **Define JSON schemas** in `schemas/services/your_tool/`
3. **Add Pydantic models** in `execute.py`
4. **Register in tool executor** in `runtime/tool_executor.py`

Tools are automatically discovered through the schema directory structure.

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive JSON schemas for all tool operations
3. Include proper error handling and logging
4. Test with the CLI interface before committing

## Recent Updates 🎉

### v2.0 - Dual Memory System & LLM Optimization (Latest)

**🚀 Major Architecture Improvements:**
- **Dual Memory System**: Revolutionary separation of conversation and execution memory
- **LLM-Optimized Flat Structure**: Improved AI processing with flat parameter schemas
- **Message Linking**: Connect conversations to their execution details via `message_id`
- **Client-Server Alignment**: Perfect synchronization between MCP client and server
- **Performance Boost**: Enhanced validation logic and server response parsing

**🧠 Memory Features:**
- **Conversation Memory**: Store user prompts and agent responses with context
- **Execution Memory**: Track tools used, errors, reasoning, and problem-solving strategies
- **Cross-Memory Queries**: Link conversation context with execution patterns
- **Pattern Learning**: Learn from past successes and failures for better problem-solving

**🎯 Technical Improvements:**
- Flat parameter structure for better LLM processing
- Fixed validation logic for empty arrays and boolean fields
- Enhanced server response parsing with fallback handling
- Comprehensive schema documentation with usage examples
- Updated Pydantic models for dual memory system

**📚 Documentation:**
- Comprehensive schema documentation in [`schemas/README.md`](./schemas/README.md)
- Updated memory usage examples with dual system
- Complete client-server integration guide

## License

MIT License - see LICENSE file for details. 