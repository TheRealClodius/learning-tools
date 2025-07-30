# Learning Tools

An AI agent system with dynamic tool discovery, conversation memory, and multi-interface support.

## Features

- 🤖 **Intelligent Agent**: Claude-powered agent with adaptive problem-solving
- 🔧 **Dynamic Tool Discovery**: Registry-based tool system with runtime discovery
- 🧠 **Conversation Memory**: MemoryOS integration for persistent conversation history
- 🌤️ **Weather Tools**: Current weather and forecasts via OpenWeather API
- 🔍 **Web Search**: Research and search capabilities via Perplexity API
- 💬 **Multiple Interfaces**: CLI, API, and Slack interfaces

## Architecture

### Core Components

- **`agents/`**: AI agent implementations with system prompts
- **`tools/`**: Individual tool implementations (weather, memory, perplexity, registry)
- **`schemas/`**: JSON schemas for tool inputs/outputs and validation
- **`runtime/`**: Tool executor for dynamic loading and execution
- **`interfaces/`**: CLI, API, and Slack interface implementations

### Tool System

Tools are dynamically discovered and executed:
1. **Registry Tools**: Search, describe, list, and categorize available tools
2. **Memory Tools**: Store and retrieve conversation context using MemoryOS
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
- Automatically stores important conversations
- Retrieves relevant context for follow-up questions
- Builds user profiles over time

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
│   └── memory.py                   # MemoryOS conversation memory
├── schemas/
│   ├── services/
│   │   ├── weather/                # Weather tool schemas
│   │   ├── perplexity/            # Perplexity tool schemas
│   │   ├── memory/                # Memory tool schemas
│   │   └── registry/              # Registry tool schemas
│   └── shared/                    # Common schemas and patterns
├── runtime/
│   └── tool_executor.py           # Dynamic tool loading and execution
├── interfaces/
│   ├── cli_interface.py           # Command-line interface
│   ├── api_interface.py           # REST API interface
│   └── slack_interface.py         # Slack bot interface
└── execute.py                     # Pydantic models for tool validation
```

## Memory System

The agent uses MemoryOS for conversation memory:
- **Short-term**: Recent conversation context
- **Mid-term**: Important conversation segments
- **Long-term**: User profiles and knowledge extraction

Memory is stored locally in `./memory_data/` and persists across sessions.

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

## License

MIT License - see LICENSE file for details. 