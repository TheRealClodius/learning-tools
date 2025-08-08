# Learning Tools

An AI agent system with dynamic tool discovery and multi-interface support.

## Features

- 🤖 **Intelligent Agent**: Claude-powered agent with adaptive problem-solving
- 🔧 **Dynamic Tool Discovery**: Registry-based tool system with runtime discovery

- 🌤️ **Weather Tools**: Current weather and forecasts via OpenWeather API
- 🔍 **Web Search**: Research and search capabilities via Perplexity API
- 💬 **Multiple Interfaces**: CLI, API, and Slack interfaces

## Architecture

### Core Components

- **`agents/`**: AI agent implementations with system prompts
- **`tools/`**: Individual tool implementations (weather, perplexity, registry)
- **`schemas/`**: JSON schemas for tool inputs/outputs with LLM-optimized flat structure (see [`schemas/README.md`](./schemas/README.md))
- **`runtime/`**: Tool executor for dynamic loading and execution
- **`interfaces/`**: CLI, API, and Slack interface implementations

### Tool System

Tools are dynamically discovered and executed:
1. **Registry Tools**: Search, describe, list, and categorize available tools

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
# Install dependencies using requirements.txt
pip install -r requirements.txt
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

├── schemas/
│   ├── services/
│   │   ├── weather/                # Weather tool schemas
│   │   ├── perplexity/            # Perplexity tool schemas

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

### v2.0 - LLM Optimization (Latest)

**🚀 Major Architecture Improvements:**
- **LLM-Optimized Flat Structure**: Improved AI processing with flat parameter schemas
- **Performance Boost**: Enhanced validation logic and server response parsing

**🎯 Technical Improvements:**
- Flat parameter structure for better LLM processing
- Fixed validation logic for empty arrays and boolean fields
- Enhanced server response parsing with fallback handling
- Comprehensive schema documentation with usage examples

**📚 Documentation:**
- Comprehensive schema documentation in [`schemas/README.md`](./schemas/README.md)
- Complete tool integration guide

## License

MIT License - see LICENSE file for details. 