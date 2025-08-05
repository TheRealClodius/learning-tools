# Tool Schemas

This directory contains JSON schemas that define the tool system API interfaces for consistent data validation.

## Architecture Overview

The schemas are organized into **input** and **output** definitions for each tool operation, following a clear client-server contract model.

## Schema Organization

### Service-Based Structure
```
schemas/
├── services/
│   ├── weather/          # Weather tool schemas
│   ├── perplexity/       # Perplexity search tool schemas
│   └── registry/         # Tool registry schemas
└── shared/               # Common schemas and patterns
```

### Input Schemas (Client → Tools)
These define what the client sends to each tool:

| Service | Schema | Purpose |
|---------|--------|---------|
| Weather | `current_input.json` | Get current weather data |
| Weather | `forecast_input.json` | Get weather forecasts |
| Weather | `search_input.json` | Search weather locations |
| Perplexity | `search_input.json` | Web search queries |
| Perplexity | `research_input.json` | Research requests |
| Registry | `search_input.json` | Tool discovery |
| Registry | `describe_input.json` | Tool descriptions |

### Output Schemas (Tools → Client) 
These serve as **API contracts** defining what each tool should return:

| Service | Schema | Purpose |
|---------|--------|---------|
| Weather | `current_output.json` | Weather data response |
| Weather | `forecast_output.json` | Forecast data response |
| Perplexity | `search_output.json` | Search results response |
| Registry | `list_output.json` | Available tools response |

## Schema Benefits

**Flat Structure Advantages:**
- ✅ **LLM-Friendly**: Flat parameters are easier for AI models to process
- ✅ **Performance**: Reduced nesting improves parsing speed
- ✅ **Validation**: Simpler field validation and error handling
- ✅ **Debugging**: Clear parameter visibility for troubleshooting

## Usage

Each tool implementation references these schemas for:
1. **Input Validation**: Ensuring correct parameters are provided
2. **Output Formatting**: Standardizing response structures
3. **Documentation**: Clear API contracts for development
4. **Testing**: Validation of tool behavior

See individual service directories for specific schema documentation.