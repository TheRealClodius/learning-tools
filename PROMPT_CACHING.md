# Prompt Caching Implementation

This document describes the comprehensive prompt caching system implemented to reduce costs and improve performance for Claude API requests.

## Overview

The prompt caching system provides intelligent caching for Claude API prompts with support for:

- **System prompt caching** with Anthropic's prompt caching features
- **Message history caching** to avoid re-sending conversation context
- **Tool schema caching** for frequently used tool definitions
- **Dual storage**: In-memory caching with optional Redis persistence
- **Automatic expiration** with configurable TTL values
- **Performance metrics** and cache statistics
- **Rate limit protection** by reducing redundant API calls

## Key Features

### 1. Multi-Level Caching
- **System Prompts**: Long-lived caching (24 hours) for stable system instructions
- **Messages**: Medium-term caching (1 hour) for conversation context
- **Tools**: Long-lived caching (24 hours) for tool schemas
- **General Cache**: Configurable TTL for any content

### 2. Anthropic Integration
- Native support for Anthropic's prompt caching using `cache_control` directives
- Automatic formatting of system prompts for optimal caching
- Ephemeral cache control for cost-effective prompt reuse

### 3. Intelligent Storage
- **Memory-first**: Fast L1 cache for immediate access
- **Redis fallback**: Persistent L2 cache for session continuity
- **Automatic cleanup**: LRU eviction and TTL-based expiration
- **Graceful degradation**: Falls back to memory-only if Redis unavailable

### 4. Performance Monitoring
- Hit/miss ratios and performance metrics
- Token usage tracking and cost savings estimation
- Cache size monitoring and memory management
- Detailed logging for debugging and optimization

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ClientAgent   │    │   PromptCache    │    │   Redis Cache   │
│                 │    │                  │    │   (Optional)    │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • process()     │───▶│ • System prompts │───▶│ • Persistent    │
│ • claude_call() │    │ • Messages       │    │ • Cross-session │
│ • get_stats()   │    │ • Tools          │    │ • Shared        │
│ • clear_cache() │    │ • Statistics     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Memory Cache    │
                       │                  │
                       │ • Fast access    │
                       │ • LRU eviction   │
                       │ • TTL expiration │
                       └──────────────────┘
```

## Usage

### Basic Integration

The prompt caching is automatically enabled in the `ClientAgent` class:

```python
from agents.client_agent import ClientAgent

# Initialize with caching enabled (default)
agent = ClientAgent(enable_caching=True)

# Process requests - caching happens automatically
response = await agent.process("Your request here", user_id="user123")

# Get cache statistics
stats = agent.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
```

### Advanced Configuration

For custom caching configuration:

```python
from runtime.prompt_cache import PromptCache

cache = PromptCache(
    redis_url="redis://localhost:6379",  # Optional Redis URL
    max_memory_entries=1000,             # Memory cache size
    default_ttl=3600,                   # 1 hour default TTL
    system_prompt_ttl=86400,            # 24 hours for system prompts
    enable_anthropic_caching=True       # Enable Anthropic prompt caching
)

# Use cache directly
cached_prompt, from_cache = await cache.get_cached_system_prompt(
    "You are a helpful AI assistant."
)
```

### Environment Configuration

Set environment variables for configuration:

```bash
# Optional Redis configuration
export REDIS_URL="redis://localhost:6379"

# Required for Claude API
export ANTHROPIC_API_KEY="your-api-key"
```

## Cache Management

### Getting Statistics

```python
# Get comprehensive cache statistics
stats = agent.get_cache_stats()

# Example output:
{
    "hits": 45,
    "misses": 12,
    "hit_rate": "78.9%",
    "system_prompt_hits": 15,
    "system_prompt_misses": 3,
    "message_cache_hits": 20,
    "message_cache_misses": 5,
    "total_tokens_saved": 15420,
    "memory_cache_size": 23,
    "redis_available": True,
    "anthropic_caching_enabled": True,
    "max_memory_entries": 1000,
    "default_ttl": 3600,
    "system_prompt_ttl": 86400
}
```

### Clearing Cache

```python
# Clear all cache entries
await agent.clear_cache()

# Clear entries matching a pattern
await agent.clear_cache(pattern="sys_prompt")

# Toggle caching on/off
agent.toggle_caching(enable=False)  # Disable
agent.toggle_caching(enable=True)   # Re-enable
```

## Performance Benefits

### Cost Reduction
- **System Prompts**: Up to 90% cost reduction for repeated system instructions
- **Message Context**: 50-70% cost reduction for conversation continuity
- **Tool Schemas**: Near 100% cost reduction for stable tool definitions

### Latency Improvement
- **Memory Cache**: Sub-millisecond retrieval times
- **Redis Cache**: 10-50ms retrieval times
- **API Avoidance**: No network latency for cached content

### Rate Limit Protection
- Reduces API call frequency by reusing cached content
- Helps prevent 429 rate limit errors during high usage
- Enables more efficient batch processing

## Anthropic Prompt Caching

The implementation leverages Anthropic's native prompt caching features:

```python
# Automatic cache control formatting
request_params = {
    "model": "claude-3-5-sonnet-20241022",
    "system": [
        {
            "type": "text",
            "text": "Your system prompt here",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": [...],
    "tools": [...]
}
```

### Benefits of Anthropic Caching
- **Server-side caching**: Reduces token processing costs
- **Persistent across requests**: Cache survives multiple API calls
- **Automatic optimization**: Anthropic handles cache management
- **Cost transparency**: Clear billing for cached vs. uncached tokens

## Configuration Options

### PromptCache Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `redis_url` | `None` | Redis connection URL (optional) |
| `max_memory_entries` | `1000` | Maximum memory cache entries |
| `default_ttl` | `3600` | Default TTL in seconds (1 hour) |
| `system_prompt_ttl` | `86400` | System prompt TTL (24 hours) |
| `enable_anthropic_caching` | `True` | Enable Anthropic prompt caching |

### ClientAgent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_caching` | `True` | Enable/disable prompt caching |
| `api_key` | `env:ANTHROPIC_API_KEY` | Anthropic API key |

## Monitoring and Debugging

### Log Messages

The caching system provides detailed logging:

```
INFO: PromptCache initialized with Redis: True, Memory limit: 1000, Default TTL: 3600s
INFO: CACHE-HIT: System=True, Messages=False, Tools=True
INFO: CACHE-STATS: {'hits': 45, 'misses': 12, 'hit_rate': '78.9%', ...}
DEBUG: System prompt cache hit: a1b2c3d4...
DEBUG: Messages cached: e5f6g7h8...
```

### Testing and Validation

Run the validation script to test cache functionality:

```bash
# Basic validation (no external dependencies)
python3 validate_cache_implementation.py

# Full testing (requires API key)
python3 test_prompt_cache.py
```

## Best Practices

### 1. System Prompt Design
- Keep system prompts stable for better cache hits
- Avoid dynamic content in system prompts
- Use consistent formatting and spacing

### 2. Message Management
- Structure conversations consistently
- Avoid unnecessary message variations
- Use tool calls efficiently to leverage tool caching

### 3. Cache Monitoring
- Monitor hit rates regularly
- Adjust TTL values based on usage patterns
- Clear cache when system prompts change significantly

### 4. Redis Configuration
- Use Redis for production environments
- Configure appropriate memory limits
- Monitor Redis memory usage

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   - Check for dynamic content in prompts
   - Verify consistent message formatting
   - Review TTL settings

2. **Redis Connection Errors**
   - Verify Redis URL and connectivity
   - Check Redis server status
   - Falls back to memory-only mode

3. **Memory Usage**
   - Adjust `max_memory_entries` setting
   - Monitor cache cleanup frequency
   - Consider shorter TTL values

### Error Handling

The cache system includes robust error handling:
- Graceful fallback to uncached requests
- Redis connection error recovery
- Memory limit enforcement
- TTL expiration cleanup

## Performance Metrics

Track these key metrics for optimization:

- **Hit Rate**: Target 70%+ for optimal performance
- **Token Savings**: Monitor cost reduction
- **Memory Usage**: Keep under system limits
- **Response Time**: Measure cache vs. API latency

## Security Considerations

- **API Key Protection**: Never cache API keys
- **Data Sensitivity**: Review cached content policies
- **Redis Security**: Use authentication and encryption
- **TTL Management**: Don't cache sensitive data too long

## Future Enhancements

Planned improvements include:
- Semantic similarity caching for near-duplicate prompts
- Compression for large cached entries
- Cache warming strategies
- Advanced eviction policies
- Cross-user cache sharing (with privacy controls)

---

For questions or issues with the prompt caching system, check the logs first, then review this documentation for configuration options and troubleshooting steps.