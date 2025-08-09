# Rate Limiting Guide

This guide explains how the rate limiting system works and how to configure it for smooth user experience when dealing with API rate limits.

## Overview

The rate limiting system provides:
- **Automatic retry with exponential backoff** - Retries failed requests with increasing delays
- **Token bucket algorithm** - Prevents burst requests that exceed API limits
- **Request queuing** - Queues requests when rate limits are reached
- **User-friendly error messages** - Shows helpful messages instead of technical errors

## How It Works

### 1. Token Bucket Algorithm
The system uses a token bucket to track API usage:
- Tokens are consumed when making API calls
- Tokens refill at a constant rate (based on your API tier)
- Requests are queued when tokens are exhausted

### 2. Automatic Retry Logic
When a rate limit error occurs:
1. First retry after 2 seconds
2. Second retry after 4 seconds (2x backoff)
3. Third retry after 8 seconds (2x backoff)
4. Maximum delay capped at 60 seconds

### 3. User Experience
Users see friendly messages instead of technical errors:
- **First retry**: "🔄 I'm experiencing high demand right now. Let me try again in just a moment..."
- **Subsequent retries**: "⏳ Still working on your request (attempt 2). The service is quite busy, but I'll keep trying..."
- **Final failure**: "😔 I'm sorry, but the service is currently overloaded..."

## Configuration

### Setting Your API Tier

Set the `ANTHROPIC_TIER` environment variable based on your plan:

```bash
# In your .env.local file
ANTHROPIC_TIER=build_1  # Options: free, build_1, build_2, build_3, build_4, scale
```

### Tier Limits

| Tier | Tokens/Minute | Burst Size |
|------|---------------|------------|
| Free | 10,000 | 2,000 |
| Build 1 | 50,000 | 10,000 |
| Build 2 | 100,000 | 20,000 |
| Build 3 | 200,000 | 40,000 |
| Build 4 | 400,000 | 80,000 |
| Scale | 800,000 | 160,000 |

### Custom Configuration

Edit `config/rate_limits.yaml` to customize settings:

```yaml
default:
  max_retries: 3          # Number of retry attempts
  initial_delay: 2.0      # First retry delay (seconds)
  max_delay: 60.0         # Maximum retry delay
  backoff_factor: 2.0     # Delay multiplier for each retry
  tokens_per_minute: 40000
  burst_size: 8000
```

## Best Practices

### 1. Break Down Large Requests
If you're hitting rate limits frequently:
- Split large documents into smaller chunks
- Process requests in batches
- Use more concise prompts

### 2. Monitor Usage
Watch for rate limit warnings in logs:
```
WARNING: Rate limit hit, retrying in 2s (attempt 1/3)
```

### 3. Adjust Token Estimates
The system estimates ~4 characters per token. For more accurate estimates:
- Monitor actual token usage in Anthropic dashboard
- Adjust the estimation logic if needed

### 4. Handle Peak Times
During high-demand periods:
- Requests are automatically queued
- Users see status updates
- System retries with backoff

## Troubleshooting

### Common Issues

1. **"Service is currently overloaded" errors**
   - Check your API tier matches your plan
   - Consider upgrading if hitting limits frequently
   - Break down large requests

2. **Requests timing out in queue**
   - Default queue timeout is 5 minutes
   - Adjust `queue_timeout` in configuration
   - Monitor queue size in logs

3. **Incorrect token estimates**
   - System uses rough estimate (4 chars = 1 token)
   - Actual usage varies by content type
   - Monitor and adjust if needed

### Debug Mode

Enable verbose logging to see rate limit details:

```bash
# CLI
python -m interfaces.cli_interface -v

# Environment variable
export LOG_LEVEL=DEBUG
```

## Integration Examples

### Slack Interface
```python
try:
    response = await agent.process_request(message)
except RateLimitError as e:
    # User sees friendly message automatically
    await send_message(str(e))
```

### API Interface
```python
@app.post("/chat")
async def chat(message: str):
    try:
        return await agent.process_request(message)
    except RateLimitError as e:
        # Returns 429 status with friendly message
        raise HTTPException(status_code=429, detail=str(e))
```

### CLI Interface
```python
try:
    response = await agent.process_request(command)
except RateLimitError as e:
    # Shows friendly message in terminal
    print(str(e))
```

## Monitoring

### Key Metrics to Track
- Rate limit hits per hour
- Average retry count
- Queue depth
- Token consumption rate

### Log Analysis
```bash
# Count rate limit hits
grep "Rate limit hit" app.log | wc -l

# View retry patterns
grep "retrying in" app.log

# Monitor queue status
grep "queue" app.log | grep -i rate
```

## Future Improvements

Potential enhancements to consider:
1. **Predictive rate limiting** - Slow down before hitting limits
2. **User-based quotas** - Different limits per user/team
3. **Priority queuing** - Premium users get faster processing
4. **Distributed rate limiting** - Share limits across multiple instances