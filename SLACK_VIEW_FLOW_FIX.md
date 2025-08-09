# Slack "View Flow" Button Fix

## Problem Description
Users were experiencing intermittent failures when clicking the "view flow" button in Slack:
- Button shows loading state but modal doesn't open
- Sometimes works after 2-3 clicks, sometimes not at all
- Inconsistent behavior across different users/times

## Root Cause Analysis
1. **Synchronous Modal Opening**: The button handler was changed from asynchronous to synchronous execution, which could cause timing issues
2. **No Retry Logic**: If the modal failed to open, there was no retry mechanism
3. **Limited Error Handling**: Errors were logged but not properly handled or communicated to users
4. **Cache Management**: No TTL on cached execution details could lead to stale data or memory issues
5. **Insufficient Logging**: Hard to diagnose issues without detailed logging

## Implemented Solutions

### 1. Enhanced Error Handling and Retry Logic
```python
# Added retry mechanism with exponential backoff
max_retries = 3
retry_delay = 0.5

for attempt in range(max_retries):
    try:
        await self._show_execution_details_modal(body, client)
        logger.info(f"Successfully opened execution details modal on attempt {attempt + 1}")
        break
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed to open modal: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
```

### 2. Improved Logging
- Added detailed logging at each step of the modal opening process
- Log trigger_id, user_id, message_ts for debugging
- Log modal size and Slack API responses
- Include stack traces for exceptions

### 3. Cache Management with TTL
- Added timestamps to cache entries
- Implemented 1-hour TTL for execution details
- Background task to clean up expired entries
- Validation to check for stale cache entries

### 4. Better User Feedback
- If ack fails, don't attempt to open modal
- Show ephemeral message if modal fails after retries
- Inform users if execution details are expired

### 5. Small Delay After Ack
- Added 0.1s delay after acknowledgment to ensure Slack has processed it
- Helps prevent race conditions with trigger_id

## Testing Recommendations
1. Test with high load to simulate timing issues
2. Test with large execution details to check modal size limits
3. Test cache expiration after 1 hour
4. Monitor logs for any new error patterns
5. Test retry mechanism by simulating network failures

## Monitoring
Key metrics to monitor:
- Button click to modal open latency
- Retry attempt counts and success rates
- Cache hit/miss rates
- Error rates by type (trigger_id, cache miss, API errors)

## Future Improvements
1. Consider implementing a persistent storage for execution details instead of in-memory cache
2. Add metrics/monitoring for button performance
3. Consider implementing a queue system for modal operations during high load
4. Add user preferences for execution detail retention time