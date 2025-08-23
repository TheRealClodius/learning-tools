# Signal Troubleshooting Guide

This guide helps diagnose and resolve common issues in the Signal system.

## Table of Contents
- [Search Issues](#search-issues)
- [Token Management Issues](#token-management-issues)
- [Performance Issues](#performance-issues)
- [Evidence Tracking Issues](#evidence-tracking-issues)
- [Guardrail Issues](#guardrail-issues)
- [System Health Monitoring](#system-health-monitoring)

## Search Issues

### Result Explosion

**Symptoms**:
- Too many search results returned
- Slow response times
- High memory usage

**Diagnosis**:
1. Check result count:
```python
info = await search_info(query="your query")
if info["stats"]["total_results"] > 1000:
    print("Result explosion detected")
```

2. Check suggested filters:
```python
result = await search_context(query="your query")
if "suggested_filters" in result:
    print("Consider using filters:", result["suggested_filters"])
```

**Solutions**:
1. Use time-based filters:
```python
filters = {
    "time_range": {
        "start": datetime.now() - timedelta(days=7),
        "end": datetime.now()
    }
}
```

2. Use channel filters:
```python
filters = {
    "channels": ["relevant-channel"]
}
```

3. Use more specific search terms

### No Results Found

**Symptoms**:
- Empty result set
- Missing expected content

**Diagnosis**:
1. Check query formatting:
```python
info = await search_info(query="your query")
print("Query interpretation:", info["query"])
```

2. Verify channel access:
```python
channels = await get_channels()
print("Accessible channels:", channels)
```

**Solutions**:
1. Broaden search terms
2. Remove restrictive filters
3. Check channel permissions

## Token Management Issues

### Token Budget Exceeded

**Symptoms**:
- Truncated content
- Missing information
- Warning logs about token limits

**Diagnosis**:
1. Check token usage:
```python
token_manager = TokenManager()
status = token_manager.get_budget_status()
print("Token usage:", status)
```

2. Monitor warnings:
```python
warnings = token_manager.check_limits()
if warnings:
    print("Token limit warnings:", warnings)
```

**Solutions**:
1. Adjust token limits in config:
```python
config = ReductionConfig(
    cluster_summary_tokens=150,  # Reduce from 200
    global_summary_tokens=300    # Reduce from 400
)
```

2. Implement progressive loading:
```python
# Load content in chunks
for chunk in chunks:
    if token_manager.count_tokens(chunk) <= remaining_budget:
        content.append(chunk)
```

### Incorrect Token Counting

**Symptoms**:
- Unexpected truncation
- Inconsistent content lengths

**Diagnosis**:
1. Verify tokenizer:
```python
import tiktoken
ref_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
ref_count = len(ref_encoder.encode(text))
our_count = token_manager.count_tokens(text)
print(f"Reference: {ref_count}, Ours: {our_count}")
```

**Solutions**:
1. Update tokenizer model
2. Recalibrate token limits
3. Add buffer for safety margin

## Performance Issues

### Slow Search Response

**Symptoms**:
- High latency
- Timeouts
- Incomplete results

**Diagnosis**:
1. Time operations:
```python
import time

start = time.time()
result = await search_context(query="test")
duration = time.time() - start
print(f"Search took {duration:.2f}s")
```

2. Check result size:
```python
print("Results:", len(result["matches"]))
print("Total tokens:", sum(
    token_manager.count_tokens(m["text"])
    for m in result["matches"]
))
```

**Solutions**:
1. Use pagination:
```python
page_size = 50
for page in range(1, 5):
    results = await search_context(
        query="test",
        page=page,
        page_size=page_size
    )
```

2. Implement caching:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_search(query: str):
    return await search_context(query=query)
```

### Memory Issues

**Symptoms**:
- Out of memory errors
- System slowdown
- Process crashes

**Diagnosis**:
1. Monitor memory usage:
```python
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

**Solutions**:
1. Implement streaming:
```python
async for chunk in stream_results(query):
    process_chunk(chunk)
```

2. Clean up resources:
```python
# Clear caches periodically
token_manager.budget.reset()
monitoring.reset()
```

## Evidence Tracking Issues

### Missing Evidence

**Symptoms**:
- Broken evidence links
- Incomplete context
- Failed retrievals

**Diagnosis**:
1. Verify evidence format:
```python
def validate_evidence_id(evidence_id: str) -> bool:
    return (
        evidence_id.startswith("slack:ch/") or
        evidence_id.startswith("slack:th/")
    )
```

2. Check evidence existence:
```python
evidence = await get_evidence(evidence_ids=[evidence_id])
if not evidence["items"]:
    print(f"Evidence not found: {evidence_id}")
```

**Solutions**:
1. Fix evidence ID format:
```python
# Message evidence
evidence_id = f"slack:ch/{channel}/{ts}"

# Thread evidence
thread_id = f"slack:th/{thread_ts}"
```

2. Implement fallback retrieval:
```python
async def get_evidence_with_fallback(evidence_id: str):
    try:
        return await get_evidence([evidence_id])
    except Exception:
        # Try alternate sources
        return await get_evidence_from_backup(evidence_id)
```

### Evidence Consistency

**Symptoms**:
- Mismatched evidence
- Incorrect threading
- Lost context

**Diagnosis**:
1. Verify thread consistency:
```python
def check_thread_consistency(messages: List[Dict]):
    thread_ts = messages[0]["thread_ts"]
    return all(m["thread_ts"] == thread_ts for m in messages)
```

**Solutions**:
1. Implement verification:
```python
async def verify_evidence(evidence_ids: List[str]):
    evidence = await get_evidence(evidence_ids)
    return all(
        item["id"] in evidence_ids
        for item in evidence["items"]
    )
```

## Guardrail Issues

### Excessive Activations

**Symptoms**:
- Frequent guardrail triggers
- Performance impact
- Degraded user experience

**Diagnosis**:
1. Monitor activation rates:
```python
monitoring = MonitoringSystem()
rates = monitoring.get_activation_rate()
print("Activation rates:", rates)
```

2. Check specific rails:
```python
metrics = monitoring.get_metrics()
print("Rail activations:", metrics["counts"])
```

**Solutions**:
1. Adjust thresholds:
```python
class GuardRails:
    def check_result_explosion(self, results, threshold=1000):
        if len(results["matches"]) > threshold:
            return self._suggest_filters(results)
```

2. Implement progressive reduction:
```python
async def reduce_with_monitoring(results):
    reduced = await reducer.reduce_results(results)
    if monitoring.get_activation_rate("compression") > 10:
        # Adjust reduction parameters
        reducer.config.cluster_summary_tokens -= 50
    return reduced
```

## System Health Monitoring

### Monitoring Setup

1. Configure logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

2. Set up metrics:
```python
monitoring = MonitoringSystem()

# Record events
monitoring.record_activation(
    rail_type="explosion",
    details={"results": 1000},
    run_id="test_run",
    result_set_id="test_set"
)

# Export metrics
print(monitoring.export_metrics(format="text"))
```

### Health Checks

1. Token budget health:
```python
def check_token_health():
    status = token_manager.get_budget_status()
    warnings = token_manager.check_limits()
    return {
        "status": status,
        "warnings": warnings,
        "healthy": not warnings
    }
```

2. Guardrail health:
```python
def check_guardrail_health():
    metrics = monitoring.get_metrics()
    rates = monitoring.get_activation_rate()
    
    return {
        "status": "healthy" if all(
            rate < 10 for rate in rates.values()
        ) else "warning",
        "metrics": metrics
    }
```

### Recovery Procedures

1. Reset system state:
```python
async def reset_system():
    # Clear token budgets
    token_manager.budget.reset()
    
    # Reset monitoring
    monitoring.reset()
    
    # Clear caches
    reducer.result_cache.clear()
```

2. Implement circuit breakers:
```python
class CircuitBreaker:
    def __init__(self, threshold=10):
        self.failures = 0
        self.threshold = threshold
    
    async def execute(self, operation):
        if self.failures >= self.threshold:
            raise Exception("Circuit breaker open")
        
        try:
            result = await operation()
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            raise
```
