# Agent Usage Patterns

This guide documents common usage patterns for Signal agents, focusing on search, evidence tracking, and result reduction.

## Table of Contents
- [Search Patterns](#search-patterns)
- [Evidence Tracking](#evidence-tracking)
- [Progressive Reduction](#progressive-reduction)
- [Guardrails](#guardrails)
- [Troubleshooting](#troubleshooting)

## Search Patterns

### Basic Search Flow
```python
# 1. Initial search with context
context = await search_context(query="API performance issues")

# 2. Get search metadata
info = await search_info(query="API performance issues")

# 3. Process results through reducer
reduced = await reducer.reduce_results(
    results=context,
    query="API performance issues"
)

# 4. Track findings with evidence
finding = await add_finding(
    summary="API latency spikes identified",
    evidence_ids=reduced["clusters"][0]["evidence_ids"],
    finding_type="issue"
)
```

### Search with Filters
```python
# Time-based search
filters = {
    "time_range": {
        "start": datetime.now() - timedelta(days=7),
        "end": datetime.now()
    }
}
context = await search_context(query="API issues", filters=filters)

# Channel-based search
filters = {
    "channels": ["engineering", "incidents"]
}
context = await search_context(query="API issues", filters=filters)
```

### Related Searches
```python
# Initial search
context = await search_context(query="database performance")

# Get metadata for time context
info = await search_info(query="database performance")

# Related search in same time period
related = await search_context(
    query="API latency",
    filters={"time_range": info["stats"]["time_range"]}
)
```

## Evidence Tracking

### Basic Evidence Flow
```python
# 1. Get search results
context = await search_context(query="API issues")

# 2. Extract evidence IDs
evidence_ids = [
    f"slack:ch/{match['channel']}/{match['ts']}"
    for match in context["matches"][:2]
]

# 3. Add finding with evidence
finding = await add_finding(
    summary="API performance degradation",
    evidence_ids=evidence_ids,
    finding_type="issue"
)

# 4. Retrieve evidence later
evidence = await get_evidence(evidence_ids=evidence_ids)
```

### Evidence in Progressive Reduction
```python
# 1. Search and reduce
reduced = await reducer.reduce_results(
    results=context,
    query="API issues"
)

# 2. Extract evidence from clusters
evidence_ids = []
for cluster in reduced["clusters"].values():
    evidence_ids.extend(cluster["evidence_ids"])

# 3. Add finding with reduced evidence
finding = await add_finding(
    summary="Key API issues identified",
    evidence_ids=evidence_ids[:5],  # Take top evidence
    finding_type="investigation"
)
```

## Progressive Reduction

### Basic Reduction Flow
```python
# 1. Configure reduction
config = ReductionConfig(
    page_size=50,
    cluster_summary_tokens=200,
    global_summary_tokens=400
)
reducer = ProgressiveReducer(config=config)

# 2. Process results
reduced = await reducer.reduce_results(
    results=context,
    query="API performance"
)

# 3. Access components
global_brief = reduced["global"]["brief"]
clusters = reduced["clusters"]
token_status = reduced["global"]["token_status"]
```

### Drill-down Flow
```python
# 1. Get reduced results
reduced = await reducer.reduce_results(
    results=context,
    query="API issues"
)

# 2. Get cluster evidence ID
cluster = next(iter(reduced["clusters"].values()))
evidence_id = cluster["evidence_ids"][0]

# 3. Drill down to detail
detail = await reducer.drill_down(
    result_set_id=reduced["result_set_id"],
    item_id=evidence_id
)
```

## Guardrails

### Token Budget Management
```python
# 1. Create token manager
token_manager = TokenManager()

# 2. Check content tokens
text = "Long content..."
tokens = token_manager.count_tokens(text)

# 3. Enforce limits
if tokens > token_manager.limits.CLUSTER_SUMMARY:
    text = token_manager.enforce_cluster_limit(text, "cluster_1")
```

### Monitoring Guardrails
```python
# 1. Create monitoring
monitoring = MonitoringSystem()

# 2. Record activations
monitoring.record_activation(
    rail_type="explosion",
    details={"results": 1000},
    run_id="test_run",
    result_set_id="test_set"
)

# 3. Get metrics
metrics = monitoring.get_metrics()
rates = monitoring.get_activation_rate()
```

## Troubleshooting

### Common Issues and Solutions

1. **Result Explosion**
   - **Symptom**: Too many search results
   - **Solution**: Check suggested filters in response
   ```python
   result = await search_context(query="common term")
   if "suggested_filters" in result:
       # Use suggested filters in next search
       filters = result["suggested_filters"]
   ```

2. **Token Budget Exceeded**
   - **Symptom**: Content truncation
   - **Solution**: Monitor token status
   ```python
   reduced = await reducer.reduce_results(...)
   token_status = reduced["global"]["token_status"]
   if token_status["clusters"]["current"] > token_status["clusters"]["limit"]:
       logger.warning("Cluster token budget exceeded")
   ```

3. **Missing Evidence**
   - **Symptom**: Evidence not found
   - **Solution**: Verify evidence ID format
   ```python
   # Correct format for message evidence
   evidence_id = f"slack:ch/{channel}/{ts}"
   
   # Correct format for thread evidence
   thread_id = f"slack:th/{thread_ts}"
   ```

4. **Deduplication Issues**
   - **Symptom**: Duplicate content in results
   - **Solution**: Check similarity threshold
   ```python
   # Manually check similarity
   similarity = guard_rails._compute_similarity(text1, text2)
   if similarity > 0.8:
       logger.warning("High similarity detected")
   ```

### Monitoring and Debugging

1. **Check Guardrail Activations**
   ```python
   monitoring = MonitoringSystem()
   metrics = monitoring.get_metrics()
   
   # Export for analysis
   json_metrics = monitoring.export_metrics(format="json")
   print(json_metrics)
   ```

2. **Token Usage Analysis**
   ```python
   token_manager = TokenManager()
   
   # Track usage
   budget_status = token_manager.get_budget_status()
   warnings = token_manager.check_limits()
   
   if warnings:
       logger.warning("Token limits exceeded:", warnings)
   ```

3. **Performance Monitoring**
   ```python
   # Track reduction performance
   start = time.time()
   reduced = await reducer.reduce_results(...)
   duration = time.time() - start
   
   if duration > 5:  # 5 second threshold
       logger.warning(f"Slow reduction: {duration:.2f}s")
   ```

### Best Practices

1. **Always check token budgets before processing large content**
2. **Use suggested filters when available**
3. **Monitor guardrail activation rates for system health**
4. **Keep evidence IDs in correct format**
5. **Use progressive reduction for large result sets**
6. **Implement proper error handling**
7. **Monitor and log system metrics**
