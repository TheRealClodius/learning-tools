# Slack Discovery Tools Specification

## Overview
This document outlines the design and implementation of integrated Slack discovery tools that combine vector search capabilities with real-time Assistant API-based discovery. These tools enable agents to efficiently search, explore, and understand Slack content using the most appropriate strategy for each task.

## Core Components

### 1. Discovery Strategies

#### Vector Search (Existing)
- Uses Pinecone for searching historical/archived content
- Pre-computed embeddings
- Efficient for large-scale historical searches
- Best for: Historical context, pattern finding, archived discussions

#### Real-time Discovery (New)
- Uses Slack's Assistant API endpoints
- Real-time context and information gathering
- Rate-limit friendly
- Best for: Recent discussions, active threads, current context

#### Combined Discovery
- Merges results from both approaches
- Provides unified view of historical and recent content
- Smart strategy selection based on query characteristics

### 2. Data Models

#### Discovery Request
```json
{
  "query": "string",
  "strategy": "auto|vector|realtime|combined",
  "options": {
    "include_pinned": boolean,
    "include_canvases": boolean,
    "max_results": number,
    "time_range": {
      "start": "ISO-8601",
      "end": "ISO-8601"
    }
  }
}
```

#### Discovery Response
```json
{
  "result_set_id": "string",
  "source": "vector|realtime|combined",
  "metadata": {
    "total_results": number,
    "vector_results": number,
    "realtime_results": number,
    "channels": [
      {
        "name": "string",
        "count": number
      }
    ],
    "dates": [
      {
        "range": "historical|recent",
        "count": number
      }
    ]
  },
  "facets": {
    "pinned": [
      {
        "type": "message|file|canvas",
        "count": number
      }
    ],
    "canvases": [
      {
        "title": "string",
        "count": number
      }
    ],
    "threads": [
      {
        "participants": number,
        "messages": number
      }
    ]
  },
  "items": [
    {
      "id": "string",
      "source": "vector|realtime",
      "type": "message|canvas|file",
      "preview": "string",
      "score": number,
      "metadata": {
        "channel": "string",
        "timestamp": "ISO-8601",
        "author": "string",
        "is_pinned": boolean,
        "has_attachments": boolean,
        "reaction_count": number,
        "reply_count": number
      }
    }
  ]
}
```

### 3. Key Features

#### Progressive Discovery
1. Initial metadata and preview fetch
2. Faceted navigation
3. Drill-down capabilities
4. Context enrichment

#### Smart Strategy Selection
- Time-based indicators in query
- Content type requirements
- Query complexity analysis
- User intent recognition

#### Result Enrichment
- Pinned item integration
- Canvas content
- Thread context
- Reaction data
- File metadata

### 4. Implementation Components

#### SlackDiscoveryTools
- Main interface for discovery operations
- Strategy selection and execution
- Result formatting and merging

#### DiscoveryStrategies
- VectorSearchStrategy
- RealtimeDiscoveryStrategy
- CombinedDiscoveryStrategy

#### ResultEnricher
- Pinned item fetching
- Canvas integration
- Thread context gathering

#### ResultReducer
- Metadata extraction
- Facet computation
- Preview generation

### 5. Rate Limiting & Caching

#### Rate Limit Management
- Assistant API: Higher limits
- Conversations API: Strict limits
- Smart endpoint selection

#### Caching Strategy
```python
{
  "metadata": {  # Redis (fast, short TTL)
    "result_sets": {...},
    "facets": {...}
  },
  "content": {   # Database (longer TTL)
    "messages": {...},
    "attachments": {...}
  }
}
```

### 6. Usage Patterns

#### Basic Discovery
```python
results = await discovery.discover(
    query="deployment issues",
    strategy="auto"
)
```

#### Targeted Search
```python
results = await discovery.discover(
    query="recent API changes",
    strategy="realtime"
)
```

#### Historical Analysis
```python
results = await discovery.discover(
    query="previous incidents",
    strategy="vector"
)
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Basic discovery interface
2. Strategy selection logic
3. Result format standardization

### Phase 2: Enhanced Features
1. Result enrichment
2. Facet computation
3. Progressive loading

### Phase 3: Optimization
1. Caching implementation
2. Rate limit optimization
3. Performance tuning

## Future Considerations

### Potential Enhancements
1. Advanced query understanding
2. Custom ranking algorithms
3. User preference learning
4. Automated summarization
5. Thread relationship mapping

### Integration Points
1. Memory systems
2. Other agent capabilities
3. External tools and services

## Security & Privacy

### Data Access
- Respect channel permissions
- User token validation
- Content visibility rules

### Rate Limiting
- Smart retry logic
- Quota management
- Request prioritization
