# Prompt Assembly Visual Flow

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL AI - PROMPT ASSEMBLY FLOW                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ User Message │ ──────┐
└──────────────┘       │
                       ▼
              ┌─────────────────┐       ┌──────────────────┐
              │ Context Extract │◄──────│ Platform Context │
              │   • User ID     │       │  • Slack/CLI     │
              │   • Timestamp   │       │  • User Info     │
              │   • Platform    │       │  • Timezone      │
              └────────┬────────┘       └──────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐        ┌────────────────────┐       ┌─────────────────┐
│ Memory Retrieval │        │ Conversation       │◄──────│ Gemini Flash    │
│   (Automatic)    │        │ Insights Agent     │       │   2.5 Model     │
│                  │        │                    │       └─────────────────┘
│ • Recent history │        │ • Analyze patterns │
│ • Last 10 items  │        │ • Update PINS      │
│ • Semantic search│        │ • Update RECS      │
└────────┬─────────┘        └─────────┬──────────┘
         │                             │
         │                             ▼
         │                  ┌──────────────────────┐
         │                  │   Local Buffer       │
         │                  │   • User-isolated    │
         │                  │   • 30-min expiry    │
         │                  │   • Important notes  │
         │                  └──────────┬───────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Relevance Scoring    │
         │ • Keyword overlap    │
         │ • Context boost      │
         │ • Threshold check    │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐        ┌─────────────────┐
         │ Prompt Assembly      │◄───────│ System Prompt   │
         │                      │        │  (YAML Config)  │
         │ • Add memory context │        │                 │
         │ • Add relevant PINS  │        │ • Base prompt   │
         │ • Add relevant RECS  │        │ • Tool schemas  │
         │ • Add user message   │        │ • Instructions  │
         └──────────┬───────────┘        └─────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Reasoning Control    │
         │ • Simple vs Complex  │
         │ • Thinking tags?     │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Final Prompt       │
         │   [Cached]           │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Claude 3.5 Sonnet   │
         │  • Tool execution    │
         │  • Response gen      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Agent Response     │
         └──────────┬───────────┘
                    │
         ┌──────────┴───────────┐
         │                      │
         ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│ Memory Storage   │   │ Buffer Update    │
│  (Automatic)     │   │ (Immediate)      │
└──────────────────┘   └──────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Insights Analysis    │
         │  (Async Background)  │
         └──────────────────────┘
```

## Detailed Component Flow

### 1. Input Processing Pipeline
```
User Message
    │
    ├─► Extract user_id ──────────► User Buffer Lookup
    │
    ├─► Extract platform ─────────► Context Formatting
    │
    └─► Extract timestamp ────────► Timezone Conversion
```

### 2. Memory & Context Assembly
```
┌─────────────────────────────────────────────────┐
│            AUTOMATIC MEMORY RETRIEVAL           │
├─────────────────────────────────────────────────┤
│ Query: "recent history"                         │
│ User ID: {user_id}                              │
│ Max Results: 10                                 │
│                                                  │
│ Returns:                                         │
│ • short_term_memory[]                           │
│ • retrieved_pages[]                             │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              BUFFER ASSEMBLY                    │
├─────────────────────────────────────────────────┤
│ PINS (Conversation Context):                    │
│ • User preferences                              │
│ • Personal information                          │
│ • Domain knowledge                              │
│ [Max 2, Relevance > 0.1]                        │
│                                                  │
│ RECOMMENDATIONS (Tool Guidance):                │
│ • Tool effectiveness patterns                   │
│ • Error handling strategies                     │
│ • Parameter optimizations                       │
│ [Max 1, Relevance > 0.05]                       │
└─────────────────────────────────────────────────┘
```

### 3. Final Prompt Structure
```
┌─────────────────────────────────────────────────┐
│              ASSEMBLED PROMPT                   │
├─────────────────────────────────────────────────┤
│ [SYSTEM PROMPT - Cached]                        │
│ You are Signal, a super-intelligent AI agent... │
│ REASONING STRATEGY: ...                         │
│ MEMORY AND CONTEXT SYSTEM: ...                  │
│ TOOL DISCOVERY SYSTEM: ...                      │
│ {+ reasoning control modifier if simple}        │
├─────────────────────────────────────────────────┤
│ [USER MESSAGE]                                  │
│                                                  │
│ Recent conversation context:                    │
│ User: {previous_message}                        │
│ Assistant: {previous_response}                  │
│                                                  │
│ Relevant conversation history:                  │
│ - {historical_entry_1}                          │
│ - {historical_entry_2}                          │
│                                                  │
│ Relevant context from our conversations:        │
│ - {pin_1}                                       │
│ - {pin_2}                                       │
│                                                  │
│ Based on previous interactions, please note:    │
│ - {recommendation_1}                            │
│                                                  │
│ {user_message}                                  │
│                                                  │
│ [Context: Platform=slack, User=John,            │
│  Current time: 2024-01-15 10:30 AM (PST)]      │
└─────────────────────────────────────────────────┘
```

### 4. Relevance Scoring Algorithm
```
┌─────────────────────────────────────────────────┐
│           RELEVANCE CALCULATION                 │
├─────────────────────────────────────────────────┤
│                                                  │
│ user_words = set(user_message.split())          │
│ note_words = set(insight_text.split())          │
│                                                  │
│ overlap = len(intersection) / len(user_words)   │
│                                                  │
│ IF first_3_words match:                         │
│    context_boost = 0.2                          │
│ ELSE:                                            │
│    context_boost = 0.0                          │
│                                                  │
│ relevance = overlap + context_boost             │
│                                                  │
│ IF insight_type == 'pin':                       │
│    threshold = 0.1                              │
│ ELSE IF insight_type == 'recommendation':       │
│    threshold = 0.05                             │
│                                                  │
│ INCLUDE IF relevance > threshold                │
└─────────────────────────────────────────────────┘
```

### 5. Async Processing Flow
```
Main Thread                    Background Tasks
    │                               │
    ├─► Generate Response           │
    │                               │
    ├─► Update Local Buffer         │
    │                               │
    ├─► Return Response ────────────┤
    │                               │
    │                               ├─► Store in Memory
    │                               │
    │                               ├─► Analyze with Gemini
    │                               │
    │                               └─► Update Insights
    │
    └─► Continue
```

## Key Performance Features

### Prompt Caching
```
┌─────────────────────────────────────────────────┐
│              CACHE STRATEGY                     │
├─────────────────────────────────────────────────┤
│                                                  │
│ System Prompt:                                  │
│   cache_control: {"type": "ephemeral"}          │
│   ► Reused across iterations                    │
│                                                  │
│ Initial User Message:                           │
│   cache_control: {"type": "ephemeral"}          │
│   ► Reused in agent loop                        │
│                                                  │
│ Benefits:                                        │
│   • Reduced token usage                         │
│   • Faster subsequent calls                     │
│   • Lower costs                                 │
└─────────────────────────────────────────────────┘
```

### User Isolation
```
┌─────────────────────────────────────────────────┐
│              USER ISOLATION                     │
├─────────────────────────────────────────────────┤
│                                                  │
│ user_buffers = {                                │
│   "user_123": {                                 │
│     "important": {...},                         │
│     "last_updated": timestamp                   │
│   },                                            │
│   "user_456": {                                 │
│     "important": {...},                         │
│     "last_updated": timestamp                   │
│   }                                             │
│ }                                                │
│                                                  │
│ • Each user has separate buffer                 │
│ • 30-minute expiry for inactive users           │
│ • No cross-contamination                        │
└─────────────────────────────────────────────────┘
```

## Monitoring & Debugging

### Log Prefixes for Tracing
```
BUFFER-SYSTEM     ► Buffer operations
AUTO-MEMORY       ► Memory retrieval/storage
PROMPT-ASSEMBLY   ► Context assembly
INSIGHTS-AGENT    ► Conversation insights
ENRICHED-PROMPT   ► Final prompt debug
AGENT-TIMING      ► Performance metrics
```

### Debug Checkpoints
```
1. user_message vs enriched_message
   ► Shows what context was added

2. memory_context content
   ► Shows retrieved conversation history

3. relevant_pins/relevant_recommendations
   ► Shows which insights were included

4. needs_reasoning flag
   ► Shows if thinking tags will be used

5. final_response extraction
   ► Shows response after tag removal
```

## Configuration Summary

### Models
- **Claude 3.5 Sonnet**: Main reasoning (8192 tokens, temp 0.4)
- **Gemini 2.5 Flash**: Insights analysis (1000 tokens, temp 0.3)

### Limits
- **Memory**: 10 recent conversations retrieved
- **PINS**: Max 2 per prompt (relevance > 0.1)
- **RECOMMENDATIONS**: Max 1 per prompt (relevance > 0.05)
- **Context Length**: 800 chars max addition
- **Buffer Expiry**: 30 minutes
- **Agent Iterations**: Max 24

### Files
- System Prompt: `/agents/system prompts/client_agent_sys_prompt.yaml`
- Insights Config: `/agents/system prompts/convo_insights_agent.yaml`
- Main Agent: `/agents/client_agent.py`
- Insights Agent: `/agents/convo_insights_agent.py`