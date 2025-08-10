# Prompt Assembly Documentation

## Overview
This document details how prompts are assembled in the Signal AI agent system. The prompt assembly process involves multiple layers of context enrichment, memory retrieval, and intelligent formatting.

## Architecture Overview

```
User Message → Context Enrichment → Memory Retrieval → Insights Assembly → Final Prompt → Claude API
```

## Components

### 1. System Prompt (Base Layer)
**Location:** `/workspace/agents/system prompts/client_agent_sys_prompt.yaml`

The system prompt is the foundation that defines Signal's personality, capabilities, and behavior:

```yaml
system_prompt: |
  You are Signal, a super-intelligent AI agent that helps people perform complex actions...
  
  REASONING STRATEGY:
  - Before responding, take a moment to think and outline your plan
  - If you need to use tools, explain which ones and why
  
  MEMORY AND CONTEXT SYSTEM:
  - Recent conversation history is automatically retrieved
  - Your responses are automatically stored in memory
  
  TOOL DISCOVERY SYSTEM:
  - reg_categories, reg_list, reg_search, reg_describe, execute_tool
  
  ADAPTIVE PROBLEM-SOLVING RULES:
  - Break complex requests into logical sub-components
  - Use iterative refinement
```

### 2. Context Enrichment Layer

**Process Flow:**
1. **Platform Context** (timestamp, user timezone, platform type)
2. **User Identity** (user_id, user_name, user_title)
3. **Automatic Memory Retrieval** (recent conversation history)
4. **Local Buffer Assembly** (pins and recommendations)

### 3. Memory System Integration

**Automatic Memory Operations:**
- **Start of conversation:** Automatically retrieves last 10 relevant memory entries
- **End of conversation:** Automatically stores the interaction

**Memory Context Format:**
```
Recent conversation context:
User: [previous message]...
Assistant: [previous response]...

Relevant conversation history:
- [historical context 1]...
- [historical context 2]...
```

### 4. Conversation Insights (Pins & Recommendations)

**Two-Section Format:**

**PINS:** User preferences, needs, constraints, conversation context
- Max 2 items added to prompt
- Relevance threshold: 0.1
- Examples: Personal info, requirements, domain knowledge

**RECOMMENDATIONS:** Tool usage guidance, error patterns, workflow optimization  
- Max 1 item added to prompt
- Relevance threshold: 0.05
- Examples: Tool effectiveness, parameter optimization

### 5. Final Prompt Assembly

The complete prompt structure:

```
[SYSTEM PROMPT with reasoning control]
[MEMORY CONTEXT from automatic retrieval]
[PINS section if relevant]
[RECOMMENDATIONS section if relevant]
[USER MESSAGE with platform context]
```

## Detailed Assembly Process

### Step 1: Initial Message Reception
```python
# From client_agent.py
full_message = message
if context:
    platform = context.get('platform', 'unknown')
    timestamp = context.get('timestamp', 'unknown')
    user_timezone = context.get('user_timezone', '')
    user_name = context.get('user_name', '')
    user_title = context.get('user_title', '')
    
    context_parts = [f"Platform={platform}"]
    if user_name:
        context_parts.append(f"User={user_name}")
    if user_title:
        context_parts.append(f"Role={user_title}")
    # Add timestamp with timezone
    full_message += f"\n\n[Context: {', '.join(context_parts)}]"
```

### Step 2: Automatic Memory Retrieval
```python
# Automatic at start of run_agent_loop
memory_args = {
    "query": "recent history",
    "user_id": user_id,
    "max_results": 10
}
memory_result = await self.tool_executor.execute_command("memory.retrieve", memory_args)

# Formats into memory_context:
memory_context = """
Recent conversation context:
User: [msg1]...
Assistant: [response1]...

Relevant conversation history:
- [historical entry1]...
- [historical entry2]...
"""
```

### Step 3: Local Buffer Assembly (Pins & Recommendations)
```python
# From assemble_from_local_buffer()
# Filters and scores insights by relevance
enriched_message = user_message

if relevant_pins:
    enriched_message += "\n\nRelevant context from our previous conversations:"
    for pin in relevant_pins[:2]:  # Max 2 pins
        enriched_message += f"\n- {pin['content']}"

if relevant_recommendations:
    enriched_message += "\n\nBased on previous interactions, please note:"
    for rec in relevant_recommendations[:1]:  # Max 1 recommendation
        enriched_message += f"\n- {rec['content']}"
```

### Step 4: Reasoning Control
```python
# Determines if thinking tags are needed
needs_reasoning = self._needs_reasoning(user_message)

if not needs_reasoning:
    # For simple messages (greetings, acknowledgments)
    adjusted_prompt = system_prompt + "\n\nIMPORTANT: For this specific message, respond directly WITHOUT using <thinking> tags."
else:
    adjusted_prompt = system_prompt
```

### Step 5: Final API Call
```python
# Complete assembled prompt sent to Claude
messages = [{
    "role": "user",
    "content": [
        {
            "type": "text", 
            "text": enriched_message,  # Contains memory + pins + recommendations + user message
            "cache_control": {"type": "ephemeral"}
        }
    ]
}]

response = self.client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=[{
        "type": "text", 
        "text": adjusted_prompt,  # System prompt with reasoning control
        "cache_control": {"type": "ephemeral"}
    }],
    messages=messages,
    tools=self.tools
)
```

## Conversation Insights Agent

**Separate Gemini-powered agent that:**
- Analyzes each interaction after completion
- Maintains evolving user understanding
- Updates pins and recommendations
- Uses stateful analysis comparing new interactions to existing insights

**Prompt Template:**
```yaml
CURRENT USER INSIGHTS:
{existing_insights}

NEW INTERACTION TO ANALYZE:
User: "{user_message}"
Agent Response: "{agent_response}"
Tools Used: {execution_summary}
Agent Reasoning: {thinking_summary}

ANALYSIS TASKS:
1. COMPARE: How does this relate to existing insights?
2. UPDATE: Do insights need refinement?
3. ADD: What new patterns emerge?
4. DELETE: Are insights outdated?
5. SYNTHESIZE: Meta-patterns across interactions
6. DIAGNOSE: Why tools failed
7. OPTIMIZE: Parameter improvements
```

## Key Features

### 1. User Isolation
- Each user has their own buffer (`user_buffers[user_id]`)
- Insights are maintained per user
- Memory is retrieved per user
- 30-minute buffer expiry for inactive users

### 2. Relevance Scoring
```python
# Keyword overlap calculation
user_words = set(user_message.lower().split())
note_words = set((notes + ' ' + related_question).lower().split())
overlap = len(user_words.intersection(note_words)) / len(user_words)

# Context boost for recent context
context_boost = 0.2 if any(word in notes.lower() 
                         for word in user_message.lower().split()[:3]) else 0

relevance = overlap + context_boost
```

### 3. Prompt Caching
- System prompt marked with `cache_control: {"type": "ephemeral"}`
- Initial user message marked as cacheable
- Reduces token usage across agent loop iterations

### 4. Adaptive Reasoning
- Simple messages (greetings, thanks) skip `<thinking>` tags
- Complex queries trigger full reasoning process
- Controlled via `_needs_reasoning()` method

## Performance Optimizations

1. **Prompt Caching:** Ephemeral caching for system prompt and initial message
2. **Selective Insights:** Only relevant pins/recommendations added (relevance > threshold)
3. **Length Limits:** Max 800 chars for context additions
4. **Background Processing:** Memory operations and insights analysis run async
5. **Rate Limiting:** Token estimation and rate limit handling

## Example Final Prompt

```
[SYSTEM PROMPT]
You are Signal, a super-intelligent AI agent...
[Full system prompt from YAML]

[USER MESSAGE]
Recent conversation context:
User: What's the weather like?
Assistant: I can help you check the weather. What location...

Relevant context from our previous conversations:
- User prefers metric units for temperature
- User frequently asks about weather in San Francisco

Based on previous interactions, please note:
- Weather tools work best with city names rather than zip codes

What's the weather in SF today?

[Context: Platform=slack, User=John Doe, Role=Engineer, Current time for user: 2024-01-15 10:30 AM (PST)]
```

## Debugging & Monitoring

**Log Prefixes:**
- `BUFFER-SYSTEM:` Buffer operations
- `AUTO-MEMORY:` Automatic memory retrieval/storage
- `PROMPT-ASSEMBLY:` Context assembly
- `INSIGHTS-AGENT:` Conversation insights generation
- `ENRICHED-PROMPT-DEBUG:` Final enriched prompt

**Key Debug Points:**
1. Check `enriched_message` vs `user_message` to see additions
2. Monitor `memory_context` for retrieved history
3. Track `relevant_pins` and `relevant_recommendations` counts
4. Verify `needs_reasoning` flag for message type

## Configuration Files

1. **System Prompt:** `/agents/system prompts/client_agent_sys_prompt.yaml`
2. **Insights Config:** `/agents/system prompts/convo_insights_agent.yaml`
3. **Model Config:** Embedded in YAML files
   - Claude: 8192 max tokens, 0.4 temperature, 24 max iterations
   - Gemini: 1000 max tokens, 0.3 temperature for insights

## Summary

The prompt assembly is a sophisticated multi-layer process that:
1. Starts with a base system prompt defining Signal's capabilities
2. Automatically retrieves relevant memory context
3. Enriches with user-specific pins and recommendations
4. Adds platform and temporal context
5. Applies reasoning control based on message complexity
6. Uses prompt caching for efficiency
7. Continuously learns through the Conversation Insights Agent

This creates a highly personalized, context-aware experience that adapts to each user's needs and patterns over time.