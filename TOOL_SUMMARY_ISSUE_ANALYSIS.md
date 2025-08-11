# Tool Summary Display Issue Analysis

## Executive Summary
Tool summaries are not being displayed during real-time execution because the **GEMINI_API_KEY environment variable is not set**. Without this key, the tool summarization feature is automatically disabled, preventing the generation and display of AI-powered tool execution summaries.

## Root Cause Analysis

### 1. Environment Configuration Issue
- **Primary Issue**: The `GEMINI_API_KEY` environment variable is not configured
- **Location**: Should be set in `.env.local` file or as an environment variable
- **Impact**: Without this key, the entire summarization pipeline is disabled

### 2. Code Flow Analysis

#### Initialization Chain
1. **ClientAgent Initialization** (`agents/client_agent.py`):
   ```python
   # Line 72-73
   self.summarization_config = self.config.get('summarization_config', {})
   self.summarization_enabled = self.summarization_config.get('enabled', True)
   
   # Line 77-82
   if self.summarization_enabled:
       self._init_summarization_client()  # This checks for GEMINI_API_KEY
       self.execution_summarizer = ExecutionSummarizer()
   else:
       self.execution_summarizer = None
   ```

2. **Summarization Client Setup** (`agents/client_agent.py:250-267`):
   ```python
   def _init_summarization_client(self):
       if model_name.startswith('gemini'):
           api_key = os.environ.get("GEMINI_API_KEY")
           if api_key:
               self.gemini_client = genai.Client(api_key=api_key)
               # ... setup continues
           else:
               logger.warning("GEMINI_API_KEY not found - tool summarization disabled")
               self.summarization_enabled = False  # DISABLED HERE!
   ```

#### Execution Flow
3. **Tool Execution** (`agents/client_agent.py:1069-1076`):
   ```python
   # When execute_tool is called:
   if streaming_callback and self.summarization_enabled:
       # Generate and stream tool summary
       await self._generate_tool_summary_streaming(...)
   # If disabled, this block is skipped entirely!
   ```

4. **Streaming Handler** (`interfaces/slack/handlers/streaming_handler.py:268-270`):
   ```python
   elif content_type == "tool_summary_chunk":
       # Stream Gemini summary chunks as they arrive
       await streaming_handler.append_to_current_tool(content)
   ```

### 3. Why Summaries Don't Appear

When `GEMINI_API_KEY` is not set:

1. ❌ `self.summarization_enabled` becomes `False`
2. ❌ `self.execution_summarizer` is set to `None`
3. ❌ Tool execution skips the summary generation step
4. ❌ No `tool_summary_chunk` events are sent to the streaming handler
5. ❌ The UI never receives summary content to display

## Visual Flow Diagram

```
User Request
    ↓
ClientAgent.process_request()
    ↓
Tool Execution (execute_tool)
    ↓
[Check: self.summarization_enabled?]
    ├─ YES (with GEMINI_API_KEY) → Generate Summary → Stream Chunks → Display in UI ✓
    └─ NO (missing key) → Skip Summary → No Display ✗ ← CURRENT STATE
```

## Configuration Requirements

### Required Environment Variables
```bash
# In .env.local or environment
ANTHROPIC_API_KEY=your_anthropic_key  # For main agent
GEMINI_API_KEY=your_gemini_key        # For tool summarization
```

### Configuration File
The feature is enabled by default in `agents/system prompts/client_agent_sys_prompt.yaml`:
```yaml
summarization_config:
  enabled: true  # Feature is enabled, but requires API key
```

## Solutions

### Solution 1: Set GEMINI_API_KEY (Recommended)
1. Create or edit `.env.local` file in the workspace root
2. Add: `GEMINI_API_KEY=your_actual_gemini_api_key`
3. Restart the application

### Solution 2: Implement Fallback Summaries
If Gemini API is not available, implement a fallback mechanism:
- Use Claude API (already available) for summarization
- Generate basic summaries without AI
- Display raw tool results with formatting

### Solution 3: Better Error Messaging
Add explicit warnings when summarization is disabled:
- Log clear message at startup
- Show notification in UI when summaries are unavailable
- Provide setup instructions to users

## Testing Verification

To verify the fix works:
1. Set `GEMINI_API_KEY` environment variable
2. Run the application
3. Execute a command that uses tools
4. You should see:
   - Real-time streaming of tool summaries
   - Formatted narrative descriptions of tool actions
   - Progressive updates as operations complete

## Code Locations

Key files involved in the tool summarization pipeline:

1. **Configuration**: `agents/system prompts/client_agent_sys_prompt.yaml`
2. **Initialization**: `agents/client_agent.py` (lines 70-82, 250-280)
3. **Summary Generation**: `agents/execution_summarizer.py` (lines 112-207)
4. **Tool Execution**: `agents/client_agent.py` (lines 1055-1077)
5. **UI Streaming**: `interfaces/slack/handlers/streaming_handler.py` (lines 105-110, 268-270)
6. **CLI Display**: `interfaces/cli_interface.py` (lines 254-257)

## Conclusion

The tool summaries feature is fully implemented and functional but requires the `GEMINI_API_KEY` environment variable to be set. Without this key, the entire summarization pipeline is disabled, resulting in no tool summaries being displayed during real-time execution. Setting the API key will immediately restore this functionality.