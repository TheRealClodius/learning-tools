"""
Tool execution result summarizer utility class.

Provides intelligent summarization of tool execution results using various LLM clients
with fallback logic for when LLM is unavailable.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ExecutionSummarizer:
    """Utility class for tool execution result summarization - accepts external clients"""
    
    def __init__(self, summarization_config: Dict[str, Any] = None):
        """
        Initialize ExecutionSummarizer with external client configuration.
        
        Args:
            summarization_config: Configuration dict with model, temperature, max_tokens
        """
        self.config = summarization_config or {}
        self.model_name = self.config.get('model', 'gemini-2.5-flash')
        self.temperature = self.config.get('temperature', 0)
        self.max_tokens = self.config.get('max_tokens', 1000)
    
    async def create_narrative_summary(
        self, 
        tool_name: str, 
        tool_args: Dict, 
        result_data: str, 
        initial_narrative: str = "",
        gemini_client=None,
        gemini_types=None,
        claude_client=None,
        streaming_callback=None
    ) -> str:
        """
        Create a complete narrative summary of a tool execution using external clients.
        
        Args:
            tool_name: Name of the tool that was executed (e.g., "weather.get")
            tool_args: Arguments that were passed to the tool
            result_data: Raw result data (usually JSON)
            initial_narrative: The initial narrative (used for fallback only)
            gemini_client: External Gemini client instance
            gemini_types: External Gemini types module
            claude_client: External Claude client instance (future use)
            
        Returns:
            Complete narrative with result summary
        """
        # Try the configured model with provided clients
        if self.model_name.startswith('gemini') and gemini_client and gemini_types:
            try:
                prompt = f"""Create a single-line narrative summary of this tool execution using the template format.

Tool executed: {tool_name}
Arguments: {json.dumps(tool_args, indent=2) if tool_args else "none"}
Result JSON: {result_data[:800]}

Template format: ⚡️Used *[tool_name]* to/for [purpose extracted from args]. [Result summary from JSON]

Examples:
- Tool: weather.get, Args: {{"location": "London"}}, Result: {{"temperature": 72, "conditions": "partly cloudy"}}
  Output: ⚡️Used *weather.get* to get info for London. Got current temperature 72°F, partly cloudy
  
- Tool: slack_chatter.search_messages, Args: {{"query": "project updates"}}, Result: {{"messages": [...5 items...], "channels": ["#general", "#dev-team"]}}
  Output: ⚡️Used *slack_chatter.search_messages* to search for 'project updates'. Found 5 messages from #general and #dev-team channels

- Tool: weather, Args: {{"city": "London"}}, Result: {{"status": "success", "data": {{"temp": 18, "description": "cloudy"}}}}
  Output: ⚡️Used *weather* to get info for London. Got 18°C, cloudy

Write the narrative summary:"""
                
                # Use the new SDK format with GenerateContentConfig
                config = gemini_types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                if streaming_callback:
                    # Streaming mode: send chunks as they arrive
                    narrative_chunks = []
                    
                    def stream_response():
                        return gemini_client.models.generate_content_stream(
                            model=self.model_name,
                            contents=prompt,
                            config=config
                        )
                    
                    response_stream = await asyncio.to_thread(stream_response)
                    
                    for chunk in response_stream:
                        if chunk.text:
                            chunk_text = chunk.text
                            narrative_chunks.append(chunk_text)
                            
                            # Stream each chunk immediately to UI
                            await streaming_callback(chunk_text, "tool_summary_chunk")
                    
                    # Combine all chunks for final return
                    narrative = "".join(narrative_chunks).strip()
                else:
                    # Non-streaming mode: get complete response
                    response = await asyncio.to_thread(
                        gemini_client.models.generate_content,
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    narrative = response.text.strip()
                
                # Ensure it starts with the emoji if not present
                if not narrative.startswith("⚡️"):
                    narrative = "⚡️" + narrative
                    
                return narrative
                
            except Exception as e:
                logger.warning(f"{self.model_name} narrative generation failed: {e}")
        
        elif self.model_name.startswith('claude') and claude_client:
            # Future: Implement Claude-based summarization
            logger.info(f"Claude summarization not yet implemented, using fallback")
        
        # Fallback using the initial narrative or generate fallback
        return self._fallback_narrative(initial_narrative, tool_name, result_data)
    
    def _fallback_narrative(self, initial_narrative: str, tool_name: str, result_data: str) -> str:
        """Fallback narrative completion when LLM is not available"""
        try:
            # Try to parse as JSON for better analysis
            if result_data.startswith("{") or result_data.startswith("["):
                data = json.loads(result_data)
                
                if isinstance(data, dict):
                    # Check for common patterns
                    if data.get("status") == "success":
                        if "weather" in tool_name.lower():
                            temp = data.get("data", {}).get("temperature")
                            conditions = data.get("data", {}).get("conditions")
                            if temp and conditions:
                                return f"{initial_narrative}. Got {temp}°F and {conditions}"
                            elif temp:
                                return f"{initial_narrative}. Got temperature {temp}°F"
                            return f"{initial_narrative}. Retrieved weather information"
                        elif "slack" in tool_name.lower():
                            if "messages" in str(data):
                                count = len(data.get("data", {}).get("messages", []))
                                return f"{initial_narrative}. Found {count} relevant messages"
                            elif "channels" in str(data):
                                count = len(data.get("data", {}).get("channels", []))
                                return f"{initial_narrative}. Retrieved {count} channels"
                        return f"{initial_narrative}. Completed successfully"
                    elif "error" in data:
                        return f"{initial_narrative}. Error: {str(data['error'])[:50]}"
                    elif "results" in data:
                        count = len(data["results"]) if isinstance(data["results"], list) else 1
                        return f"{initial_narrative}. Found {count} results"
                    
                elif isinstance(data, list):
                    return f"{initial_narrative}. Retrieved {len(data)} items"
                    
        except:
            pass
        
        # Generic fallback
        return f"{initial_narrative}. Completed operation"
