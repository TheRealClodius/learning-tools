"""
Tool execution result summarizer utility class.

Provides intelligent summarization of tool execution results using various LLM clients
with fallback logic for when LLM is unavailable.
"""

import json
import asyncio
import logging
import yaml
import os
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ExecutionSummarizer:
    """Utility class for tool execution result summarization - accepts external clients"""
    
    def __init__(self, config_path: str = "agents/system prompts/execution_summarizer.yaml", 
                 summarization_config: Dict[str, Any] = None):
        """
        Initialize ExecutionSummarizer with YAML configuration or fallback dict.
        
        Args:
            config_path: Path to YAML configuration file
            summarization_config: Legacy fallback configuration dict
        """
        # Load configuration from YAML file or use fallback
        if summarization_config:
            # Legacy mode: use provided config dict
            self.config = summarization_config
            self.system_prompt = None
        else:
            # New mode: load from YAML file
            self.config = self._load_config(config_path)
            self.system_prompt = self.config.get('system_prompt', '')
        
        # Extract model configuration
        model_config = self.config.get('model_config', {})
        self.model_name = model_config.get('model', 'gemini-2.5-flash')
        self.temperature = model_config.get('temperature', 0.1)
        self.max_tokens = model_config.get('max_tokens', 500)
        
        # Extract other configurations
        self.fallback_config = self.config.get('fallback_config', {})
        self.tool_formatting = self.config.get('tool_formatting', {})
        
        # Token limit for result data
        self.max_result_tokens = self.fallback_config.get('max_result_tokens', 2000)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load ExecutionSummarizer configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"EXECUTION-SUMMARIZER: Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"EXECUTION-SUMMARIZER: Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"EXECUTION-SUMMARIZER: Error parsing YAML configuration: {e}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"EXECUTION-SUMMARIZER: Unexpected error loading configuration: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'system_prompt': 'Create a single-line narrative summary of this tool execution.',
            'model_config': {
                'model': 'gemini-2.5-flash',
                'temperature': 0.1,
                'max_tokens': 150
            },
            'fallback_config': {
                'ensure_emoji_prefix': True,
                'max_result_tokens': 2000
            },
            'tool_formatting': {}
        }
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens using rough estimation"""
        # Rough token estimation: ~4 characters per token for English text
        # This is a conservative estimate that works reasonably well for JSON data
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at character boundary, try to end at a reasonable JSON boundary
        truncated = text[:max_chars]
        
        # Try to end at a complete JSON object/array boundary for cleaner truncation
        for boundary in ['},', ']', '}', '",']:
            last_boundary = truncated.rfind(boundary)
            if last_boundary > max_chars * 0.8:  # Don't go too far back
                return truncated[:last_boundary + len(boundary)]
        
        # If no good boundary found, just truncate at character limit
        return truncated + "..."
    
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
        logger.info(f"EXEC-SUMMARIZER: Starting summary for {tool_name}, model={self.model_name}")
        logger.info(f"EXEC-SUMMARIZER: gemini_client provided: {gemini_client is not None}")
        logger.info(f"EXEC-SUMMARIZER: gemini_types provided: {gemini_types is not None}")
        
        # Special handling for registry tools - no need for expensive Gemini calls
        if tool_name.startswith('reg.') or tool_name.startswith('registry.'):
            logger.info(f"EXEC-SUMMARIZER: Using predefined summary for registry tool {tool_name}")
            return self._generate_registry_summary(tool_name, tool_args, result_data)
        
        if self.model_name.startswith('gemini') and gemini_client and gemini_types:
            try:
                logger.info(f"EXEC-SUMMARIZER: Using Gemini path for {tool_name}")
                # Truncate result data based on token estimate for Gemini processing
                max_tokens = self.max_result_tokens
                truncated_result = self._truncate_to_tokens(result_data, max_tokens)
                
                # Build prompt using YAML-configured system prompt
                prompt = f"""{self.system_prompt}

Tool executed: {tool_name}
Arguments: {json.dumps(tool_args, indent=2) if tool_args else "none"}
Result JSON: {truncated_result}"""
                
                # Use modern google-genai SDK format
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
                    
                    logger.info(f"EXEC-SUMMARIZER: About to call stream_response...")
                    response_stream = await asyncio.to_thread(stream_response)
                    logger.info(f"EXEC-SUMMARIZER: Stream response type: {type(response_stream)}")
                    
                    chunk_count = 0
                    # Process stream directly without nested async
                    for chunk in response_stream:
                        chunk_count += 1
                        logger.info(f"EXEC-SUMMARIZER: Streaming chunk #{chunk_count}, type: {type(chunk)}")
                        logger.info(f"EXEC-SUMMARIZER: Chunk attributes: {dir(chunk)}")
                        
                        # Try different ways to access text content based on new API
                        chunk_text = None
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                        elif hasattr(chunk, 'content') and chunk.content:
                            # Some versions use .content instead of .text
                            chunk_text = chunk.content
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            # Some versions have parts array
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    chunk_text = part.text
                                    break
                        elif hasattr(chunk, 'candidates') and chunk.candidates:
                            # Try candidates structure
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'content') and candidate.content:
                                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if hasattr(part, 'text') and part.text:
                                                chunk_text = part.text
                                                break
                                    elif hasattr(candidate.content, 'text') and candidate.content.text:
                                        chunk_text = candidate.content.text
                                        break
                                if chunk_text:
                                    break
                        
                        if chunk_text:
                            logger.info(f"EXEC-SUMMARIZER: Streaming chunk text: {chunk_text[:50]}...")
                            narrative_chunks.append(chunk_text)
                            
                            # Stream each chunk immediately to UI
                            await streaming_callback(chunk_text, "tool_summary_chunk")
                        else:
                            logger.warning(f"EXEC-SUMMARIZER: Chunk has no text content: {str(chunk)[:100]}...")
                            logger.warning(f"EXEC-SUMMARIZER: Full chunk structure: {chunk}")
                    
                    logger.info(f"EXEC-SUMMARIZER: Finished iterating, received {chunk_count} chunks")
                    # Combine all chunks for final return
                    narrative = "".join(narrative_chunks).strip()
                else:
                    # Non-streaming mode: get complete response
                    def generate_response():
                        return gemini_client.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=config
                        )
                    
                    response = await asyncio.to_thread(generate_response)
                    logger.info(f"EXEC-SUMMARIZER: Gemini response type: {type(response)}")
                    
                    # Check for text in response
                    if hasattr(response, 'text') and response.text:
                        narrative = response.text.strip()
                        logger.info(f"EXEC-SUMMARIZER: Extracted text from response: {narrative[:100]}...")
                    else:
                        # Handle common issues with Gemini responses
                        error_reason = "unknown"
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'finish_reason'):
                                error_reason = str(candidate.finish_reason)
                                
                        logger.warning(f"EXEC-SUMMARIZER: No text from Gemini (reason: {error_reason}), using fallback")
                        # Use fallback instead of trying to parse the response object
                        raise Exception(f"Gemini returned no text (finish_reason: {error_reason})")
                
                # Ensure it starts with the emoji if not present
                if not narrative.startswith("⚡️"):
                    narrative = "⚡️" + narrative
                    
                return narrative
                
            except Exception as e:
                logger.warning(f"EXEC-SUMMARIZER: {self.model_name} narrative generation failed: {e}")
        
        elif self.model_name.startswith('claude') and claude_client:
            # Future: Implement Claude-based summarization
            logger.info(f"EXEC-SUMMARIZER: Claude summarization not yet implemented, using fallback")
        else:
            logger.warning(f"EXEC-SUMMARIZER: No compatible client found for model {self.model_name}, using fallback")
        
        # Fallback using the initial narrative or generate fallback
        logger.info(f"EXEC-SUMMARIZER: Using fallback narrative for {tool_name}")
        return self._fallback_narrative(initial_narrative, tool_name, result_data)
    
    def _generate_registry_summary(self, tool_name: str, tool_args: Dict, result_data: str) -> str:
        """Generate predefined summaries for registry tools - no LLM needed"""
        try:
            # Parse result data to extract useful info
            if result_data.strip().startswith("{") or result_data.strip().startswith("["):
                data = json.loads(result_data)
            else:
                data = {}
            
            # Get query from tool arguments
            query = tool_args.get('query', tool_args.get('tool_name', ''))
            
            if tool_name in ['reg.search', 'registry.search']:
                if query:
                    return f"⚡️Searching for tools for \"{query}\" in the registry"
                else:
                    return f"⚡️Searching for tools in the registry"
                    
            elif tool_name in ['reg.describe', 'registry.describe']:
                tool_to_describe = tool_args.get('tool_name', query)
                if tool_to_describe:
                    return f"⚡️Verifying {tool_to_describe} tool description"
                else:
                    return f"⚡️Verifying tool description"
                    
            elif tool_name in ['reg.list', 'registry.list']:
                return f"⚡️Listing all tools in the registry"
                
            elif tool_name in ['reg.categories', 'registry.categories']:
                return f"⚡️Getting tool categories from registry"
                
            else:
                # Fallback for other registry operations
                operation = tool_name.split('.')[-1] if '.' in tool_name else tool_name
                return f"⚡️Performing {operation} operation on registry"
                
        except Exception as e:
            logger.debug(f"Registry summary generation failed: {e}")
            # Simple fallback
            operation = tool_name.split('.')[-1] if '.' in tool_name else 'registry'
            return f"⚡️Used *{tool_name}* for {operation} operation"
    
    def _fallback_narrative(self, initial_narrative: str, tool_name: str, result_data: str) -> str:
        """Fallback narrative completion when LLM is not available"""
        try:
            # Try to parse as JSON for better analysis
            if result_data.strip().startswith("{") or result_data.strip().startswith("["):
                data = json.loads(result_data)
                
                if isinstance(data, dict):
                    # Enhanced weather tool handling
                    if "weather" in tool_name.lower():
                        if data.get("success") and "data" in data:
                            weather_data = data["data"]
                            # Handle OpenWeatherMap format
                            if "main" in weather_data and "weather" in weather_data:
                                temp = weather_data.get("main", {}).get("temp")
                                weather_desc = weather_data.get("weather", [{}])[0].get("description", "")
                                city = weather_data.get("name", "location")
                                
                                if temp and weather_desc:
                                    return f"⚡️Used *{tool_name}* to get weather for {city}. Currently {temp}°C, {weather_desc}"
                                elif temp:
                                    return f"⚡️Used *{tool_name}* to get weather for {city}. Currently {temp}°C"
                                else:
                                    return f"⚡️Used *{tool_name}* to get weather for {city}. Retrieved current conditions"
                            
                            # Handle search results format
                            elif isinstance(weather_data, list) and weather_data:
                                location = weather_data[0]
                                name = location.get("name", "location")
                                country = location.get("country", "")
                                return f"⚡️Used *{tool_name}* to search locations. Found {name}, {country}"
                        
                        elif data.get("success") == False:
                            error_msg = data.get("message", "unknown error")
                            return f"⚡️Used *{tool_name}* but encountered an issue: {error_msg}"
                    
                    # Handle registry tools
                    elif "reg" in tool_name.lower() and data.get("success"):
                        if "reg.search" in tool_name.lower():
                            tools_data = data.get("data", {}).get("tools", [])
                            count = len(tools_data)
                            return f"⚡️Used *{tool_name}* to search tool registry. Found {count} matching tools"
                        elif "reg.describe" in tool_name.lower():
                            tool_data = data.get("data", {}).get("tool", {})
                            tool_display_name = tool_data.get("display_name", "tool")
                            return f"⚡️Used *{tool_name}* to get details for {tool_display_name}"
                    
                    # Handle other tool types
                    elif data.get("success"):
                        if "slack" in tool_name.lower():
                            if "messages" in str(data):
                                count = len(data.get("data", {}).get("messages", []))
                                return f"⚡️Used *{tool_name}* to search messages. Found {count} relevant messages"
                            elif "channels" in str(data):
                                count = len(data.get("data", {}).get("channels", []))
                                return f"⚡️Used *{tool_name}* to get channels. Retrieved {count} channels"
                        return f"⚡️Used *{tool_name}* successfully. Retrieved requested data"
                    
                    elif "error" in data:
                        return f"⚡️Used *{tool_name}* but got error: {str(data['error'])[:50]}"
                    elif "results" in data:
                        count = len(data["results"]) if isinstance(data["results"], list) else 1
                        return f"⚡️Used *{tool_name}* successfully. Found {count} results"
                    
                elif isinstance(data, list):
                    return f"⚡️Used *{tool_name}* successfully. Retrieved {len(data)} items"
                    
        except Exception as e:
            logger.debug(f"Fallback parsing failed for {tool_name}: {e}")
        
        # Ensure we always have a meaningful response
        ensure_emoji = self.fallback_config.get('ensure_emoji_prefix', True)
        
        if initial_narrative and len(initial_narrative.strip()) > 10:
            fallback = f"{initial_narrative}. Operation completed"
        else:
            fallback = f"⚡️Used *{tool_name}* - completed successfully"
        
        # Ensure emoji prefix if configured
        if ensure_emoji and not fallback.startswith("⚡️"):
            fallback = "⚡️" + fallback
            
        return fallback
