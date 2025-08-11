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
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local', override=True)

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
        self.max_tokens = model_config.get('max_tokens', 150)
        
        # Extract other configurations
        self.fallback_config = self.config.get('fallback_config', {})
        self.tool_formatting = self.config.get('tool_formatting', {})
    
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
                'max_result_preview': 800
            },
            'tool_formatting': {}
        }
    
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
                # Truncate result data for performance
                max_preview = self.fallback_config.get('max_result_preview', 800)
                truncated_result = result_data[:max_preview]
                
                # Build prompt using YAML-configured system prompt
                prompt = f"""{self.system_prompt}

Tool executed: {tool_name}
Arguments: {json.dumps(tool_args, indent=2) if tool_args else "none"}
Result JSON: {truncated_result}"""
                
                # Use the correct Gemini SDK format
                config = gemini_types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                if streaming_callback:
                    # Streaming mode: send chunks as they arrive
                    narrative_chunks = []
                    
                    def stream_response():
                        model = gemini_client.GenerativeModel(self.model_name)
                        return model.generate_content(
                            prompt,
                            generation_config=config,
                            stream=True
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
                    def generate_response():
                        model = gemini_client.GenerativeModel(self.model_name)
                        return model.generate_content(
                            prompt,
                            generation_config=config
                        )
                    
                    response = await asyncio.to_thread(generate_response)
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
