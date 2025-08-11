"""
Gemini Simple Agent

Fast conversational agent using Gemini 2.5 Flash for simple queries.
Handles greetings, basic questions, and casual conversation without tool access.
"""

import logging
import os
import time
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GeminiSimpleAgent:
    """
    Simple conversational agent using Gemini 2.5 Flash
    """
    
    def __init__(self, config_path: str = "agents/system prompts/gemini_simple_agent.yaml"):
        """Initialize the Gemini simple agent"""
        self.config = self._load_config(config_path)
        
        # Extract configuration
        self.model_config = self.config.get('model_config', {})
        self.agent_config = self.config.get('agent_config', {})
        
        # Configure model settings
        self.model = self.model_config.get('model', 'gemini-2.5-flash')
        self.max_tokens = self.model_config.get('max_tokens', 1000)
        self.temperature = self.model_config.get('temperature', 0.7)
        self.timeout = self.model_config.get('timeout', 10)
        
        # Load system prompt and agent responses
        self.system_prompt = self.config.get('system_prompt', '')
        self.response_acknowledgment = self.agent_config.get('response_acknowledgment', 
            "I understand. I'll be conversational, natural, and helpful for simple interactions.")
        self.fallback_message = self.agent_config.get('fallback_message',
            "I'm sorry, I'm not available right now. Please try asking for more complex assistance.")
        
        logger.info(f"GeminiSimpleAgent initialized with {self.model}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"GEMINI-SIMPLE: Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"GEMINI-SIMPLE: Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"GEMINI-SIMPLE: Error parsing YAML configuration: {e}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"GEMINI-SIMPLE: Unexpected error loading configuration: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'system_prompt': 'You are Signal, a friendly and helpful AI assistant for simple conversations.',
            'model_config': {
                'model': 'gemini-2.5-flash',
                'max_tokens': 1000,
                'temperature': 0.7,
                'timeout': 10
            },
            'agent_config': {
                'response_acknowledgment': "I understand. I'll be conversational, natural, and helpful for simple interactions.",
                'fallback_message': "I'm sorry, I'm not available right now. Please try asking for more complex assistance."
            }
        }
    
    async def process_simple_request(self, 
                                   user_message: str, 
                                   conversation_history: str = "",
                                   user_context: str = "",
                                   current_time: str = "") -> Dict[str, Any]:
        """
        Process a simple user request using Gemini Flash
        
        Args:
            user_message: The user's message
            conversation_history: Formatted conversation history
            user_context: User context information
            current_time: Current timestamp
            
        Returns:
            Dict containing response data
        """
        try:
            start_time = time.time()
            
            # Check for Gemini API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "response": self.fallback_message,
                    "message": "Gemini API not configured",
                    "agent": "gemini_simple_fallback"
                }
            
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model)
                
                # Assemble the prompt using the same structure as Claude
                prompt_parts = []
                
                # Add conversation history if available
                if conversation_history:
                    prompt_parts.append(conversation_history)
                    prompt_parts.append("")
                
                # Add user context if available
                if user_context:
                    prompt_parts.append("GENERAL USER CONTEXT")
                    prompt_parts.append(user_context)
                    prompt_parts.append("")
                
                # Add current time
                if current_time:
                    prompt_parts.append("CURRENT TIME")
                    prompt_parts.append(current_time)
                    prompt_parts.append("")
                
                # Add the highlighted current request
                prompt_parts.append("=" * 50)
                prompt_parts.append("🎯 CURRENT USER REQUEST")
                prompt_parts.append("Based on the above context, respond to the following prompt:")
                prompt_parts.append("")
                prompt_parts.append(user_message)
                prompt_parts.append("=" * 50)
                
                full_prompt = "\n".join(prompt_parts)
                
                # Generate response
                response = model.generate_content(
                    [
                        {"role": "user", "parts": [{"text": self.system_prompt}]},
                        {"role": "model", "parts": [{"text": self.response_acknowledgment}]},
                        {"role": "user", "parts": [{"text": full_prompt}]}
                    ],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                )
                
                response_text = response.text.strip()
                
                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"GEMINI-SIMPLE: Generated response in {duration_ms}ms")
                
                return {
                    "success": True,
                    "response": response_text,
                    "message": "Request processed successfully",
                    "agent": "gemini_simple",
                    "model": "gemini-2.5-flash",
                    "processing_time_ms": duration_ms
                }
                
            except Exception as gemini_error:
                logger.error(f"GEMINI-SIMPLE: API error: {gemini_error}")
                return {
                    "success": False,
                    "response": "I'm having trouble right now. Could you try asking for more complex assistance?",
                    "message": f"Gemini API error: {str(gemini_error)}",
                    "agent": "gemini_simple_error"
                }
                
        except Exception as e:
            logger.error(f"GEMINI-SIMPLE: Processing error: {e}")
            return {
                "success": False,
                "response": "I'm sorry, I encountered an error. Please try asking for more complex assistance.",
                "message": f"Processing error: {str(e)}",
                "agent": "gemini_simple_error"
            }
