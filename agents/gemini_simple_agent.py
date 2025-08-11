"""
Gemini Simple Agent

Fast conversational agent using Gemini 2.5 Flash for simple queries.
Handles greetings, basic questions, and casual conversation without tool access.
"""

import logging
import os
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GeminiSimpleAgent:
    """
    Simple conversational agent using Gemini 2.5 Flash
    """
    
    def __init__(self):
        """Initialize the Gemini simple agent"""
        self.model = "gemini-2.5-flash"
        self.max_tokens = 1000
        self.temperature = 0.7  # More conversational
        self.timeout = 10
        
        # Simplified system prompt for conversational interactions
        self.system_prompt = """You are Signal, a friendly and helpful AI assistant.

You handle simple, conversational interactions with users. You are warm, personable, and direct in your responses.

GUIDELINES:
- Be conversational and natural
- Give direct, helpful answers
- Don't overthink simple questions
- Be warm and personable
- Keep responses concise but friendly
- You don't have access to external tools or live data
- For weather, current events, or tool-requiring requests, let users know they need to ask for more complex assistance

RESPONSE STYLE:
- Natural and conversational
- No thinking tags or complex reasoning
- Direct and to the point
- Friendly and helpful tone

Remember: You're handling simple conversations and basic questions. Keep it natural and friendly!"""
        
        logger.info("GeminiSimpleAgent initialized with Gemini 2.5 Flash")
    
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
                    "response": "I'm sorry, I'm not available right now. Please try asking for more complex assistance.",
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
                        {"role": "model", "parts": [{"text": "I understand. I'll be conversational, natural, and helpful for simple interactions."}]},
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
