"""
Model Routing Agent

Super-fast agent that determines whether a user query should be handled by:
- Gemini 2.5 Flash (simple, conversational queries)
- Claude 3.5 Sonnet (complex, tool-requiring queries)

Uses Gemini 2.5 Flash for extremely fast routing decisions.
"""

import logging
import os
import time
from typing import Dict, Any, Literal
from enum import Enum

logger = logging.getLogger(__name__)

class RouteDecision(str, Enum):
    """Route decision options"""
    SIMPLE = "simple"    # Route to Gemini Flash
    COMPLEX = "complex"  # Route to Claude Sonnet

class ModelRoutingAgent:
    """
    Fast model routing agent using Gemini 2.5 Flash
    """
    
    def __init__(self):
        """Initialize the routing agent"""
        self.model = "gemini-2.5-flash"
        self.max_tokens = 50  # Very short responses
        self.temperature = 0.0  # Deterministic routing
        self.timeout = 3  # Super fast timeout
        
        # Routing system prompt
        self.routing_prompt = """You are a routing agent that decides which AI model should handle a user query.

RESPOND WITH EXACTLY ONE WORD:

"simple" - Use Gemini Flash for:
- Greetings, pleasantries, basic chitchat
- Simple questions with direct answers
- Thank you messages, acknowledgments
- Basic factual questions (what is X, who is Y)
- Casual conversation, small talk
- Single-step requests

"complex" - Use Claude Sonnet for:
- Multi-step problem solving
- Requests requiring tool usage (weather, search, etc.)
- Analysis, comparisons, explanations
- Planning, strategy, troubleshooting
- Code review, debugging, optimization
- Research requiring external information
- Questions needing reasoning or synthesis

Analyze this user query and respond with ONLY "simple" or "complex":"""
        
        logger.info("ModelRoutingAgent initialized with Gemini 2.5 Flash")
    
    async def route_query(self, user_message: str) -> RouteDecision:
        """
        Route a user query to the appropriate model
        
        Args:
            user_message: The user's query to route
            
        Returns:
            RouteDecision indicating which model to use
        """
        try:
            start_time = time.time()
            
            # Try Gemini API first
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("ROUTING-AGENT: GEMINI_API_KEY not set, using fallback")
                return self._fallback_routing(user_message)
            
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model)
                
                # Construct routing prompt
                full_prompt = f"{self.routing_prompt}\n\nUser query: {user_message}"
                
                # Generate routing decision
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                )
                
                decision_text = response.text.strip().lower()
                
                # Parse the decision
                if "simple" in decision_text:
                    decision = RouteDecision.SIMPLE
                elif "complex" in decision_text:
                    decision = RouteDecision.COMPLEX
                else:
                    logger.warning(f"ROUTING-AGENT: Unexpected response '{decision_text}', defaulting to complex")
                    decision = RouteDecision.COMPLEX
                
                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"ROUTING-AGENT: Routed to {decision.value} in {duration_ms}ms - '{user_message[:50]}...'")
                
                return decision
                
            except Exception as gemini_error:
                logger.warning(f"ROUTING-AGENT: Gemini API failed ({gemini_error}), using fallback")
                return self._fallback_routing(user_message)
            
        except Exception as e:
            logger.error(f"ROUTING-AGENT: Failed to route query: {e}")
            # Default to complex route when in doubt
            return RouteDecision.COMPLEX
    
    def _fallback_routing(self, user_message: str) -> RouteDecision:
        """
        Fallback routing logic using pattern matching (similar to current _needs_reasoning)
        """
        normalized = user_message.lower().strip()
        
        # Simple patterns that should go to Gemini Flash
        simple_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
            'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure',
            'who are you', 'what are you', 'your name', 'what can you do',
            'goodbye', 'bye', 'see you'
        ]
        
        # Check for simple patterns
        if normalized in simple_patterns:
            return RouteDecision.SIMPLE
        
        # Short messages (≤5 words) that start with simple patterns
        if len(normalized.split()) <= 5:
            for pattern in simple_patterns:
                if normalized.startswith(pattern):
                    return RouteDecision.SIMPLE
        
        # Complex indicators that need Claude
        complex_indicators = [
            'how to', 'why', 'explain', 'analyze', 'compare', 'calculate',
            'plan', 'strategy', 'solve', 'debug', 'fix', 'troubleshoot',
            'weather', 'search', 'find', 'research', 'lookup',
            'tools', 'execute', 'run', 'install', 'configure'
        ]
        
        # Check for complex indicators
        for indicator in complex_indicators:
            if indicator in normalized:
                return RouteDecision.COMPLEX
        
        # Simple factual questions
        if normalized.startswith(('what is', 'who is', 'where is')) and len(normalized.split()) <= 8:
            return RouteDecision.SIMPLE
        
        # Default to complex for ambiguous cases
        return RouteDecision.COMPLEX
