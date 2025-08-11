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
import yaml
from typing import Dict, Any, Literal, List
from enum import Enum

logger = logging.getLogger(__name__)

class RouteDecision(str, Enum):
    """Route decision options"""
    TIER_1 = "TIER_1"    # Route to Gemini Flash (Fast & Cheap)
    TIER_2 = "TIER_2"    # Route to Claude Sonnet (Advanced & Capable)
    
    # Legacy compatibility
    SIMPLE = "TIER_1"
    COMPLEX = "TIER_2"

class ModelRoutingAgent:
    """
    Fast model routing agent using Gemini 2.5 Flash
    """
    
    def __init__(self, config_path: str = "agents/system prompts/model_routing_agent.yaml"):
        """Initialize the routing agent"""
        self.config = self._load_config(config_path)
        
        # Extract configuration
        self.model_config = self.config.get('model_config', {})
        self.fallback_config = self.config.get('fallback_config', {})
        
        # Configure model settings
        self.model = self.model_config.get('model', 'gemini-2.5-flash')
        self.max_tokens = self.model_config.get('max_tokens', 50)
        self.temperature = self.model_config.get('temperature', 0.0)
        self.timeout = self.model_config.get('timeout', 3)
        
        # Load routing prompt
        self.routing_prompt = self.config.get('routing_prompt', '')
        
        # Load fallback routing patterns
        self.simple_patterns = self.fallback_config.get('simple_patterns', [])
        self.complex_indicators = self.fallback_config.get('complex_indicators', [])
        self.simple_question_starters = self.fallback_config.get('simple_question_starters', [])
        self.max_simple_question_words = self.fallback_config.get('max_simple_question_words', 8)
        self.max_simple_message_words = self.fallback_config.get('max_simple_message_words', 5)
        
        logger.info(f"ModelRoutingAgent initialized with {self.model}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"ROUTING-AGENT: Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"ROUTING-AGENT: Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"ROUTING-AGENT: Error parsing YAML configuration: {e}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"ROUTING-AGENT: Unexpected error loading configuration: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'routing_prompt': 'You are a routing agent. Respond with "simple" or "complex".',
            'model_config': {
                'model': 'gemini-2.5-flash',
                'max_tokens': 50,
                'temperature': 0.0,
                'timeout': 3
            },
            'fallback_config': {
                'simple_patterns': ['hello', 'hi', 'thanks', 'yes', 'no'],
                'complex_indicators': ['how to', 'why', 'explain', 'weather', 'search'],
                'simple_question_starters': ['what is', 'who is'],
                'max_simple_question_words': 8,
                'max_simple_message_words': 5
            }
        }
    
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
                
                response_text = response.text.strip()
                
                # Try to parse JSON response
                try:
                    import json
                    # Extract JSON from response (might have thinking tags)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        parsed_response = json.loads(json_text)
                        
                        tier = parsed_response.get('tier', '').upper()
                        reasoning = parsed_response.get('reasoning', '')
                        
                        if tier == "TIER_1":
                            decision = RouteDecision.TIER_1
                        elif tier == "TIER_2":
                            decision = RouteDecision.TIER_2
                        else:
                            logger.warning(f"ROUTING-AGENT: Invalid tier '{tier}', defaulting to TIER_2")
                            decision = RouteDecision.TIER_2
                        
                        # Log the reasoning for debugging
                        if reasoning:
                            logger.info(f"ROUTING-AGENT: Reasoning - {reasoning}")
                    else:
                        # Fallback to simple text parsing
                        logger.warning("ROUTING-AGENT: No JSON found, trying text parsing")
                        decision_text = response_text.lower()
                        if "tier_1" in decision_text:
                            decision = RouteDecision.TIER_1
                        elif "tier_2" in decision_text:
                            decision = RouteDecision.TIER_2
                        else:
                            logger.warning(f"ROUTING-AGENT: Unexpected response format, defaulting to TIER_2")
                            decision = RouteDecision.TIER_2
                            
                except json.JSONDecodeError:
                    logger.warning("ROUTING-AGENT: Failed to parse JSON, using fallback parsing")
                    decision_text = response_text.lower()
                    if "tier_1" in decision_text:
                        decision = RouteDecision.TIER_1
                    else:
                        decision = RouteDecision.TIER_2
                
                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"ROUTING-AGENT: Routed to {decision.value} in {duration_ms}ms - '{user_message[:50]}...'")
                
                return decision
                
            except Exception as gemini_error:
                logger.warning(f"ROUTING-AGENT: Gemini API failed ({gemini_error}), using fallback")
                return self._fallback_routing(user_message)
            
        except Exception as e:
            logger.error(f"ROUTING-AGENT: Failed to route query: {e}")
            # Default to TIER_2 route when in doubt
            return RouteDecision.TIER_2
    
    def _fallback_routing(self, user_message: str) -> RouteDecision:
        """
        Fallback routing logic using configured patterns
        """
        normalized = user_message.lower().strip()
        
        # Check for simple patterns
        if normalized in self.simple_patterns:
            return RouteDecision.TIER_1
        
        # Short messages that start with simple patterns
        if len(normalized.split()) <= self.max_simple_message_words:
            for pattern in self.simple_patterns:
                if normalized.startswith(pattern):
                    return RouteDecision.TIER_1
        
        # Check for complex indicators
        for indicator in self.complex_indicators:
            if indicator in normalized:
                return RouteDecision.TIER_2
        
        # Simple factual questions
        for starter in self.simple_question_starters:
            if normalized.startswith(starter) and len(normalized.split()) <= self.max_simple_question_words:
                return RouteDecision.TIER_1
        
        # Default to complex for ambiguous cases
        return RouteDecision.TIER_2
