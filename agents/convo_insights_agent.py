"""
Conversation Insights Agent

Stateful agent that builds evolving understanding of users across two dimensions:
- PINS: User preferences, needs, constraints, conversation context
- RECOMMENDATIONS: Tool usage guidance, error patterns, workflow optimization

This agent maintains user-specific insights in the local buffer and evolves them over time.
"""

import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger(__name__)

class ConvoInsightsAgent:
    """
    Stateful Conversation Insights Agent that maintains evolving user understanding
    """
    
    def __init__(self, config_path: str = "agents/system prompts/convo_insights_agent.yaml"):
        """Initialize the Conversation Insights Agent"""
        self.config = self._load_config(config_path)
        
        # Extract configuration
        self.model_config = self.config.get('model_config', {})
        self.analysis_config = self.config.get('analysis_config', {})
        self.prompt_templates = self.config.get('prompt_templates', {})
        self.logging_config = self.config.get('logging', {})
        
        # Configure model settings
        self.model = self.model_config.get('model', 'gemini-2.5-flash')
        self.max_tokens = self.model_config.get('max_tokens', 1000)
        self.temperature = self.model_config.get('temperature', 0.3)
        self.timeout = self.model_config.get('timeout', 15)
        
        logger.info(f"{self._log_prefix()}: Initialized with model {self.model}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"INSIGHTS-AGENT: Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"INSIGHTS-AGENT: Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"INSIGHTS-AGENT: Error parsing YAML configuration: {e}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"INSIGHTS-AGENT: Unexpected error loading configuration: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            'model_config': {
                'model': 'gemini-2.5-flash',
                'max_tokens': 1000,
                'temperature': 0.3,
                'timeout': 15
            },
            'analysis_config': {
                'min_message_length': 10,
                'require_tools_for_short_messages': True,
                'max_pins_per_user': 3,
                'max_recommendations_per_user': 3,
                'pin_relevance_threshold': 0.1,
                'recommendation_relevance_threshold': 0.05,
                'max_context_length': 800,
                'max_pins_in_prompt': 2,
                'max_recommendations_in_prompt': 1
            },
            'prompt_templates': {
                'empty_insights': "PINS:\n• No previous conversation insights.\n\nRECOMMENDATIONS:\n• No previous tool usage patterns."
            },
            'logging': {
                'prefix': 'INSIGHTS-AGENT',
                'debug_prompts': False,
                'debug_responses': True
            }
        }
    
    def _log_prefix(self) -> str:
        """Get logging prefix from config"""
        return self.logging_config.get('prefix', 'INSIGHTS-AGENT')
    
    async def analyze_interaction(self, user_message: str, agent_response: str, 
                                tool_usage_log: List[Dict], thinking_content: List[str], 
                                user_id: str, user_buffers: Dict[str, Dict[str, Any]]) -> bool:
        """
        Analyze user interaction and update insights in the provided buffer
        
        Args:
            user_message: User's input message
            agent_response: Agent's response
            tool_usage_log: List of tool usage records
            thinking_content: Agent's thinking process
            user_id: User identifier
            user_buffers: Reference to client agent's user buffers
            
        Returns:
            bool: True if insights were updated, False if skipped
        """
        try:
            logger.info(f"{self._log_prefix()}: Analyzing interaction for user {user_id}")
            
            # Check if we should skip this interaction
            if self._should_skip_interaction(user_message, tool_usage_log):
                logger.info(f"{self._log_prefix()}: Skipping trivial interaction for user {user_id}")
                return False
            
            # Get existing insights for comparison and evolution
            existing_insights = self._get_current_insights(user_id, user_buffers)
            insights_count = len(existing_insights.split('•')) - 1 if existing_insights != self._get_empty_insights() else 0
            logger.info(f"{self._log_prefix()}: Found {insights_count} existing insights")
            
            # Generate updated insights using stateful analysis
            updated_insights = await self._generate_updated_insights(
                user_message, agent_response, tool_usage_log, 
                thinking_content, existing_insights, user_id
            )
            
            # Store the evolved understanding
            self._replace_insights(user_id, updated_insights, user_buffers)
            
            logger.info(f"{self._log_prefix()}: Updated insights for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"{self._log_prefix()}: Failed to analyze interaction for user {user_id}: {e}")
            import traceback
            logger.error(f"{self._log_prefix()}: Traceback: {traceback.format_exc()}")
            return False
    
    def _should_skip_interaction(self, user_message: str, tool_usage_log: List[Dict]) -> bool:
        """Determine if this interaction should be skipped for insights generation"""
        min_length = self.analysis_config.get('min_message_length', 10)
        require_tools = self.analysis_config.get('require_tools_for_short_messages', True)
        
        # Skip if message is too short and no tools were used
        if not tool_usage_log and len(user_message.split()) < min_length:
            return True
        
        return False
    
    async def _generate_updated_insights(self, user_message: str, agent_response: str, 
                                       tool_usage_log: List[Dict], thinking_content: List[str],
                                       existing_insights: str, user_id: str) -> str:
        """Generate updated insights using Gemini API or fallback"""
        
        # Prepare execution details
        execution_summary = ""
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            tool_failures = [tool['tool'] for tool in tool_usage_log if not tool['success']]
            execution_summary = f"Tools used: {', '.join(tools_used)}. "
            if tool_failures:
                execution_summary += f"Failed tools: {', '.join(tool_failures)}. "
        else:
            execution_summary = "No tools used"
        
        # Prepare thinking insights
        thinking_summary = ""
        if thinking_content:
            # Extract key insights from thinking (first 200 chars of each thinking block)
            key_thoughts = [thinking[:200] for thinking in thinking_content[:2]]
            thinking_summary = " ".join(key_thoughts)
        else:
            thinking_summary = "No detailed reasoning captured"
        
        # Try Gemini API first
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning(f"{self._log_prefix()}: GEMINI_API_KEY not set, using fallback analysis")
            return self._generate_stateful_fallback_insights(
                user_message, agent_response, tool_usage_log, thinking_content, existing_insights, user_id
            )
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model)
            
            # Use template from config
            prompt_template = self.prompt_templates.get('main_analysis', '')
            if not prompt_template:
                logger.warning(f"{self._log_prefix()}: No main_analysis template found, using fallback")
                return self._generate_stateful_fallback_insights(
                    user_message, agent_response, tool_usage_log, thinking_content, existing_insights, user_id
                )
            
            prompt = prompt_template.format(
                existing_insights=existing_insights,
                user_message=user_message,
                agent_response=agent_response,
                execution_summary=execution_summary,
                thinking_summary=thinking_summary
            )
            
            if self.logging_config.get('debug_prompts', False):
                logger.debug(f"{self._log_prefix()}: Prompt: {prompt}")
            
            response = model.generate_content(prompt, timeout=self.timeout)
            updated_insights = response.text.strip()
            
            if self.logging_config.get('debug_responses', True):
                logger.info(f"{self._log_prefix()}: Generated stateful insights for user {user_id}")
            
            return updated_insights
            
        except Exception as gemini_error:
            logger.warning(f"{self._log_prefix()}: Gemini API failed ({gemini_error}), using fallback")
            return self._generate_stateful_fallback_insights(
                user_message, agent_response, tool_usage_log, thinking_content, existing_insights, user_id
            )
    
    def _generate_stateful_fallback_insights(self, user_message: str, agent_response: str, 
                                           tool_usage_log: List[Dict], thinking_content: List[str], 
                                           existing_insights: str, user_id: str) -> str:
        """Generate stateful insights without using Gemini API - compares against existing insights"""
        logger.info(f"{self._log_prefix()}: Using stateful fallback analysis for user {user_id}")
        
        # Parse existing insights to understand what we already know
        existing_patterns = set()
        existing_pins = []
        existing_recommendations = []
        
        # Parse existing insights by section
        if "PINS:" in existing_insights:
            sections = existing_insights.split('\n\n')
            for section in sections:
                if section.strip().startswith('PINS:'):
                    existing_pins = [line.strip() for line in section.split('\n')[1:] if line.strip().startswith('•')]
                elif section.strip().startswith('RECOMMENDATIONS:'):
                    existing_recommendations = [line.strip() for line in section.split('\n')[1:] if line.strip().startswith('•')]
            
            # Extract patterns from existing insights
            all_existing_text = ' '.join(existing_pins + existing_recommendations).lower()
            if 'personal' in all_existing_text or 'preference' in all_existing_text:
                existing_patterns.add('personal_info')
            if 'requirement' in all_existing_text or 'need' in all_existing_text:
                existing_patterns.add('user_requirements')
            if 'system' in all_existing_text or 'tool' in all_existing_text or 'error' in all_existing_text:
                existing_patterns.add('system_issues')
            if 'memory' in all_existing_text or 'context' in all_existing_text:
                existing_patterns.add('memory_operations')
        
        # Analyze current interaction for new patterns
        new_pins = []
        new_recommendations = []
        user_lower = user_message.lower()
        
        # === PINS ANALYSIS ===
        # Personal information or preferences (only if not already captured)
        if 'personal_info' not in existing_patterns:
            if any(word in user_lower for word in ['my name is', 'i am', 'call me', 'i prefer']):
                new_pins.append(f"• **Personal Info**: User shared: {user_message[:100]}...")
        
        # User requirements (check if it's a new/different requirement)
        if any(word in user_lower for word in ['need', 'want', 'require', 'must', 'should']):
            new_pins.append(f"• **User Requirement**: {user_message[:80]}...")
        
        # System investigations 
        if any(word in user_lower for word in ['prompt', 'memory', 'context', 'assembly', 'debug', 'issue']):
            new_pins.append(f"• **System Inquiry**: User investigating {user_message[:60]}...")
        
        # === RECOMMENDATIONS ANALYSIS ===
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            failed_tools = [(tool['tool'], tool.get('result_preview', '')) for tool in tool_usage_log if not tool['success']]
            successful_tools = [tool['tool'] for tool in tool_usage_log if tool['success']]
            
            # Analyze tool failures with nuance
            for tool_name, error_preview in failed_tools[:2]:  # Max 2 failure analyses
                error_lower = error_preview.lower()
                
                # Classify the type of failure
                if any(phrase in error_lower for phrase in ['not found', 'no results', 'empty']):
                    new_recommendations.append(f"• **Query Refinement**: {tool_name} found no results - try broader or different search terms")
                elif any(phrase in error_lower for phrase in ['timeout', 'network', 'connection']):
                    new_recommendations.append(f"• **Retry Strategy**: {tool_name} had connection issues - retry with shorter timeout or different timing")
                elif any(phrase in error_lower for phrase in ['api key', 'unauthorized', 'forbidden']):
                    new_recommendations.append(f"• **Configuration**: {tool_name} has authentication issues - check API key setup")
                elif any(phrase in error_lower for phrase in ['rate limit', 'quota', 'too many']):
                    new_recommendations.append(f"• **Rate Limiting**: {tool_name} hit limits - space out requests or use alternative tools")
                elif any(phrase in error_lower for phrase in ['invalid', 'bad request', 'malformed']):
                    new_recommendations.append(f"• **Parameter Fix**: {tool_name} received invalid parameters - review argument format and required fields")
                else:
                    # Generic failure - suggest investigation
                    new_recommendations.append(f"• **Investigate**: {tool_name} failed unexpectedly - check error logs and consider alternative approaches")
            
            # Successful tool patterns with context
            if successful_tools:
                # Group similar tools
                memory_tools = [tool for tool in successful_tools if 'memory' in tool]
                search_tools = [tool for tool in successful_tools if any(term in tool for term in ['search', 'perplexity', 'reg_'])]
                weather_tools = [tool for tool in successful_tools if 'weather' in tool]
                
                if memory_tools:
                    new_recommendations.append(f"• **Memory Pattern**: User benefits from {', '.join(memory_tools[:2])} for context retrieval")
                
                if search_tools:
                    new_recommendations.append(f"• **Discovery Pattern**: User effectively uses {', '.join(search_tools[:2])} for finding information")
                
                if weather_tools:
                    new_recommendations.append(f"• **Weather Pattern**: User regularly needs {', '.join(weather_tools[:2])} for location-based data")
        
        # Agent response analysis for recommendations
        response_lower = agent_response.lower()
        if any(phrase in response_lower for phrase in ['sorry', 'error', 'unable', 'cannot', 'failed']):
            new_recommendations.append(f"• **Retry Strategy**: Current approach failed, consider alternative tools/methods")
        
        # Preserve existing insights and merge intelligently
        final_pins = []
        final_recommendations = []
        
        # Add new insights first (prioritize fresh learnings)
        max_new_pins = self.analysis_config.get('max_pins_per_user', 3) // 2
        max_new_recs = self.analysis_config.get('max_recommendations_per_user', 3) // 2
        
        final_pins.extend(new_pins[:max_new_pins])
        final_recommendations.extend(new_recommendations[:max_new_recs])
        
        # Add most relevant existing insights if we have room
        max_pins = self.analysis_config.get('max_pins_per_user', 3)
        max_recs = self.analysis_config.get('max_recommendations_per_user', 3)
        
        remaining_pin_slots = max_pins - len(final_pins)
        remaining_rec_slots = max_recs - len(final_recommendations)
        
        if remaining_pin_slots > 0 and existing_pins:
            valid_pins = [pin for pin in existing_pins if "No previous conversation insights" not in pin]
            final_pins.extend(valid_pins[:remaining_pin_slots])
        
        if remaining_rec_slots > 0 and existing_recommendations:
            valid_recs = [rec for rec in existing_recommendations if "No previous tool usage patterns" not in rec]
            final_recommendations.extend(valid_recs[:remaining_rec_slots])
        
        # Default insights if nothing meaningful found
        if not final_pins:
            if len(tool_usage_log) > 0:
                final_pins.append(f"• **Interaction**: User query requiring {len(tool_usage_log)} tool operations")
            else:
                final_pins.append(f"• **Simple Exchange**: Direct Q&A about {user_message[:40]}...")
        
        if not final_recommendations:
            if len(tool_usage_log) > 0:
                final_recommendations.append(f"• **Tool Pattern**: User interaction involved {len(tool_usage_log)} tools")
            else:
                final_recommendations.append(f"• **Direct Response**: No tools needed for this user query")
        
        # Format in two sections
        pins_section = "PINS:\n" + "\n".join(final_pins)
        recommendations_section = "RECOMMENDATIONS:\n" + "\n".join(final_recommendations)
        
        return f"{pins_section}\n\n{recommendations_section}"
    
    def _get_current_insights(self, user_id: str, user_buffers: Dict[str, Dict[str, Any]]) -> str:
        """Get formatted previous insights for the agent to review"""
        if user_id not in user_buffers:
            return self._get_empty_insights()
        
        important_items = user_buffers[user_id].get('important', {})
        if not important_items:
            return self._get_empty_insights()
        
        # Separate insights by type and sort by recency
        pins = []
        recommendations = []
        
        sorted_items = sorted(
            important_items.items(), 
            key=lambda x: x[1]['timestamp'], 
            reverse=True
        )
        
        for note_id, note_data in sorted_items:
            timestamp = note_data.get('timestamp', 0)
            age_minutes = (time.time() - timestamp) / 60
            insight_type = note_data.get('insight_type', 'pin')  # Default to pin for backward compatibility
            notes = note_data.get('notes', '')
            
            formatted_insight = f"• {notes} (noted {age_minutes:.0f}m ago)"
            
            if insight_type == 'recommendation':
                recommendations.append(formatted_insight)
            else:
                pins.append(formatted_insight)
        
        # Format in two sections
        pins_section = "PINS:\n" + ("\n".join(pins) if pins else "• No previous conversation insights.")
        recommendations_section = "RECOMMENDATIONS:\n" + ("\n".join(recommendations) if recommendations else "• No previous tool usage patterns.")
        
        return f"{pins_section}\n\n{recommendations_section}"
    
    def _get_empty_insights(self) -> str:
        """Get empty insights template"""
        return self.prompt_templates.get('empty_insights', 
            "PINS:\n• No previous conversation insights.\n\nRECOMMENDATIONS:\n• No previous tool usage patterns.")
    
    def _replace_insights(self, user_id: str, updated_insights: str, user_buffers: Dict[str, Dict[str, Any]]):
        """Replace entire insights set with evolved understanding (two-section format)"""
        if user_id not in user_buffers:
            return
        
        # Clear existing insights
        user_buffers[user_id]['important'] = {}
        
        # Handle case where agent returns no insights
        if not updated_insights or "No significant insights yet" in updated_insights:
            logger.info(f"{self._log_prefix()}: No significant insights to store for user {user_id}")
            return
        
        # Parse two-section format
        sections = updated_insights.split('\n\n')
        pins_section = ""
        recommendations_section = ""
        
        for section in sections:
            if section.strip().startswith('PINS:'):
                pins_section = section.strip()
            elif section.strip().startswith('RECOMMENDATIONS:'):
                recommendations_section = section.strip()
        
        insights_count = 0
        max_pins = self.analysis_config.get('max_pins_per_user', 3)
        max_recs = self.analysis_config.get('max_recommendations_per_user', 3)
        
        # Process PINS section
        if pins_section:
            pin_lines = [line.strip() for line in pins_section.split('\n') if line.strip().startswith('•')]
            for i, pin_line in enumerate(pin_lines[:max_pins]):
                pin_text = pin_line[1:].strip()  # Remove '•' and whitespace
                
                if pin_text and "No significant conversation insights" not in pin_text:
                    note_id = f"pin_{uuid.uuid4().hex[:8]}"
                    user_buffers[user_id]['important'][note_id] = {
                        'notes': pin_text,
                        'timestamp': time.time(),
                        'insight_type': 'pin',
                        'related_question': f"Conversation insight #{i+1}"
                    }
                    insights_count += 1
        
        # Process RECOMMENDATIONS section
        if recommendations_section:
            rec_lines = [line.strip() for line in recommendations_section.split('\n') if line.strip().startswith('•')]
            for i, rec_line in enumerate(rec_lines[:max_recs]):
                rec_text = rec_line[1:].strip()  # Remove '•' and whitespace
                
                if rec_text and "No tool usage patterns" not in rec_text:
                    note_id = f"rec_{uuid.uuid4().hex[:8]}"
                    user_buffers[user_id]['important'][note_id] = {
                        'notes': rec_text,
                        'timestamp': time.time(),
                        'insight_type': 'recommendation',
                        'related_question': f"Tool recommendation #{i+1}"
                    }
                    insights_count += 1
        
        pins_count = len([item for item in user_buffers[user_id]['important'].values() 
                         if item.get('insight_type') == 'pin'])
        recs_count = len([item for item in user_buffers[user_id]['important'].values() 
                         if item.get('insight_type') == 'recommendation'])
        
        logger.info(f"{self._log_prefix()}: Stored {pins_count} pins + {recs_count} recommendations for user {user_id}")