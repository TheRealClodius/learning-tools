"""
Simple Client Agent using Anthropic Claude API

This agent provides a basic interface for processing user requests using
Anthropic's Claude language model with tool discovery capabilities and thinking streaming.
"""

import anthropic
import logging
import os
import time
import json
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv
import re
import yaml

from runtime.tool_executor import ToolExecutor
# COMMENTED OUT: from agents.convo_insights_agent import ConvoInsightsAgent
from agents.model_routing_agent import ModelRoutingAgent, RouteDecision
from agents.gemini_simple_agent import GeminiSimpleAgent
from runtime.rate_limit_handler import (
    RateLimitHandler, RateLimitConfig, with_rate_limit
)
from tools.memory_mcp import close_mcp_client  # Import cleanup function
from agents.execution_summarizer import ExecutionSummarizer

# Load environment variables from .env.local file
load_dotenv('.env.local', override=True)

logger = logging.getLogger(__name__)

class ClientAgent:
    """
    Simple client agent that uses Gemini API for natural language processing
    and integrates with the tool registry for dynamic tool discovery.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client agent with Claude API"""
        # Setup Claude API
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Claude client with Prompt Caching beta header
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            default_headers={
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
        )
        
        # Initialize rate limit handler with configuration from YAML
        from runtime.rate_limit_handler import load_config_from_yaml
        tier = os.getenv('ANTHROPIC_TIER', 'default')
        rate_limit_config = load_config_from_yaml(tier)
        self.rate_limit_handler = RateLimitHandler(rate_limit_config)
        logger.info(f"Initialized rate limiter for tier: {tier} with {rate_limit_config.tokens_per_minute} tokens/min")
        
        # Load configuration from YAML file
        self.config = self._load_config()
        
        # Model configuration
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = self.config.get('model_config', {}).get('max_tokens', 4096)
        self.temperature = self.config.get('model_config', {}).get('temperature', 0.7)
        self.max_iterations = self.config.get('model_config', {}).get('max_iterations', 50)
        
        # Tool summarization configuration
        self.summarization_config = self.config.get('summarization_config', {})
        self.summarization_enabled = self.summarization_config.get('enabled', True)
        
        # Initialize summarization client based on config
        self.summarizer = None
        if self.summarization_enabled:
            self._init_summarization_client()
            # Initialize ExecutionSummarizer utility class
            self.execution_summarizer = ExecutionSummarizer(self.summarization_config)
        else:
            self.execution_summarizer = None

        
        # Define function schemas for Claude - use underscore format per Anthropic API constraints
        self.tools = [
            {
                "name": "reg_search",
                "description": "Search for tools in the registry using semantic queries, filters, and advanced search criteria",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why you're searching"
                        },
                        "query": {
                            "type": "string", 
                            "description": "Search terms to find relevant tools"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["description", "capabilities", "tags", "use_cases"],
                            "description": "Type of search to perform (defaults to description)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (defaults to 10)"
                        }
                    },
                    "required": ["explanation", "query"]
                }
            },
            {
                "name": "reg_describe", 
                "description": "Get detailed information about a specific tool including its schema and usage",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why you need tool details"
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "The name of the tool (format: service.action)"
                        },
                        "include_schema": {
                            "type": "boolean",
                            "description": "Include input/output schemas in response (defaults to true)"
                        }
                    },
                    "required": ["explanation", "tool_name"]
                }
            },
            {
                "name": "reg_list",
                "description": "List all available tools with optional filtering and pagination",
                "input_schema": {
                    "type": "object", 
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why you need the tool list"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tools to return (defaults to 50)"
                        }
                    },
                    "required": ["explanation"]
                }
            },
            {
                "name": "reg_categories",
                "description": "Get all available tool categories with their descriptions and metadata",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why you need the categories"
                        }
                    },
                    "required": ["explanation"]
                }
            },


            {
                "name": "execute_tool",
                "description": "Execute any discovered tool dynamically. Use this after discovering tools through registry search to actually run weather, perplexity, or other tools.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why you're executing this tool"
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "The name of the tool to execute (e.g., 'weather.search', 'perplexity.search')"
                        },
                        "tool_args": {
                            "type": "object",
                            "description": "Arguments to pass to the tool - refer to the tool's schema from reg_describe"
                        }
                    },
                    "required": ["explanation", "tool_name", "tool_args"]
                }
            },
            {
                "name": "memory_add",
                "description": "Add conversation to the memory system for persistence across interactions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The user's input message to store"
                        },
                        "agent_response": {
                            "type": "string", 
                            "description": "The agent's response message to store"
                        }
                    },
                    "required": ["user_input", "agent_response"]
                }
            },
            {
                "name": "memory_retrieve",
                "description": "Retrieve relevant conversation history and context from memory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search for relevant memories and context"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        # Track background tasks for cleanup
        self._background_tasks: set[asyncio.Task] = set()
        
        # Initialize tool executor for dynamic tool discovery
        self.tool_executor = ToolExecutor()
        
        # COMMENTED OUT: Initialize conversation insights agent
        # self.insights_agent = ConvoInsightsAgent()
        
        # Initialize routing agents
        self.routing_agent = ModelRoutingAgent()
        self.gemini_simple_agent = GeminiSimpleAgent()
        
        # Enhanced user buffers for prompt assembly
        self.user_buffers: Dict[str, Dict[str, Any]] = {}
        self.buffer_expiry_minutes = 30  # Buffer expires after 30 minutes
        
        # Load system prompt from YAML configuration
        self.system_prompt = self.config.get('system_prompt', '')
        
        logger.info("ClientAgent initialized with Claude + Gemini routing system")
    
    def _init_summarization_client(self):
        """Initialize summarization client based on configuration"""
        model_name = self.summarization_config.get('model', 'gemini-2.5-flash')
        
        try:
            if model_name.startswith('gemini'):
                # Initialize Gemini client
                from google import genai
                from google.genai import types
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    self.gemini_client = genai.Client(api_key=api_key)
                    self.gemini_types = types
                    self.summarizer_model = model_name
                    logger.info(f"Initialized {model_name} for tool summarization")
                else:
                    logger.warning("GEMINI_API_KEY not found - tool summarization disabled")
                    self.summarization_enabled = False
            elif model_name.startswith('claude'):
                # Future: Initialize Claude client for summarization
                # For now, use existing self.client
                self.summarizer_model = model_name
                logger.info(f"Configured {model_name} for tool summarization (using existing Claude client)")
            else:
                logger.warning(f"Unknown summarization model: {model_name} - tool summarization disabled")
                self.summarization_enabled = False
                
        except ImportError as e:
            logger.warning(f"Failed to import summarization dependencies: {e} - tool summarization disabled")
            self.summarization_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize summarization client: {e} - tool summarization disabled")
            self.summarization_enabled = False
    
    def _generate_tool_start_message(self, tool_name: str, tool_args: Dict) -> str:
        """Generate a 'working' message from tool name and arguments"""
        
        if tool_name.startswith("reg_"):
            # Registry tools
            if "search" in tool_name:
                query = tool_args.get("query", "tools")
                return f"🔄 Searching registry for '{query}'..."
            elif "describe" in tool_name:
                tool = tool_args.get("tool_name", "tool")
                return f"🔄 Getting details for {tool}..."
            elif "list" in tool_name:
                return f"🔄 Listing available tools..."
            elif "categories" in tool_name:
                return f"🔄 Getting tool categories..."
            else:
                return f"🔄 Using registry tool..."
                
        elif tool_name.startswith("memory"):
            # Memory tools
            if "retrieve" in tool_name:
                query = tool_args.get("query", "information")
                return f"🔄 Searching memory for '{query}'..."
            elif "add" in tool_name:
                return f"🔄 Saving conversation to memory..."
            else:
                return f"🔄 Using memory tool..."
                
        else:
            # Execute tools - extract from tool_args
            if "location" in tool_args or "city" in tool_args:
                location = tool_args.get("location") or tool_args.get("city")
                service = tool_name.split(".")[0] if "." in tool_name else tool_name
                return f"🔄 Getting {service} info for {location}..."
            elif "query" in tool_args:
                query = tool_args["query"]
                service = tool_name.split(".")[0] if "." in tool_name else tool_name
                query_preview = query[:30] + "..." if len(query) > 30 else query
                return f"🔄 Searching {service} for '{query_preview}'..."
            elif "message" in tool_args or "text" in tool_args:
                service = tool_name.split(".")[0] if "." in tool_name else tool_name
                return f"🔄 Processing {service} request..."
            else:
                service = tool_name.split(".")[0] if "." in tool_name else tool_name
                return f"🔄 Running {service}..."

    async def _generate_tool_summary_streaming(self, tool_name: str, tool_args: Dict, result: Any, streaming_callback) -> str:
        """Generate tool summary with real-time streaming to UI"""
        if not self.summarization_enabled or not self.execution_summarizer:
            fallback = f"⚡️Used *{tool_name}* - summarization disabled"
            await streaming_callback(fallback, "tool_result")
            return fallback
        
        # Convert result to string for summarization
        if isinstance(result, dict):
            result_data = json.dumps(result, indent=2)
        else:
            result_data = str(result) if result is not None else "No result returned"
        
        try:
            # Use ExecutionSummarizer with streaming callback
            summary = await self.execution_summarizer.create_narrative_summary(
                tool_name=tool_name,
                tool_args=tool_args,
                result_data=result_data,
                initial_narrative="",
                gemini_client=getattr(self, 'gemini_client', None),
                gemini_types=getattr(self, 'gemini_types', None),
                claude_client=getattr(self, 'client', None) if self.summarizer_model.startswith('claude') else None,
                streaming_callback=streaming_callback  # Pass streaming callback to Gemini
            )
            return summary
        except Exception as e:
            logger.warning(f"Tool summarization failed: {e}")
            # Fallback to simple summary
            fallback = f"⚡️Used *{tool_name}* - completed successfully"
            await streaming_callback(fallback, "tool_result")
            return fallback

    async def _generate_tool_summary(self, tool_name: str, tool_args: Dict, result: Any) -> str:
        """Generate tool summary using configured LLM client"""
        if not self.summarization_enabled or not self.execution_summarizer:
            return f"⚡️Used *{tool_name}* - summarization disabled"
        
        # Convert result to string for summarization
        if isinstance(result, dict):
            result_data = json.dumps(result, indent=2)
        else:
            result_data = str(result) if result is not None else "No result returned"
        
        try:
            # Use ExecutionSummarizer with our initialized clients
            summary = await self.execution_summarizer.create_narrative_summary(
                tool_name=tool_name,
                tool_args=tool_args,
                result_data=result_data,
                initial_narrative="",
                gemini_client=getattr(self, 'gemini_client', None),
                gemini_types=getattr(self, 'gemini_types', None),
                claude_client=getattr(self, 'client', None) if self.summarizer_model.startswith('claude') else None
            )
            return summary
        except Exception as e:
            logger.warning(f"Tool summarization failed: {e}")
            # Fallback to simple summary
            return f"⚡️Used *{tool_name}* - completed successfully"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_path = "agents/system prompts/client_agent_sys_prompt.yaml"
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found. Using default values.")
            return {
                'system_prompt': 'You are a helpful AI assistant.',
                'model_config': {
                    'max_tokens': 4096,
                    'temperature': 0.7,
                    'max_iterations': 50
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}. Using default values.")
            return {
                'system_prompt': 'You are a helpful AI assistant.',
                'model_config': {
                    'max_tokens': 4096,
                    'temperature': 0.7,
                    'max_iterations': 50
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}. Using default values.")
            return {
                'system_prompt': 'You are a helpful AI assistant.',
                'model_config': {
                    'max_tokens': 4096,
                    'temperature': 0.7,
                    'max_iterations': 50
                }
            }
    
    async def process_request(self, message: str, context: Dict[str, Any] = None, streaming_callback=None) -> Dict[str, Any]:
        """
        Process user request with intelligent model routing
        
        Args:
            message: User's message/request
            context: Additional context (platform, timestamp, etc.)
            streaming_callback: Optional callback for streaming thinking process
            
        Returns:
            Dict containing response data
        """
        try:
            t_overall = time.time()
            # Extract user_id from context and add basic context information
            user_id = context.get('user_id') if context else None
            
            # DEBUG: Log user_id flow in ClientAgent
            logger.info(f"CLIENT-AGENT-USER-ID: Received user_id='{user_id}' from context")
            logger.info(f"CLIENT-AGENT-CONTEXT: Full context={context}")
            
            # Build context information
            full_message = message
            context_info = ""
            current_time = ""
            
            if context:
                platform = context.get('platform', 'unknown')
                timestamp = context.get('timestamp', 'unknown')
                user_timezone = context.get('user_timezone', '')
                user_name = context.get('user_name', '')
                user_title = context.get('user_title', '')
                
                # Build context string with available user information
                context_parts = [f"Platform={platform}"]
                
                # Add user information if available
                if user_name:
                    context_parts.append(f"User={user_name}")
                if user_title:
                    context_parts.append(f"Role={user_title}")
                
                # Add timestamp with timezone
                if user_timezone and user_timezone != 'UTC':
                    context_parts.append(f"Current time for user: {timestamp} ({user_timezone})")
                    current_time = f"{timestamp} ({user_timezone})"
                else:
                    context_parts.append(f"Current time: {timestamp}")
                    current_time = timestamp
                
                full_message += f"\n\n[Context: {', '.join(context_parts)}]"
                context_info = "\n".join(context_parts)
            
            # =============================================================================
            # MODEL ROUTING DECISION - Route to appropriate agent
            # =============================================================================
            
            if streaming_callback:
                await streaming_callback("🤖 Determining optimal model for your request...", "operation")
            
            # Get routing decision
            t_routing = time.time()
            route_decision = await self.routing_agent.route_query(message)
            routing_time_ms = int((time.time() - t_routing) * 1000)
            
            logger.info(f"MODEL-ROUTING: Routed to {route_decision.value} in {routing_time_ms}ms")
            
            if route_decision in [RouteDecision.SIMPLE, RouteDecision.TIER_1]:
                # Route to Gemini Flash for simple queries
                if streaming_callback:
                    await streaming_callback("⚡ Processing with fast conversational agent...", "operation")
                
                # Get conversation history for context
                conversation_history = await self._get_formatted_conversation_history(user_id, streaming_callback)
                
                # Process with Gemini Simple Agent
                t_gemini = time.time()
                result = await self.gemini_simple_agent.process_simple_request(
                    message, conversation_history, context_info, current_time
                )
                gemini_time_ms = int((time.time() - t_gemini) * 1000)
                
                # Add timing information
                result["routing_time_ms"] = routing_time_ms
                result["processing_time_ms"] = gemini_time_ms
                result["total_time_ms"] = int((time.time() - t_overall) * 1000)
                
                # Store in memory if successful
                if result.get("success") and user_id:
                    await self._store_conversation_in_memory(message, result["response"], user_id)
                
                return result
                
            else:
                # Route to Claude for complex queries
                if streaming_callback:
                    await streaming_callback("🧠 Processing with advanced reasoning agent...", "operation")
                
                # Run the Claude agent loop with iterative tool calling
                logger.info(f"CLAUDE-ROUTING: Calling run_agent_loop with user_id='{user_id}'")
                t_loop = time.time()
                response = await self.run_agent_loop(full_message, streaming_callback, user_id)
                claude_time_ms = int((time.time() - t_loop) * 1000)
                
                return {
                    "success": True,
                    "response": response,
                    "message": "Request processed successfully",
                    "agent": "claude_complex",
                    "model": "claude-3-5-sonnet",
                    "routing_time_ms": routing_time_ms,
                    "processing_time_ms": claude_time_ms,
                    "total_time_ms": int((time.time() - t_overall) * 1000)
                }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "response": f"I encountered an error while processing your request: {str(e)}",
                "message": "Processing failed",
                "agent": "client_agent",
                "error": str(e)
            }
    

    def _needs_reasoning(self, message: str) -> bool:
        """
        Determine if a message needs explicit reasoning (thinking tags) or can be answered directly.
        
        Messages that DON'T need reasoning:
        - Greetings and pleasantries
        - Basic chitchat
        - Thank you messages
        - Simple acknowledgments
        - Basic questions about the assistant itself
        - Simple factual responses
        
        Messages that DO need reasoning:
        - Multi-step problem solving
        - Questions requiring analysis or calculation
        - Complex information synthesis
        - Requests that need planning or strategy
        
        Note: This is NOT about tool usage - even simple messages should use memory tools!
        """
        normalized = message.lower().strip()
        
        # Simple patterns that don't need reasoning
        simple_patterns = [
            # Greetings
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', "what's up", "how are you", "howdy", "sup", "yo",
            'hiya', 'heya', 'hola', 'bonjour', 'salut', 'ciao', 'aloha',
            "how's it going", "how do you do", 'welcome', 'g\'day',
            
            # Pleasantries and acknowledgments
            'thanks', 'thank you', 'thx', 'ty', 'cheers', 'appreciated',
            'ok', 'okay', 'alright', 'sure', 'got it', 'understood',
            'goodbye', 'bye', 'see you', 'later', 'farewell', 'take care',
            'please', 'sorry', 'excuse me', 'pardon',
            
            # Basic questions about the assistant
            'who are you', 'what are you', 'your name', 'what can you do',
            'are you there', 'can you help', 'are you ai', 'are you a bot',
            
            # Simple affirmations/negations
            'yes', 'no', 'yeah', 'nope', 'yep', 'nah', 'maybe', 'perhaps'
        ]
        
        # Check if it's a simple pattern - these DON'T need reasoning
        if normalized in simple_patterns:
            return False  # False = doesn't need reasoning
        
        # Check if it starts with a simple pattern and is short
        if len(normalized.split()) <= 5:
            for pattern in simple_patterns:
                if normalized.startswith(pattern) or normalized.endswith(pattern):
                    return False  # False = doesn't need reasoning
        
        # Indicators that definitely NEED reasoning
        reasoning_indicators = [
            'how to', 'why', 'explain', 'analyze', 'compare', 'calculate',
            'plan', 'strategy', 'solve', 'debug', 'fix', 'troubleshoot',
            'optimize', 'improve', 'design', 'create', 'build', 'develop',
            'what should', 'how should', 'what would', 'how would',
            'pros and cons', 'advantages', 'disadvantages', 'best way'
        ]
        
        # If it contains reasoning indicators, it needs reasoning
        for indicator in reasoning_indicators:
            if indicator in normalized:
                return True  # True = needs reasoning
        
        # Very short messages (1-3 words) usually don't need reasoning
        word_count = len(normalized.split())
        if word_count <= 3:
            return False  # False = doesn't need reasoning
        
        # Complex questions (long with question mark) probably need reasoning
        if '?' in normalized and word_count > 15:
            return True  # True = needs reasoning
        
        # Simple factual questions don't need reasoning
        if normalized.startswith(('what is', 'who is', 'where is', 'when is')):
            return False  # False = doesn't need reasoning
        
        # Default to no reasoning for simple responses
        return False  # False = doesn't need reasoning
    
    async def run_agent_loop(self, user_message: str, streaming_callback=None, user_id: Optional[str] = None) -> str:
        """Run the agent loop with improved prompt assembly structure"""
        if not user_id:
            raise ValueError("user_id is required for agent processing - cannot proceed without user identification")
        
        # =============================================================================
        # NEW PROMPT ASSEMBLY STRUCTURE - Following improved schema
        # =============================================================================
        
        # 1. COMPLEXITY ROUTING - Determine reasoning strategy upfront
        needs_reasoning = self._needs_reasoning(user_message)
        
        if not needs_reasoning:
            logger.info(f"COMPLEXITY-ROUTING: Simple message - no thinking tags: '{user_message[:50]}...'")
        else:
            logger.info(f"COMPLEXITY-ROUTING: Complex message - full reasoning: '{user_message[:50]}...'")
        
        # 2. RETRIEVE CONVERSATION HISTORY
        conversation_history = await self._get_formatted_conversation_history(user_id, streaming_callback)
        
        # 3. ASSEMBLE STRUCTURED PROMPT
        enriched_message = await self._assemble_structured_prompt(
            user_message, user_id, conversation_history, streaming_callback
        )
        
        # DEBUG: Log enriched message structure
        logger.info(f"STRUCTURED-PROMPT: Assembled prompt for user {user_id}")
        logger.info(f"PROMPT-LENGTH: {len(enriched_message)} characters")
        
        # Mark the message as cacheable for agent loop iterations
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": enriched_message, "cache_control": {"type": "ephemeral"}}
            ]
        }]
        max_iterations = self.max_iterations  # Load from config
        
        # Capture thinking content and tool usage for run summary
        all_thinking_content = []
        tool_usage_log = []
        final_response = None
        final_iteration = 0
        # Capture text responses from any iteration (in case Claude provides answer + tool calls together)
        text_responses = []
        
        for iteration in range(max_iterations):
            t_iter = time.time()
            logger.info(f"Agent iteration {iteration + 1}")
            
            if streaming_callback:
                await streaming_callback(f"Agent iteration {iteration + 1}", "status")
            
            # Generate response using Claude with rate limiting
            loop = asyncio.get_event_loop()
            t_llm = time.time()
            
            # Estimate tokens for rate limiting (rough estimate)
            message_text = " ".join([
                msg.get("content", "") if isinstance(msg.get("content"), str) 
                else " ".join([c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text"])
                for msg in messages
            ])
            token_estimate = self.rate_limit_handler.estimate_tokens(
                self.system_prompt + message_text
            )
            
            # Adjust system prompt based on complexity routing decision
            if not needs_reasoning:
                # Add instruction to skip thinking tags for simple messages
                adjusted_prompt = self.system_prompt + "\n\nCOMPLEXITY ROUTING: This is a simple message. Respond directly WITHOUT using <thinking> tags. Be conversational and natural."
            else:
                adjusted_prompt = self.system_prompt + "\n\nCOMPLEXITY ROUTING: This is a complex message requiring full reasoning. Use <thinking> tags to show your reasoning process."
            
            # Make API call with rate limit handling
            async def make_api_call():
                return await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        # Mark system prompt as cacheable
                        system=[{"type": "text", "text": adjusted_prompt, "cache_control": {"type": "ephemeral"}}],
                        messages=messages,
                        tools=self.tools,
                        timeout=60.0  # Add timeout to prevent streaming errors
                    )
                )
            
            response = await self.rate_limit_handler.execute_with_retry(
                make_api_call,
                estimate_tokens=token_estimate
            )
            
            logger.info(f"AGENT-TIMING: LLM call took {int((time.time()-t_llm)*1000)} ms user={user_id} iter={iteration+1}")
            
            logger.info(f"Claude response iteration {iteration + 1}: {response}")
            
            # Process response content
            assistant_content = []
            tool_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    text = content_block.text
                    assistant_content.append({"type": "text", "text": text})
                    
                    # Capture text responses from this iteration
                    clean_text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
                    if clean_text:
                        text_responses.append(clean_text)
                        logger.info(f"RESPONSE-DEBUG: Captured text response: {clean_text[:50]}...")
                    
                    # Capture thinking content for run summary
                    thinking_matches = re.findall(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
                    for thinking in thinking_matches:
                        all_thinking_content.append(thinking.strip())
                    if thinking_matches:
                        logger.info(f"DEBUG-THINKING: Found {len(thinking_matches)} thinking blocks in text response")
                    
                    # Extract and stream thinking content
                    if streaming_callback:
                        await self._extract_and_stream_thinking(text, streaming_callback)
                
                elif content_block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
                    tool_calls.append(content_block)
            
            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": assistant_content})
            
            # If no tool calls, we're done - extract response and break
            if not tool_calls:
                # Extract final text response
                final_text = ""
                for content in assistant_content:
                    if content["type"] == "text":
                        final_text += content["text"]
                
                # Remove thinking tags from final response
                final_response = re.sub(r'<thinking>.*?</thinking>', '', final_text, flags=re.DOTALL).strip()
                final_iteration = iteration + 1
                logger.info(f"RESPONSE-DEBUG: No tool calls, extracted final_response: {final_response[:50]}...")
                
                # If the extracted response is empty but we have captured text responses, use those
                if not final_response and text_responses:
                    final_response = max(text_responses, key=len)
                    logger.info(f"RESPONSE-DEBUG: Using captured text response as fallback: {final_response[:50]}...")
                
                break
            
            # Execute tool calls and add results
            tool_results = []
            
            for tool_call in tool_calls:
                if streaming_callback:
                    # For execute_tool, show the actual tool name being executed, not "execute_tool"
                    if tool_call.name == "execute_tool" and "tool_name" in tool_call.input:
                        actual_tool = tool_call.input.get("tool_name", "unknown")
                        tool_args = tool_call.input.get("tool_args", {})
                        # Pass both name and args for execute_tool
                        await streaming_callback({"name": f"⚡️{actual_tool}", "args": tool_args}, "tool_start")
                    else:
                        await streaming_callback(tool_call.name, "tool_start")
                
                t_tool = time.time()
                result = await self._execute_claude_tool(tool_call, streaming_callback, user_id)
                logger.info(f"AGENT-TIMING: tool {tool_call.name} took {int((time.time()-t_tool)*1000)} ms user={user_id}")
                
                # Log tool usage for run summary
                tool_usage_log.append({
                    "tool": tool_call.name,
                    "args": tool_call.input,
                    "success": "Error:" not in str(result),
                    "result_preview": str(result)[:200]
                })
                
                # Enhanced tool result logging for memory debugging
                if tool_call.name in ["memory_retrieve", "memory_add"]:
                    logger.info(f"MEMORY-DEBUG tool={tool_call.name} result_type={type(result)} result_preview={str(result)[:500]}")
                
                # For any dict result, format as JSON so the model can parse it reliably
                if isinstance(result, dict):
                    import json
                    content = json.dumps(result, indent=2)
                else:
                    content = str(result)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": content
                })
                
                # Note: Tools are now discovered purely through registry search
                # All execution is handled dynamically by tool_executor
                
                # REMOVED tool_result streaming for cleaner output
                # The ⚡ summary is already shown at tool execution time
                # if streaming_callback:
                #     # For execute_tool, pass the actual tool name with consistent formatting
                #     if tool_call.name == "execute_tool" and "tool_name" in tool_call.input:
                #         actual_tool = tool_call.input.get("tool_name", "unknown")
                #         # Parse tool name into service and action for consistent formatting
                #         if "." in actual_tool:
                #             service, action = actual_tool.split(".", 1)
                #             formatted_tool = f"*{service}* *{action}*"
                #         else:
                #             formatted_tool = f"*{actual_tool}*"
                #         await streaming_callback(f"⚡️{formatted_tool}: {str(result)[:100]}...", "tool_result")
                #     else:
                #         await streaming_callback(f"{tool_call.name}: {str(result)[:100]}...", "tool_result")
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
        
        # If we reach here, max iterations was hit - use captured text responses if available
        if final_response is None:
            if text_responses:
                # Use the most substantial text response (longest one, as it likely contains the answer)
                final_response = max(text_responses, key=len)
                final_iteration = max_iterations
                logger.info(f"RESPONSE-FIX: Using captured text response: {final_response[:100]}")
            else:
                final_response = "Maximum iterations reached. Unable to complete the request."
                final_iteration = max_iterations
                logger.info(f"RESPONSE-FIX: No text responses captured, total captured: {len(text_responses)}")
        
        # Update local buffer for context management
        logger.info(f"CONTROL-FLOW: Updating local buffer - Response: {final_response[:100]}...")
        
        if streaming_callback:
            await streaming_callback("Updating local conversation buffer...", "operation")
        
        # Update local buffer (immediate)
        self._update_buffer(user_message, final_response, tool_usage_log, all_thinking_content, user_id)
        
        # =============================================================================
        # AUTOMATIC MEMORY ADDITION - Store conversation in memory at the end
        # =============================================================================
        try:
            logger.info(f"AUTO-MEMORY: Adding conversation to memory for user {user_id}")
            
            # Automatically add the conversation to memory
            memory_add_args = {
                "user_input": user_message,
                "agent_response": final_response,
                "user_id": user_id
            }
            
            # Run memory addition in background to not block response
            async def add_memory_background():
                try:
                    result = await self.tool_executor.execute_command("memory.add", memory_add_args, user_id=user_id)
                    if isinstance(result, dict) and result.get("status") != "error":
                        logger.info(f"AUTO-MEMORY: Successfully added conversation to memory for user {user_id}")
                    else:
                        logger.warning(f"AUTO-MEMORY: Failed to add memory - result: {result}")
                except Exception as e:
                    logger.warning(f"AUTO-MEMORY: Failed to add conversation to memory: {e}")
            
            # Create background task for memory addition
            memory_task = asyncio.create_task(add_memory_background())
            self._track_background_task(memory_task)
            
        except Exception as e:
            logger.warning(f"AUTO-MEMORY: Failed to initiate memory addition: {e}")
            # Continue without storing memory if addition fails
        
        # COMMENTED OUT: Update insights asynchronously using Conversation Insights Agent (non-blocking)
        # insights_task = asyncio.create_task(self.insights_agent.analyze_interaction(
        #     user_message, final_response, tool_usage_log, all_thinking_content, user_id, self.user_buffers
        # ))
        # self._track_background_task(insights_task)
        
        return final_response
    

    

    
    # Removed dynamic tool loading methods - agent now relies purely on registry discovery
    # All tool execution is handled by tool_executor with dynamic loading
    
    async def _extract_and_stream_thinking(self, text: str, streaming_callback):
        """Extract thinking content and stream it efficiently"""
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
        
        if thinking_matches:
            logger.info(f"DEBUG-THINKING: Streaming {len(thinking_matches)} thinking blocks")
        
        for thinking_content in thinking_matches:
            # Stream the entire thinking block at once (more efficient)
            clean_thinking = thinking_content.strip()
            if clean_thinking:
                await streaming_callback(clean_thinking, "thinking")
                # Small delay between thinking blocks (not lines) to avoid rate limits
                if len(thinking_matches) > 1:
                    await asyncio.sleep(0.1)
    
    async def _execute_claude_tool(self, tool_use_block, streaming_callback=None, user_id: str = None) -> str:
        """Execute a Claude tool call and return the result"""
        try:
            function_name = tool_use_block.name
            args = tool_use_block.input
            
            logger.info(f"Executing Claude tool: {function_name}, Args: {args}")
            
            # Stream tool details if callback provided
            if streaming_callback and self.stream_tool_details:
                # Don't show tool details for registry tools or execute_tool
                if function_name not in ["reg_search", "reg_describe", "reg_list", "reg_categories", "execute_tool", "memory_add", "memory_retrieve"]:
                    # Build a preview of arguments
                    args_preview = []
                    for key, value in args.items():
                        if isinstance(value, str) and len(value) > 50:
                            args_preview.append(f"{key}: {value[:50]}...")
                        elif isinstance(value, (list, dict)):
                            args_preview.append(f"{key}: {type(value).__name__}[{len(value)}]")
                        else:
                            args_preview.append(f"{key}: {value}")
                    
                    if args_preview:
                        formatted_args = ", ".join(args_preview)
                        # REMOVED: Parameters output for cleaner display
                        # await streaming_callback(f"Parameters: {formatted_args}", "tool_details")
                    else:
                        # REMOVED: Running message for cleaner display
                        # await streaming_callback(f"Running {function_name}", "tool_details")
                        pass
            
            # Handle registry tools directly, then try dynamic execution for others
            if function_name == "reg_search":
                # Phase 1: Send working message immediately
                if streaming_callback:
                    start_message = self._generate_tool_start_message("reg_search", args)
                    await streaming_callback(start_message, "operation")
                
                # Execute tool
                args.setdefault("search_type", "description")
                args.setdefault("limit", 10)
                result = await self.tool_executor.execute_command("reg.search", args, user_id=user_id)
                
                # Phase 2: Generate and stream comprehensive Gemini summary
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("reg.search", args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for reg.search: {e}")
                        # Fallback summary
                        if isinstance(result, dict) and result.get("status") == "success":
                            tools = result.get("data", {}).get("tools", [])
                            count = len(tools) if tools else 0
                            query = args.get('query', 'tools')
                            await streaming_callback(f"⚡️Used *registry* *search* for '{query}' and found {count} results", "tool_result")
                        else:
                            await streaming_callback(f"⚡️Used *registry* *search* - completed", "tool_result")
            elif function_name == "reg_describe":
                # Phase 1: Send working message immediately
                if streaming_callback:
                    start_message = self._generate_tool_start_message("reg_describe", args)
                    await streaming_callback(start_message, "operation")
                
                # Execute tool
                args.setdefault("include_schema", True)
                result = await self.tool_executor.execute_command("reg.describe", args, user_id=user_id)
                
                # Phase 2: Generate and stream comprehensive Gemini summary
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("reg.describe", args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for reg.describe: {e}")
                        # Fallback summary
                        tool_name = args.get('tool_name', 'tool')
                        await streaming_callback(f"⚡️Used *registry* *describe* to get details for {tool_name}", "tool_result")
            elif function_name == "reg_list":
                args.setdefault("limit", 50)
                result = await self.tool_executor.execute_command("reg.list", args, user_id=user_id)
                
                # Generate and stream Gemini summary for registry list
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("reg.list", args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for reg.list: {e}")
                        # Fallback with basic info
                        explanation = args.get('explanation', 'list all available tools')
                        await streaming_callback(f"⚡️Used *registry* *list* to {explanation.lower()}", "tool_result")
            elif function_name == "reg_categories":
                result = await self.tool_executor.execute_command("reg.categories", args, user_id=user_id)
                
                # Generate and stream Gemini summary for registry categories
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("reg.categories", args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for reg.categories: {e}")
                        # Fallback with basic info
                        explanation = args.get('explanation', 'get tool categories')
                        await streaming_callback(f"⚡️Used *registry* *categories* to {explanation.lower()}", "tool_result")


            elif function_name == "execute_tool":
                # Execute any discovered tool dynamically - completely generic
                tool_name = args["tool_name"]
                tool_args = args.get("tool_args", {})
                
                # Phase 1: Send working message immediately
                if streaming_callback:
                    start_message = self._generate_tool_start_message(tool_name, tool_args)
                    await streaming_callback(start_message, "operation")
                
                # Execute tool
                result = await self.tool_executor.execute_command(tool_name, tool_args, user_id=user_id)
                
                # Phase 2: Generate and stream tool summary with real-time chunks
                if streaming_callback and self.summarization_enabled:
                    try:
                        # Use streaming version that sends chunks as they arrive
                        await self._generate_tool_summary_streaming(tool_name, tool_args, result, streaming_callback)
                    except Exception as e:
                        logger.warning(f"Failed to generate streaming tool summary for {tool_name}: {e}")
                        # Fall back to simple completion message
                        await streaming_callback(f"⚡️Used *{tool_name}* - completed", "tool_result")
                
            elif function_name == "memory_add":
                # Add user_id to args for memory operations
                memory_args = dict(args)
                memory_args["user_id"] = user_id
                
                result = await self.tool_executor.execute_command("memory.add", memory_args, user_id=user_id)
                
                # Generate and stream Gemini summary for memory add
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("memory.add", memory_args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for memory.add: {e}")
                        # Fallback with basic info
                        await streaming_callback("⚡️Used *memory* *add* to store conversation in memory", "tool_result")
            elif function_name == "memory_retrieve":
                # Add user_id to args for memory operations
                memory_args = dict(args)
                memory_args["user_id"] = user_id
                
                result = await self.tool_executor.execute_command("memory.retrieve", memory_args, user_id=user_id)
                
                # Generate and stream Gemini summary for memory retrieve
                if streaming_callback and self.summarization_enabled:
                    try:
                        summary = await self._generate_tool_summary("memory.retrieve", memory_args, result)
                        await streaming_callback(summary, "tool_result")
                    except Exception as e:
                        logger.warning(f"Failed to generate tool summary for memory.retrieve: {e}")
                        # Fallback with basic info
                        query = args.get('query', 'recent context')
                        await streaming_callback(f"⚡️Used *memory* *retrieve* to search for \"{query}\"", "tool_result")
            else:
                # SECURITY CHECK: Prevent direct execution of discovered tools
                # Claude should only use execute_tool for weather, perplexity, etc.
                if "." in function_name and function_name not in ["memory_add", "memory_retrieve"]:
                    error_msg = f"Direct execution of '{function_name}' is not allowed. Use execute_tool with tool_name='{function_name}' instead."
                    logger.warning(f"SECURITY: Blocked direct execution of discovered tool: {function_name}")
                    if streaming_callback:
                        await streaming_callback(error_msg, "error")
                    return error_msg
                
                error_msg = f"Unknown function: {function_name}. Available tools: reg_search, reg_describe, reg_list, reg_categories, execute_tool, memory_add, memory_retrieve"
                if streaming_callback:
                    await streaming_callback(error_msg, "error")
                return error_msg
            
            logger.info(f"Tool executor result: {result}")
            
            # Stream the result details - REMOVED for cleaner output
            # Only keep the ⚡ summary messages above
            # if streaming_callback:
            #     if isinstance(result, dict):
            #         # Check both "success" (boolean) and "status" (string) fields for compatibility
            #         success = (result.get("success") == True) or (result.get("status") == "success")
            #         # Provide a more helpful default message for tools that do not include 'message'
            #         default_message = "Operation completed successfully"
            #         if function_name in ("memory_retrieve", "memory_add"):
            #             # memory tools often return structured data without a 'message'
            #             # so avoid confusing 'No message' in the stream
            #             default_message = "OK"
            #         message = result.get("message", default_message)
            #         if success:
            #             await streaming_callback(f"{function_name} succeeded: {message}", "result")
            #             # Also stream key data if available
            #             data = result.get("data", {})
            #             if isinstance(data, dict):
            #                 if "total_results" in data:
            #                     await streaming_callback(f"Found {data['total_results']} results", "result_detail")
            #                 elif "answer" in data:
            #                     # Clean up answer display - remove think tags and show clean preview
            #                     answer_text = str(data['answer'])
            #                     # Remove <think> tags and their content
            #                     import re
            #                     clean_answer = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL)
            #                     clean_answer = clean_answer.strip()
            #                     preview = clean_answer[:150] + "..." if len(clean_answer) > 150 else clean_answer
            #                     await streaming_callback(f"Retrieved analysis: {preview}", "result_detail")
            #         else:
            #             await streaming_callback(f"{function_name} failed: {message}", "error")
            #     else:
            #         await streaming_callback(f"Got result: {str(result)[:150]}...", "result")
            
            # Return the FULL result so the model can actually use it
            # Do not collapse dict results to a short message; Anthropic expects the tool_result
            # content to include the structured data for the assistant to reference.
            if isinstance(result, dict):
                return result
            else:
                # For non-dict results, return as string
                return str(result) if result is not None else "No result returned"
                
        except Exception as e:
            logger.error(f"Claude tool execution error: {e}")
            return f"Error executing {function_name}: {str(e)}"
    

    # =============================================================================
    # STRUCTURED PROMPT ASSEMBLY - Following improved schema
    # =============================================================================
    
    async def _get_formatted_conversation_history(self, user_id: str, streaming_callback=None) -> str:
        """
        Retrieve and format conversation history in proper chronological order
        Following schema: newest conversations first, clear structure
        """
        try:
            if streaming_callback:
                await streaming_callback("Retrieving conversation history...", "operation")
            
            # Automatically retrieve recent conversation history
            memory_args = {
                "query": "recent history",
                "user_id": user_id,
                "max_results": 10
            }
            
            logger.info(f"CONVERSATION-HISTORY: Retrieving for user {user_id}")
            memory_result = await self.tool_executor.execute_command("memory.retrieve", memory_args, user_id=user_id)
            
            if not isinstance(memory_result, dict) or memory_result.get("status") == "error":
                logger.info(f"CONVERSATION-HISTORY: No memory retrieved for user {user_id}")
                return ""
            
            short_term = memory_result.get("short_term_memory", [])
            retrieved_pages = memory_result.get("retrieved_pages", [])
            
            if not short_term and not retrieved_pages:
                return ""
            
            history_parts = ["CONVERSATION HISTORY"]
            
            # Format short-term memory (reverse order to show most recent first)
            if short_term:
                # Reverse the FIFO order to show newest first
                for i, entry in enumerate(reversed(short_term), 1):
                    if isinstance(entry, dict):
                        user_msg = entry.get("user_input", "")
                        agent_msg = entry.get("agent_response", "")
                        timestamp = entry.get("timestamp", "")
                        
                        if user_msg or agent_msg:
                            # Clear labels: most recent exchange first
                            if i == 1:
                                history_parts.append(f"\nPREVIOUS MESSAGE ({i}) - Most Recent:")
                            else:
                                history_parts.append(f"\nMESSAGE {i}:")
                            
                            if user_msg:
                                history_parts.append(f"{user_msg}")
                            
                            if agent_msg:
                                history_parts.append(f"\nRESPONSE {i}:")
                                history_parts.append(f"{agent_msg}")
            
            # Add historical context if available
            if retrieved_pages:
                history_parts.append(f"\nHIGHLIGHTS (Historical Context):")
                for page in retrieved_pages[:3]:  # Limit to 3 most relevant
                    if isinstance(page, dict):
                        content = page.get("content", "")
                        if content:
                            # Truncate long historical entries
                            truncated_content = content[:200] + "..." if len(content) > 200 else content
                            history_parts.append(f"• {truncated_content}")
            
            conversation_history = "\n".join(history_parts)
            
            logger.info(f"CONVERSATION-HISTORY: Formatted {len(short_term)} recent + {len(retrieved_pages)} historical entries")
            return conversation_history
            
        except Exception as e:
            logger.warning(f"CONVERSATION-HISTORY: Failed to retrieve: {e}")
            return ""
    
    async def _assemble_structured_prompt(self, user_message: str, user_id: str, 
                                        conversation_history: str, streaming_callback=None) -> str:
        """
        Assemble the complete prompt following the improved schema structure:
        1. Conversation History (if any)
        2. General User Context (platform, user, role)  
        3. Current Time
        4. Current User Message (prominently highlighted)
        """
        if streaming_callback:
            await streaming_callback("Assembling structured prompt...", "operation")
        
        prompt_parts = []
        
        # 1. CONVERSATION HISTORY (if available)
        if conversation_history:
            prompt_parts.append(conversation_history)
            prompt_parts.append("")  # Add spacing
        
        # 2. Get current context from process_request (we'll need to pass this)
        # For now, we'll extract from user_message if it has context
        context_info = self._extract_context_from_message(user_message)
        clean_user_message = self._clean_user_message(user_message)
        
        # 3. GENERAL USER CONTEXT (if available)
        if context_info:
            prompt_parts.append("GENERAL USER CONTEXT")
            prompt_parts.append(context_info)
            prompt_parts.append("")
        
        # 4. HIGHLIGHTS (pins/recommendations - currently disabled)
        highlights = await self._get_highlights(user_id)
        if highlights:
            prompt_parts.append("HIGHLIGHTS")
            prompt_parts.append(highlights)
            prompt_parts.append("")
        
        # 5. CURRENT TIME
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt_parts.append("CURRENT TIME")
        prompt_parts.append(current_time)
        prompt_parts.append("")
        
        # 6. CURRENT USER MESSAGE (prominently highlighted at bottom)
        prompt_parts.append("=" * 50)
        prompt_parts.append("🎯 CURRENT USER REQUEST")
        prompt_parts.append("Based on the above context, respond to the following prompt:")
        prompt_parts.append("")
        prompt_parts.append(clean_user_message)
        prompt_parts.append("=" * 50)
        
        structured_prompt = "\n".join(prompt_parts)
        
        logger.info(f"STRUCTURED-PROMPT: Assembled {len(prompt_parts)} sections for user {user_id}")
        return structured_prompt
    
    def _extract_context_from_message(self, user_message: str) -> str:
        """Extract context information appended to user message"""
        if "[Context:" in user_message:
            context_start = user_message.find("[Context:")
            context_end = user_message.find("]", context_start)
            if context_end != -1:
                context_part = user_message[context_start+9:context_end]  # Skip "[Context:"
                # Format context nicely
                context_items = [item.strip() for item in context_part.split(",")]
                return "\n".join(context_items)
        return ""
    
    def _clean_user_message(self, user_message: str) -> str:
        """Remove context information from user message to get clean query"""
        if "[Context:" in user_message:
            context_start = user_message.find("\n\n[Context:")
            if context_start != -1:
                return user_message[:context_start].strip()
        return user_message.strip()
    
    async def _store_conversation_in_memory(self, user_message: str, agent_response: str, user_id: str):
        """
        Store conversation in memory for both Gemini and Claude responses
        """
        try:
            logger.info(f"MEMORY-STORAGE: Adding conversation to memory for user {user_id}")
            
            memory_add_args = {
                "user_input": user_message,
                "agent_response": agent_response,
                "user_id": user_id
            }
            
            # Run memory addition in background to not block response
            async def add_memory_background():
                try:
                    result = await self.tool_executor.execute_command("memory.add", memory_add_args, user_id=user_id)
                    if isinstance(result, dict) and result.get("status") != "error":
                        logger.info(f"MEMORY-STORAGE: Successfully added conversation to memory for user {user_id}")
                    else:
                        logger.warning(f"MEMORY-STORAGE: Failed to add memory - result: {result}")
                except Exception as e:
                    logger.warning(f"MEMORY-STORAGE: Failed to add conversation to memory: {e}")
            
            # Create background task for memory addition
            memory_task = asyncio.create_task(add_memory_background())
            self._track_background_task(memory_task)
            
        except Exception as e:
            logger.warning(f"MEMORY-STORAGE: Failed to initiate memory addition: {e}")


    # =============================================================================
    # LEGACY BUFFER SYSTEM - REMOVED (pins/recommendations taken out)
    # =============================================================================
    
    # This method is no longer used since we removed the pins/recommendations system
    async def assemble_from_local_buffer(self, user_message: str, user_id: str) -> str:
        """
        ENHANCED: Assemble enriched prompt with two-section insights from Conversation Insights Agent
              
        This method handles insights from the stateful Conversation Insights Agent in two sections:
        - PINS: User preferences, needs, constraints, conversation context (max 2 items)
        - RECOMMENDATIONS: Tool usage guidance, error patterns, workflow optimization (max 1 item)
        
        Args:
            user_message: Current user message
            user_id: User identifier for buffer isolation
            
        Returns:
            Enriched message with relevant pins and recommendations sections
        """
        logger.info(f"PROMPT-ASSEMBLY: Assembling pins + recommendations for user {user_id}")
        
        # Early exit if buffer expired
        if not self._is_buffer_fresh(user_id):
            logger.info(f"PROMPT-ASSEMBLY: Buffer expired for user {user_id}, using message as-is")
            return user_message
        
        buffer = self.user_buffers.get(user_id, {})
        if not buffer:
            return user_message
        
        # Only process important items
        important = buffer.get('important', {})
        if not important:
            return user_message
        
        current_time = time.time()
        
        # Pre-filter by recency (30 minutes) - avoid processing old items
        recency_threshold = current_time - 1800  # 30 minutes
        recent_items = {
            k: v for k, v in important.items() 
            if v.get('timestamp', 0) > recency_threshold
        }
        
        if not recent_items:
            return user_message
        
        # Separate pins and recommendations for different treatment
        pins = []
        recommendations = []
        
        # Sort by timestamp once (most recent first)
        sorted_items = sorted(
            recent_items.items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            )
        
        # Process items with relevance scoring, separated by type
        for note_id, note_data in sorted_items:
            notes = note_data.get('notes', '')
            related_question = note_data.get('related_question', '')
            insight_type = note_data.get('insight_type', 'pin')  # Default to pin
            
            # Calculate relevance based on keyword overlap
            user_words = set(user_message.lower().split())
            note_words = set((notes + ' ' + related_question).lower().split())
            
            overlap = (len(user_words.intersection(note_words)) / len(user_words) 
                     if user_words else 0)
            
            # Boost if related to current context
            context_boost = 0.2 if any(word in notes.lower() 
                                     for word in user_message.lower().split()[:3]) else 0
            
            relevance = overlap + context_boost
            
            # Different relevance thresholds for different types
            relevance_threshold = 0.05 if insight_type == 'recommendation' else 0.1  # Lower bar for recommendations
            
            if relevance > relevance_threshold:
                # Clean up the notes - remove markdown formatting that might confuse Claude
                clean_notes = notes.replace("**", "").replace("*", "")
                content = f"- {clean_notes[:150]}{'...' if len(clean_notes) > 150 else ''}"
                
                candidate = {
                    'content': content,
                    'relevance_score': relevance,
                    'timestamp': note_data.get('timestamp', 0),
                    'insight_type': insight_type
                }
                
                if insight_type == 'recommendation':
                    recommendations.append(candidate)
                else:
                    pins.append(candidate)
        
        # Sort each type by relevance
        pins.sort(key=lambda x: x['relevance_score'], reverse=True)
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Build context with length limit, prioritizing pins for conversation context
        context_parts = [user_message]
        current_length = len(user_message)
        max_context_length = 800
        
        # Determine what we'll include
        relevant_pins = pins[:2]  # Max 2 most relevant pins
        relevant_recommendations = recommendations[:1]  # Max 1 most relevant recommendation
        
        # Add pins section if we have relevant pins
        if relevant_pins:
            pins_header = "\n\nRelevant context from our previous conversations:"
            if current_length + len(pins_header) < max_context_length:
                context_parts.append(pins_header)
                current_length += len(pins_header)
                
                for pin in relevant_pins:
                    if current_length + len(pin['content']) > max_context_length:
                        break
                    context_parts.append(pin['content'])
                    current_length += len(pin['content'])
        
        # Add recommendations section if we have relevant ones and space
        if relevant_recommendations and current_length < max_context_length - 100:  # Leave some space
            rec_header = "\n\nBased on previous interactions, please note:"
            if current_length + len(rec_header) < max_context_length:
                context_parts.append(rec_header)
                current_length += len(rec_header)
                
                for rec in relevant_recommendations:
                    if current_length + len(rec['content']) > max_context_length:
                        break
                    context_parts.append(rec['content'])
                    current_length += len(rec['content'])
        
        # Only add context if we found something relevant
        if len(context_parts) == 1:  # Only user_message, no context added
            return user_message
        
        enriched_message = "\n".join(context_parts)
        
        pins_count = len(relevant_pins) if relevant_pins else 0
        recs_count = len(relevant_recommendations) if relevant_recommendations else 0
        logger.info(f"PROMPT-ASSEMBLY: Added {pins_count} pins + {recs_count} recommendations, "
                   f"length: {current_length}")
        
        
        return enriched_message
    
    def _is_buffer_fresh(self, user_id: str) -> bool:
        """Check if user buffer is fresh (within 30-minute expiry)"""
        buffer = self.user_buffers.get(user_id)
        if not buffer:
            return False
        
        last_updated = buffer.get('last_updated', 0)
        expiry_threshold = time.time() - (self.buffer_expiry_minutes * 60)
        
        return last_updated > expiry_threshold
    

    
    def _update_buffer(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list, user_id: str):
        """Update local buffer for conversation context."""
        logger.info(f"BUFFER: Updating buffer for user {user_id}")
        
        # Initialize user buffer if not exists
        if user_id not in self.user_buffers:
            logger.info(f"BUFFER-NEW-USER: Creating new buffer for user_id='{user_id}'")
            self.user_buffers[user_id] = {
                'important': {},
                'last_updated': time.time()
            }
        else:
            logger.info(f"BUFFER-EXISTING-USER: Using existing buffer for user_id='{user_id}'")
        
        # Update timestamp
        self.user_buffers[user_id]['last_updated'] = time.time()
        logger.info(f"BUFFER: Buffer timestamp updated for user {user_id}")



    
    

    

    
    # =============================================================================
    # CONVERSATION INSIGHTS METHODS - MOVED TO ConvoInsightsAgent
    # =============================================================================
    # 
    # The following methods have been refactored into a separate ConvoInsightsAgent:
    # - _update_important() -> insights_agent.analyze_interaction()
    # - _get_current_insights() -> moved to ConvoInsightsAgent
    # - _replace_insights() -> moved to ConvoInsightsAgent  
    # - _generate_stateful_fallback_insights() -> moved to ConvoInsightsAgent
    #
    # These methods are now called via:
    #   self.insights_agent.analyze_interaction(user_message, agent_response, tool_usage_log, thinking_content, user_id, self.user_buffers)
    #
    # =============================================================================

    # DEPRECATED: This method has been replaced by ConvoInsightsAgent.analyze_interaction()
    async def _update_important(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list, user_id: str):
        """Stateful Conversation Insights Agent - builds evolving understanding of user"""
        try:
            logger.info(f"INSIGHTS-AGENT: Analyzing interaction for user {user_id}")
            
            # Only generate important notes if there's significant activity
            if not tool_usage_log and len(user_message.split()) < 10:
                logger.info(f"INSIGHTS-AGENT: Skipping trivial interaction for user {user_id}")
                return
            
            # Get existing insights for comparison and evolution
            existing_insights = self._get_current_insights(user_id)
            logger.info(f"INSIGHTS-AGENT: Found {len(existing_insights.split('•')) - 1 if existing_insights != 'No previous insights.' else 0} existing insights")
            
            # Prepare execution details
            execution_summary = ""
            if tool_usage_log:
                tools_used = [tool['tool'] for tool in tool_usage_log]
                tool_failures = [tool['tool'] for tool in tool_usage_log if not tool['success']]
                execution_summary = f"Tools used: {', '.join(tools_used)}. "
                if tool_failures:
                    execution_summary += f"Failed tools: {', '.join(tool_failures)}. "
            
            # Prepare thinking insights
            thinking_summary = ""
            if thinking_content:
                # Extract key insights from thinking (first 200 chars of each thinking block)
                key_thoughts = [thinking[:200] for thinking in thinking_content[:2]]
                thinking_summary = " ".join(key_thoughts)
            
            # Generate updated insights using stateful Conversation Insights Agent
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("INSIGHTS-AGENT: GEMINI_API_KEY not set, using fallback analysis")
                updated_insights = self._generate_stateful_fallback_insights(
                    user_message, agent_response, tool_usage_log, thinking_content, existing_insights, user_id
                )
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""You are the Conversation Insights Agent. Your job is to maintain an evolving understanding of this user across two key dimensions.

CURRENT USER INSIGHTS:
{existing_insights}

NEW INTERACTION TO ANALYZE:
User: "{user_message}"
Agent Response: "{agent_response}"
Tools Used: {execution_summary if execution_summary else "No tools used"}
Agent Reasoning: {thinking_summary if thinking_summary else "No detailed reasoning captured"}

ANALYSIS TASKS:
1. COMPARE: How does this new interaction relate to existing insights?
2. UPDATE: Do any existing insights need refinement/correction?
3. ADD: What new patterns or preferences emerge?
4. DELETE: Are any insights now outdated or contradicted?
5. SYNTHESIZE: Can you spot meta-patterns across interactions?

OUTPUT FORMAT:
Return insights organized into TWO distinct sections (maximum 3 items per section):

PINS:
• [User preference, need, constraint, or critical conversation context]
• [Personal information, domain expertise, communication style]
• [Recurring patterns, evolving context, important facts]

RECOMMENDATIONS:
• [Tool usage guidance: which tools work best for this user]
• [Error patterns: what approaches to avoid or retry]
• [Action space optimization: preferred workflows, successful strategies]

FOCUS AREAS:
PINS: User identity, preferences, needs, constraints, domain knowledge, communication patterns
RECOMMENDATIONS: Tool effectiveness, error handling, workflow optimization, technical approaches

If no meaningful insights exist in either section, return:
PINS:
• No significant conversation insights yet.
RECOMMENDATIONS:
• No tool usage patterns identified yet.

Generate updated insights:"""

                try:
                    response = model.generate_content(prompt)
                    updated_insights = response.text.strip()
                    logger.info(f"INSIGHTS-AGENT: Generated stateful insights for user {user_id}")
                except Exception as gemini_error:
                    logger.warning(f"INSIGHTS-AGENT: Gemini API failed ({gemini_error}), using fallback")
                    updated_insights = self._generate_stateful_fallback_insights(
                        user_message, agent_response, tool_usage_log, thinking_content, existing_insights, user_id
                    )
            
            # Replace entire insights set with evolved understanding
            self._replace_insights(user_id, updated_insights)
            
            logger.info(f"INSIGHTS-AGENT: Updated insights for user {user_id}")
            
        except Exception as e:
            logger.error(f"INSIGHTS-AGENT: Failed to update insights for user {user_id}: {e}")
            import traceback
            logger.error(f"INSIGHTS-AGENT: Traceback: {traceback.format_exc()}")
    
    def _get_current_insights(self, user_id: str) -> str:
        """Get formatted previous insights for the Conversation Insights Agent to review"""
        if user_id not in self.user_buffers:
            return "PINS:\n• No previous conversation insights.\n\nRECOMMENDATIONS:\n• No previous tool usage patterns."
        
        important_items = self.user_buffers[user_id].get('important', {})
        if not important_items:
            return "PINS:\n• No previous conversation insights.\n\nRECOMMENDATIONS:\n• No previous tool usage patterns."
        
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
    
    def _replace_insights(self, user_id: str, updated_insights: str):
        """Replace entire insights set with evolved understanding (two-section format)"""
        if user_id not in self.user_buffers:
            return
        
        # Clear existing insights
        self.user_buffers[user_id]['important'] = {}
        
        # Handle case where agent returns no insights
        if not updated_insights or "No significant insights yet" in updated_insights:
            logger.info(f"INSIGHTS-AGENT: No significant insights to store for user {user_id}")
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
        
        import uuid
        insights_count = 0
        
        # Process PINS section
        if pins_section:
            pin_lines = [line.strip() for line in pins_section.split('\n') if line.strip().startswith('•')]
            for i, pin_line in enumerate(pin_lines[:3]):  # Max 3 pins
                pin_text = pin_line[1:].strip()  # Remove '•' and whitespace
                
                if pin_text and "No significant conversation insights" not in pin_text:
                    note_id = f"pin_{uuid.uuid4().hex[:8]}"
                    self.user_buffers[user_id]['important'][note_id] = {
                        'notes': pin_text,
                        'timestamp': time.time(),
                        'insight_type': 'pin',
                        'related_question': f"Conversation insight #{i+1}"
                    }
                    insights_count += 1
        
        # Process RECOMMENDATIONS section
        if recommendations_section:
            rec_lines = [line.strip() for line in recommendations_section.split('\n') if line.strip().startswith('•')]
            for i, rec_line in enumerate(rec_lines[:3]):  # Max 3 recommendations
                rec_text = rec_line[1:].strip()  # Remove '•' and whitespace
                
                if rec_text and "No tool usage patterns" not in rec_text:
                    note_id = f"rec_{uuid.uuid4().hex[:8]}"
                    self.user_buffers[user_id]['important'][note_id] = {
                        'notes': rec_text,
                        'timestamp': time.time(),
                        'insight_type': 'recommendation',
                        'related_question': f"Tool recommendation #{i+1}"
                    }
                    insights_count += 1
        
        pins_count = len([item for item in self.user_buffers[user_id]['important'].values() 
                         if item.get('insight_type') == 'pin'])
        recs_count = len([item for item in self.user_buffers[user_id]['important'].values() 
                         if item.get('insight_type') == 'recommendation'])
        
        logger.info(f"INSIGHTS-AGENT: Stored {pins_count} pins + {recs_count} recommendations for user {user_id}")
    
    def _generate_stateful_fallback_insights(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list, existing_insights: str, user_id: str) -> str:
        """Generate stateful insights without using Gemini API - compares against existing insights (two-section format)"""
        logger.info(f"INSIGHTS-AGENT: Using stateful fallback analysis for user {user_id}")
        
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
        if any(word in user_lower for word in ['prompt', 'context', 'assembly', 'debug', 'issue']):
            new_pins.append(f"• **System Inquiry**: User investigating {user_message[:60]}...")
        
        # === RECOMMENDATIONS ANALYSIS ===
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            failed_tools = [tool['tool'] for tool in tool_usage_log if not tool['success']]
            successful_tools = [tool['tool'] for tool in tool_usage_log if tool['success']]
            
            # Tool failure patterns
            if failed_tools:
                new_recommendations.append(f"• **Avoid**: Tools that failed: {', '.join(failed_tools[:2])}")
            
            # Successful tool patterns
            if successful_tools:
                new_recommendations.append(f"• **Prefer**: Successful tools for this user: {', '.join(successful_tools[:2])}")
            
            # Specific tool usage recommendations

            
            if any('perplexity' in tool for tool in tools_used):
                new_recommendations.append(f"• **Web Search**: User benefits from current information lookup")
            
            if any('weather' in tool for tool in tools_used):
                new_recommendations.append(f"• **Weather Tools**: User requires weather information access")
        
        # Agent response analysis for recommendations
        response_lower = agent_response.lower()
        if any(phrase in response_lower for phrase in ['sorry', 'error', 'unable', 'cannot', 'failed']):
            new_recommendations.append(f"• **Retry Strategy**: Current approach failed, consider alternative tools/methods")
        
        # Preserve existing insights and merge intelligently
        final_pins = []
        final_recommendations = []
        
        # Add new insights first (prioritize fresh learnings)
        final_pins.extend(new_pins[:2])  # Max 2 new pins
        final_recommendations.extend(new_recommendations[:2])  # Max 2 new recommendations
        
        # Add most relevant existing insights if we have room
        remaining_pin_slots = 3 - len(final_pins)
        remaining_rec_slots = 3 - len(final_recommendations)
        
        if remaining_pin_slots > 0 and existing_pins:
            # Filter out "no insights" messages and take most recent
            valid_pins = [pin for pin in existing_pins if "No previous conversation insights" not in pin]
            final_pins.extend(valid_pins[:remaining_pin_slots])
        
        if remaining_rec_slots > 0 and existing_recommendations:
            # Filter out "no patterns" messages and take most recent
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
    
    def _generate_fallback_important_notes(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list) -> str:
        """LEGACY: Generate important notes without using Gemini API (stateless fallback)"""
        notes = []
        
        # Analyze user message for key information
        user_lower = user_message.lower()
        
        # Check for personal information or preferences
        if any(word in user_lower for word in ['my name is', 'i am', 'call me', 'i prefer']):
            notes.append(f"• **Personal Info**: User shared: {user_message[:100]}...")
        
        # Check for specific requests or requirements
        if any(word in user_lower for word in ['need', 'want', 'require', 'must', 'should']):
            notes.append(f"• **User Requirement**: {user_message[:80]}...")
        
        # Check for system queries or debugging
        if any(word in user_lower for word in ['prompt', 'context', 'assembly', 'debug', 'issue']):
            notes.append(f"• **System Inquiry**: User investigating {user_message[:60]}...")
        
        # Analyze tool usage for important patterns
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            failed_tools = [tool['tool'] for tool in tool_usage_log if not tool['success']]
            
            if failed_tools:
                notes.append(f"• **System Issue**: Tools failed: {', '.join(failed_tools[:2])}")
            

            
            if any('perplexity' in tool for tool in tools_used):
                notes.append(f"• **Information Search**: User requested current information lookup")
        
        # Check agent response for important facts or limitations
        response_lower = agent_response.lower()
        if any(phrase in response_lower for phrase in ['sorry', 'error', 'unable', 'cannot', 'failed']):
            notes.append(f"• **System Limitation**: Agent encountered issues responding to user request")
        
        # If no specific patterns found, create a general note
        if not notes:
            if len(tool_usage_log) > 0:
                notes.append(f"• **Interaction**: User query with {len(tool_usage_log)} tool operations")
            else:
                notes.append(f"• **Simple Exchange**: Direct Q&A about {user_message[:40]}...")
        
        # Limit to 2 most important notes
        return '\n'.join(notes[:2])

    

    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_type": "ClientAgent",
            "model": "gemini-2.5-flash",
            "api_configured": bool(self.api_key),
            "tools_loaded": len(self.tool_executor.get_loaded_tools()),
            "status": "active"
        }

    async def cleanup(self):
        """Cleanup resources including background tasks and MCP connections"""
        logger.info("ClientAgent cleanup: Starting cleanup process")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for all tasks to complete or be cancelled
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        # Clear the task set
        self._background_tasks.clear()
        
        # Close MCP client connection
        await close_mcp_client()
        
        logger.info("ClientAgent cleanup: Cleanup completed")
    
    def _track_background_task(self, task: asyncio.Task):
        """Track a background task and remove it when done"""
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))
    
    async def __aenter__(self):
        """Enter context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup"""
        await self.cleanup()
        return False
