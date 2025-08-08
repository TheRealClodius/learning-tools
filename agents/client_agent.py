"""
Simple Client Agent using Anthropic Claude API

This agent provides a basic interface for processing user requests using
Anthropic's Claude language model with tool discovery capabilities and thinking streaming.
"""

import anthropic
import logging
import os
import time
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv
import re
import yaml

from runtime.tool_executor import ToolExecutor
from agents.convo_insights_agent import ConvoInsightsAgent

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
        
        # Initialize Claude client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Load configuration from YAML file
        self.config = self._load_config()
        
        # Model configuration
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = self.config.get('model_config', {}).get('max_tokens', 4096)
        self.temperature = self.config.get('model_config', {}).get('temperature', 0.7)
        self.max_iterations = self.config.get('model_config', {}).get('max_iterations', 50)

        
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
            }
        ]
        
        # Initialize tool executor for dynamic tool discovery
        self.tool_executor = ToolExecutor()
        
        # Initialize conversation insights agent
        self.insights_agent = ConvoInsightsAgent()
        
        # Enhanced user buffers for prompt assembly
        self.user_buffers: Dict[str, Dict[str, Any]] = {}
        self.buffer_expiry_minutes = 30  # Buffer expires after 30 minutes
        
        # Load system prompt from YAML configuration
        self.system_prompt = self.config.get('system_prompt', '')
        
        logger.info("ClientAgent initialized with Claude and ConvoInsightsAgent")
    
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
        Process user request using Claude API with agent loop for tool discovery
        
        Args:
            message: User's message/request
            context: Additional context (platform, timestamp, etc.)
            streaming_callback: Optional callback for streaming thinking process
            
        Returns:
            Dict containing response data
        """
        try:
            # Extract user_id from context and add basic context information
            user_id = context.get('user_id') if context else None
            
            # DEBUG: Log user_id flow in ClientAgent
            logger.info(f"CLIENT-AGENT-USER-ID: Received user_id='{user_id}' from context")
            logger.info(f"CLIENT-AGENT-CONTEXT: Full context={context}")
            
            full_message = message
            if context:
                platform = context.get('platform', 'unknown')
                timestamp = context.get('timestamp', 'unknown')
                user_timezone = context.get('user_timezone', '')
                
                if user_timezone and user_timezone != 'UTC':
                    full_message += f"\n\n[Context: Platform={platform}, Current time for user: {timestamp} ({user_timezone})]"
                else:
                    full_message += f"\n\n[Context: Platform={platform}, Current time: {timestamp}]"
            
            # Run the agent loop with iterative tool calling and user isolation
            logger.info(f"CLIENT-AGENT-BUFFER: Calling run_agent_loop with user_id='{user_id}'")
            response = await self.run_agent_loop(full_message, streaming_callback, user_id)
            
            return {
                "success": True,
                "response": response,
                "message": "Request processed successfully",
                "agent": "client_agent",
                "model": "claude-3-5-sonnet"
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
    

    async def run_agent_loop(self, user_message: str, streaming_callback=None, user_id: Optional[str] = None) -> str:
        """Run the agent loop with iterative tool calling"""
        # Enhanced prompt assembly from local buffer
        if not user_id:
            raise ValueError("user_id is required for agent processing - cannot proceed without user identification")
        
        # DEBUG: Log buffer system usage
        logger.info(f"BUFFER-SYSTEM: Using user_id='{user_id}' for prompt assembly")
        logger.info(f"BUFFER-SYSTEM: Current buffer users={list(self.user_buffers.keys())}")
        
        # MEMORY MANAGEMENT: Retrieve conversation history first
        if streaming_callback:
            await streaming_callback("Retrieving conversation history from memory...", "operation")
        
        memory_context = await self._retrieve_memory_context(user_message, user_id, streaming_callback)
        
        if streaming_callback:
            await streaming_callback("Assembling context from conversation history...", "operation")
        
        enriched_message = await self.assemble_from_local_buffer(user_message, user_id)
        
        # Add memory context to the enriched message if available
        if memory_context:
            enriched_message = f"{enriched_message}\n\nRelevant conversation history:\n{memory_context}"
        
        # DEBUG: Log enriched message to detect thinking-only issues
        if enriched_message != user_message:
            logger.info(f"ENRICHED-PROMPT-DEBUG: User {user_id} prompt was enriched")
            logger.info(f"ENRICHED-CONTENT: {enriched_message}")
        
        messages = [{"role": "user", "content": enriched_message}]
        max_iterations = self.max_iterations  # Load from config
        
        # Capture thinking content and tool usage for run summary
        all_thinking_content = []
        tool_usage_log = []
        final_response = None
        final_iteration = 0
        
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}")
            
            if streaming_callback:
                await streaming_callback(f"Agent iteration {iteration + 1}", "status")
            
            # Generate response using Claude
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.tools,
                    timeout=60.0  # Add timeout to prevent streaming errors
                )
            )
            
            logger.info(f"Claude response iteration {iteration + 1}: {response}")
            
            # Process response content
            assistant_content = []
            tool_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    text = content_block.text
                    assistant_content.append({"type": "text", "text": text})
                    
                    # Capture thinking content for run summary
                    thinking_matches = re.findall(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
                    for thinking in thinking_matches:
                        all_thinking_content.append(thinking.strip())
                    
                    # Extract and stream thinking content
                    if streaming_callback:
                        await self._extract_and_stream_thinking(text, streaming_callback)
                
                elif content_block.type == "thinking":
                    # Stream Anthropic native thinking blocks if present
                    thinking_text = getattr(content_block, "text", None) or getattr(content_block, "thinking", None) or str(content_block)
                    if thinking_text:
                        all_thinking_content.append(thinking_text.strip())
                        if streaming_callback:
                            for line in thinking_text.strip().split('\n'):
                                line = line.strip()
                                if line:
                                    await streaming_callback(line, "thinking")
                
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
                break
            
            # Execute tool calls and add results
            tool_results = []
            
            for tool_call in tool_calls:
                if streaming_callback:
                    await streaming_callback(tool_call.name, "tool_start")
                
                result = await self._execute_claude_tool(tool_call, streaming_callback, user_id)
                
                # Log tool usage for run summary
                tool_usage_log.append({
                    "tool": tool_call.name,
                    "args": tool_call.input,
                    "success": "Error:" not in str(result),
                    "result_preview": str(result)[:200]
                })
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": str(result)
                })
                
                # Note: Tools are now discovered purely through registry search
                # All execution is handled dynamically by tool_executor
                
                if streaming_callback:
                    await streaming_callback(f"{tool_call.name}: {str(result)[:100]}...", "tool_result")
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
        
        # If we reach here, max iterations was hit
        if final_response is None:
            final_response = "Maximum iterations reached. Unable to complete the request."
            final_iteration = max_iterations
        
        # Update local buffer for context management
        logger.info(f"CONTROL-FLOW: Updating local buffer - Response: {final_response[:100]}...")
        
        if streaming_callback:
            await streaming_callback("Updating local conversation buffer...", "operation")
        
        # Update local buffer (immediate)
        self._update_buffer(user_message, final_response, tool_usage_log, all_thinking_content, user_id)
        
        # Update insights asynchronously using Conversation Insights Agent (non-blocking)
        asyncio.create_task(self.insights_agent.analyze_interaction(
            user_message, final_response, tool_usage_log, all_thinking_content, user_id, self.user_buffers
        ))
        
        # MEMORY MANAGEMENT: Store conversation in persistent memory
        if streaming_callback:
            await streaming_callback("Storing conversation in memory...", "operation")
        
        asyncio.create_task(self._store_memory_conversation(user_message, final_response, user_id))
        
        return final_response
    

    

    
    # Removed dynamic tool loading methods - agent now relies purely on registry discovery
    # All tool execution is handled by tool_executor with dynamic loading
    
    async def _extract_and_stream_thinking(self, text: str, streaming_callback):
        """Extract thinking content and stream it"""
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
        
        for thinking_content in thinking_matches:
            # Split thinking into lines and stream each
            lines = thinking_content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    await streaming_callback(line, "thinking")
                    await asyncio.sleep(0.3)  # Brief pause between thoughts
    
    async def _execute_claude_tool(self, tool_use_block, streaming_callback=None, user_id: str = None) -> str:
        """Execute a Claude tool call and return the result"""
        try:
            function_name = tool_use_block.name
            args = tool_use_block.input
            
            logger.info(f"Executing Claude tool: {function_name}, Args: {args}")
            
            if streaming_callback:
                await streaming_callback(f"Running {function_name} with args: {str(args)[:200]}...", "tool_details")
            
            # Handle registry tools directly, then try dynamic execution for others
            if function_name == "reg_search":
                if streaming_callback:
                    await streaming_callback(f"Searching registry for: {args.get('query', 'N/A')}", "operation")
                args.setdefault("search_type", "description")
                args.setdefault("limit", 10)
                result = await self.tool_executor.execute_command("reg.search", args, user_id=user_id)
            elif function_name == "reg_describe":
                if streaming_callback:
                    await streaming_callback(f"Getting tool details for: {args.get('tool_name', 'N/A')}", "operation")
                args.setdefault("include_schema", True)
                result = await self.tool_executor.execute_command("reg.describe", args, user_id=user_id)
            elif function_name == "reg_list":
                if streaming_callback:
                    await streaming_callback("Listing all available tools", "operation")
                args.setdefault("limit", 50)
                result = await self.tool_executor.execute_command("reg.list", args, user_id=user_id)
            elif function_name == "reg_categories":
                if streaming_callback:
                    await streaming_callback("Getting tool categories", "operation")
                result = await self.tool_executor.execute_command("reg.categories", args, user_id=user_id)


            elif function_name == "execute_tool":
                # Execute any discovered tool dynamically - completely generic
                tool_name = args["tool_name"]
                tool_args = args.get("tool_args", {})
                
                if streaming_callback:
                    await streaming_callback(f"Executing discovered tool: {tool_name}", "operation")
                
                result = await self.tool_executor.execute_command(tool_name, tool_args, user_id=user_id)
            else:
                error_msg = f"Unknown function: {function_name}. Available tools: reg_search, reg_describe, reg_list, reg_categories, execute_tool"
                if streaming_callback:
                    await streaming_callback(error_msg, "error")
                return error_msg
            
            logger.info(f"Tool executor result: {result}")
            
            # Stream the result details
            if streaming_callback:
                if isinstance(result, dict):
                    success = result.get("success", False)
                    message = result.get("message", "No message")
                    if success:
                        await streaming_callback(f"{function_name} succeeded: {message}", "result")
                        # Also stream key data if available
                        data = result.get("data", {})
                        if isinstance(data, dict):
                            if "total_results" in data:
                                await streaming_callback(f"Found {data['total_results']} results", "result_detail")
                            elif "answer" in data:
                                # Clean up answer display - remove think tags and show clean preview
                                answer_text = str(data['answer'])
                                # Remove <think> tags and their content
                                import re
                                clean_answer = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL)
                                clean_answer = clean_answer.strip()
                                preview = clean_answer[:150] + "..." if len(clean_answer) > 150 else clean_answer
                                await streaming_callback(f"Retrieved analysis: {preview}", "result_detail")
                    else:
                        await streaming_callback(f"{function_name} failed: {message}", "error")
                else:
                    await streaming_callback(f"Got result: {str(result)[:150]}...", "result")
            
            # Return JSON-serializable result
            if isinstance(result, dict):
                # Format the result nicely for display
                if "success" in result and result["success"]:
                    data = result.get("data", result.get("result", "No data"))
                    return str(data)
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return str(result) if result is not None else "No result returned"
                
        except Exception as e:
            logger.error(f"Claude tool execution error: {e}")
            return f"Error executing {function_name}: {str(e)}"
    

    # =============================================================================
    # ENHANCED BUFFER SYSTEM FOR PROMPT ASSEMBLY
    # =============================================================================
    
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

    
    async def _retrieve_memory_context(self, user_message: str, user_id: str, streaming_callback=None) -> str:
        """Retrieve relevant conversation history from memory"""
        try:
            # First get recent conversation history
            recent_memory_args = {
                "query": "recent conversation history", 
                "user_id": user_id,
                "max_results": 5
            }
            
            recent_result = await self.tool_executor.execute_command("memory.retrieve", recent_memory_args, user_id=user_id)
            
            # Then get semantically relevant history based on current message
            semantic_query = f"previous discussion about {user_message[:100]}"
            semantic_memory_args = {
                "query": semantic_query,
                "user_id": user_id, 
                "max_results": 3
            }
            
            semantic_result = await self.tool_executor.execute_command("memory.retrieve", semantic_memory_args, user_id=user_id)
            
            # Format the memory context
            context_parts = []
            
            if isinstance(recent_result, dict) and recent_result.get("success"):
                data = recent_result.get("data", {})
                short_term = data.get("short_term_memory", [])
                if short_term:
                    context_parts.append("Recent conversations:")
                    for i, conv in enumerate(short_term[-3:]):  # Last 3 recent conversations
                        user_msg = conv.get("user", "")[:100]
                        agent_msg = conv.get("assistant", "")[:100] 
                        context_parts.append(f"  {i+1}. User: {user_msg}...")
                        context_parts.append(f"     Assistant: {agent_msg}...")
            
            if isinstance(semantic_result, dict) and semantic_result.get("success"):
                data = semantic_result.get("data", {})
                mid_term = data.get("mid_term_memory", [])
                if mid_term:
                    context_parts.append("\nRelevant past discussions:")
                    for i, conv in enumerate(mid_term[:2]):  # Top 2 relevant conversations
                        user_msg = conv.get("user", "")[:100]
                        agent_msg = conv.get("assistant", "")[:100]
                        context_parts.append(f"  {i+1}. User: {user_msg}...")
                        context_parts.append(f"     Assistant: {agent_msg}...")
            
            context = "\n".join(context_parts) if context_parts else ""
            
            if streaming_callback and context:
                await streaming_callback(f"Retrieved {len(context_parts)} memory items", "result")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory context: {e}")
            if streaming_callback:
                await streaming_callback(f"Memory retrieval failed: {e}", "error")
            return ""
    
    async def _store_memory_conversation(self, user_message: str, agent_response: str, user_id: str):
        """Store the conversation in persistent memory"""
        try:
            memory_args = {
                "user_input": user_message,
                "agent_response": agent_response,
                "user_id": user_id
            }
            
            result = await self.tool_executor.execute_command("memory.add", memory_args, user_id=user_id)
            
            if isinstance(result, dict) and result.get("success"):
                logger.info(f"Successfully stored conversation in memory for user {user_id}")
            else:
                logger.warning(f"Failed to store conversation in memory: {result}")
                
        except Exception as e:
            logger.error(f"Error storing conversation in memory: {e}")
    
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_type": "ClientAgent",
            "model": "gemini-2.5-flash",
            "api_configured": bool(self.api_key),
            "tools_loaded": len(self.tool_executor.get_loaded_tools()),
            "status": "active"
        }
