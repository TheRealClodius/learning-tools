"""
Simple Client Agent using Anthropic Claude API

This agent provides a basic interface for processing user requests using
Anthropic's Claude language model with tool discovery capabilities and thinking streaming.
"""

import anthropic
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv
import re
import yaml

from runtime.tool_executor import ToolExecutor

# Load environment variables from .env.local file
load_dotenv('.env.local')

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
                "name": "memory_conversation_retrieve",
                "description": "Retrieve conversation memories from MemoryOS dual memory system to get user prompts and agent responses with execution linking info.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why this memory retrieval is being performed"
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant memories and context"
                        },
                        "message_id": {
                            "type": "string",
                            "description": "Optional: Specific message ID to retrieve (for linked queries)"
                        },
                        "time_range": {
                            "type": "object",
                            "description": "Optional time range to filter conversations"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["explanation", "query"]
                }
            },

            {
                "name": "memory_execution_retrieve",
                "description": "Retrieve execution memories from MemoryOS dual memory system to learn from past problem-solving approaches, tools used, and error patterns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why this execution retrieval is being performed"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for execution patterns to learn from (e.g., 'weather tool usage', 'error handling strategies')"
                        },
                        "message_id": {
                            "type": "string",
                            "description": "Optional: Specific message ID to get execution details for"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of execution records to return"
                        }
                    },
                    "required": ["explanation", "query"]
                }
            },
            {
                "name": "memory_get_profile",
                "description": "Retrieve user profile information from MemoryOS dual memory system including personality traits, preferences, and knowledge extracted from conversation patterns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why the user profile is being requested"
                        },
                        "include_knowledge": {
                            "type": "boolean",
                            "description": "Whether to include user-specific knowledge entries in the response",
                            "default": True
                        },
                        "include_assistant_knowledge": {
                            "type": "boolean",
                            "description": "Whether to include assistant knowledge base entries in the response",
                            "default": False
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
        
        # Enhanced user buffers for prompt assembly
        self.user_buffers: Dict[str, Dict[str, Any]] = {}
        self.buffer_expiry_minutes = 30  # Buffer expires after 30 minutes
        
        # Load system prompt from YAML configuration
        self.system_prompt = self.config.get('system_prompt', '')
        
        logger.info("ClientAgent initialized with Claude")
    
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
                full_message += f"\n\n[Context: Platform={platform}, Time={timestamp}]"
            
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
        
        if streaming_callback:
            await streaming_callback("Assembling context from conversation history...", "memory")
        
        enriched_message = await self.assemble_from_local_buffer(user_message, user_id)
        
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
        
        # ALWAYS store conversation and update buffer (regardless of tool usage or max iterations)
        logger.info(f"CONTROL-FLOW: Storing memory and updating buffer - Response: {final_response[:100]}...")
        
        if streaming_callback:
            await streaming_callback("Storing conversation to memory...", "memory")
        
        # 1. Store Q&A pair to MemoryOS (immediate)
        message_id = await self._store_conversation_memory(user_message, final_response, user_id)
        
        if streaming_callback:
            await streaming_callback("Updating local conversation buffer...", "memory")
        
        # 2. Update local buffer (immediate)
        self._update_buffer(user_message, final_response, tool_usage_log, all_thinking_content, user_id)
        
        # 3. Process execution details asynchronously (non-blocking)
        if tool_usage_log:  # Only if there were tools used
            asyncio.create_task(self._store_execution_memory(tool_usage_log, all_thinking_content, user_id, message_id))
        
        # 4. Update recent flow and important sections asynchronously (non-blocking)
        asyncio.create_task(self._update_recent_flow(user_id))
        asyncio.create_task(self._update_important(user_message, final_response, tool_usage_log, all_thinking_content, user_id))
        
        return final_response
    
    async def _store_conversation_memory(self, user_message: str, agent_response: str, user_id: str):
        """Store Q&A pair to MemoryOS conversation memory"""
        # Automatically store Q&A pair in memory (unless it's a simple greeting)
        clean_user_message = user_message.strip().lower()
        simple_greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
        
        if not any(greeting in clean_user_message for greeting in simple_greetings):
            logger.info(f"CONVERSATION-MEMORY: Storing Q&A pair for user {user_id}")
            try:
                # Generate unique message ID for linking
                import uuid
                message_id = f"auto_{uuid.uuid4().hex[:12]}"
                
                # Store conversation memory
                await self.tool_executor.execute_command("memory.conversation.add", {
                    "message_id": message_id,
                    "explanation": "Automatic conversation storage with agent response",
                    "user_input": user_message,
                    "agent_response": agent_response.strip(),
                    "meta_data": {
                        "auto_stored": True,
                        "user_id": user_id
                    }
                }, user_id=user_id)
                logger.info(f"CONVERSATION-MEMORY: Successfully stored Q&A pair for user {user_id}")
                return message_id  # Return for linking execution memory
                
            except Exception as e:
                logger.error(f"CONVERSATION-MEMORY: Failed to store Q&A pair for user {user_id}: {e}")
                import traceback
                logger.error(f"CONVERSATION-MEMORY: Traceback: {traceback.format_exc()}")
                return None
        else:
            logger.info(f"CONVERSATION-MEMORY: Skipping storage for greeting: {clean_user_message}")
            return None
    
    async def _generate_run_summary(self, thinking_content: list, tool_usage_log: list) -> str:
        """Generate a 2-line run summary using Gemini Flash 2.5"""
        logger.info("=== RUN SUMMARY GENERATION DEBUG ===")
        logger.info(f"Input thinking_content: {len(thinking_content)} blocks")
        logger.info(f"Input tool_usage_log: {len(tool_usage_log)} entries")
        
        for i, thinking in enumerate(thinking_content):
            logger.info(f"Thinking {i+1}: {thinking[:200]}...")
            
        for i, tool in enumerate(tool_usage_log):
            logger.info(f"Tool {i+1}: {tool['tool']} - Success: {tool['success']} - Preview: {tool['result_preview'][:100]}...")
        
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            logger.info(f"GEMINI: API key status: {'✅ SET' if api_key else '❌ NOT SET'}")
            if api_key:
                logger.info(f"GEMINI: API key length: {len(api_key)}")
                logger.info(f"GEMINI: API key prefix: {api_key[:10]}...")
            
            if not api_key:
                logger.warning("GEMINI_API_KEY not set, using fallback summary")
                fallback = self._fallback_run_summary(thinking_content, tool_usage_log)
                logger.info(f"FALLBACK SUMMARY: {fallback}")
                return fallback
            
            logger.info("GEMINI: Configuring with API key...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Prepare thinking content
            thinking_text = "\n".join(thinking_content) if thinking_content else "No thinking recorded"
            logger.info(f"GEMINI: Thinking text length: {len(thinking_text)}")
            
            # Prepare tool usage summary
            tool_summary = []
            for tool in tool_usage_log:
                status = "✅" if tool["success"] else "❌"
                tool_summary.append(f"{status} {tool['tool']}")
            tool_text = ", ".join(tool_summary) if tool_summary else "No tools used"
            logger.info(f"GEMINI: Tool summary: {tool_text}")
            
            # Create summarization prompt
            prompt = f"""Generate a natural language summary of this agent's problem-solving process (2-3 sentences max):

THINKING PROCESS:
{thinking_text}

TOOLS USED:
{tool_text}

Requirements:
- Describe the approach/strategy used in natural language
- Mention key tools that worked or failed and why
- Include specific error details when tools failed
- Focus on the problem-solving workflow and outcomes
- Be concise but informative

Example format: "Discovered correct tools to use for weather. Ran into a One Call 3.0 limitation for forecasting because system has no paid plan and then got forecast from the web instead."

Generate summary:"""

            logger.info("=== GEMINI PROMPT ===")
            logger.info(prompt)
            logger.info("=== END PROMPT ===")
            
            logger.info("GEMINI: Sending request to Gemini Flash 2.5...")
            response = model.generate_content(prompt)
            summary = response.text.strip()
            logger.info(f"GEMINI: Raw response: {summary}")
            
            # Clean up the summary (remove any extra formatting)
            final_summary = summary.replace('\n', ' ').strip()
            logger.info(f"GEMINI: Final summary: {final_summary}")
            return final_summary
            
        except Exception as e:
            logger.error(f"GEMINI: Summarization failed: {e}")
            import traceback
            logger.error(f"GEMINI: Traceback: {traceback.format_exc()}")
            fallback = self._fallback_run_summary(thinking_content, tool_usage_log)
            logger.info(f"GEMINI: Using fallback: {fallback}")
            return fallback
    
    def _fallback_run_summary(self, thinking_content: list, tool_usage_log: list) -> str:
        """Fallback summary when Gemini is unavailable"""
        logger.info("=== FALLBACK SUMMARY GENERATION ===")
        tool_names = [tool['tool'] for tool in tool_usage_log]
        failed_tools = [tool['tool'] for tool in tool_usage_log if not tool['success']]
        
        logger.info(f"FALLBACK: Tool names: {tool_names}")
        logger.info(f"FALLBACK: Failed tools: {failed_tools}")
        
        if not tool_names:
            final = "Provided direct response without using external tools."
        elif failed_tools:
            final = f"Used tool discovery approach with {len(tool_names)} tools. Encountered failures with {', '.join(failed_tools[:2])} but completed the task successfully."
        else:
            final = f"Successfully used tool discovery approach with {len(tool_names)} tools including {', '.join(tool_names[:3])}."
        
        logger.info(f"FALLBACK: Generated summary: {final}")
        return final
    
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
                await streaming_callback(f"Running {function_name} with args: {str(args)[:100]}...", "tool_details")
            
            # Handle registry tools directly, then try dynamic execution for others
            if function_name == "reg_search":
                if streaming_callback:
                    await streaming_callback(f"Searching registry for: {args.get('query', 'N/A')}", "operation")
                args.setdefault("search_type", "description")
                args.setdefault("limit", 10)
                result = await self.tool_executor.execute_command("reg.search", args)
            elif function_name == "reg_describe":
                if streaming_callback:
                    await streaming_callback(f"Getting tool details for: {args.get('tool_name', 'N/A')}", "operation")
                args.setdefault("include_schema", True)
                result = await self.tool_executor.execute_command("reg.describe", args)
            elif function_name == "reg_list":
                if streaming_callback:
                    await streaming_callback("Listing all available tools", "operation")
                args.setdefault("limit", 50)
                result = await self.tool_executor.execute_command("reg.list", args)
            elif function_name == "reg_categories":
                if streaming_callback:
                    await streaming_callback("Getting tool categories", "operation")
                result = await self.tool_executor.execute_command("reg.categories", args)
            elif function_name == "memory_conversation_retrieve":
                if streaming_callback:
                    await streaming_callback(f"Retrieving conversation memory for query: {args.get('query', 'N/A')[:50]}...", "memory")
                args.setdefault("max_results", 10)
                result = await self.tool_executor.execute_command("memory.conversation.retrieve", args, user_id=user_id)
            elif function_name == "memory_execution_retrieve":
                if streaming_callback:
                    await streaming_callback(f"Retrieving execution memory for query: {args.get('query', 'N/A')[:50]}...", "memory")
                args.setdefault("max_results", 10)
                result = await self.tool_executor.execute_command("memory.execution.retrieve", args, user_id=user_id)
            elif function_name == "memory_get_profile":
                if streaming_callback:
                    await streaming_callback("Getting user profile from memory", "memory")
                args.setdefault("include_knowledge", True)
                args.setdefault("include_assistant_knowledge", False)
                result = await self.tool_executor.execute_command("memory.get_profile", args)
            elif function_name == "execute_tool":
                # Execute any discovered tool dynamically - completely generic
                tool_name = args["tool_name"]
                tool_args = args.get("tool_args", {})
                
                if streaming_callback:
                    await streaming_callback(f"Executing discovered tool: {tool_name}", "operation")
                
                result = await self.tool_executor.execute_command(tool_name, tool_args)
            else:
                error_msg = f"Unknown function: {function_name}. Available tools: reg_search, reg_describe, reg_list, reg_categories, memory_conversation_retrieve, memory_execution_retrieve, memory_get_profile, execute_tool"
                if streaming_callback:
                    await streaming_callback(f"ERROR: {error_msg}", "error")
                return error_msg
            
            logger.info(f"Tool executor result: {result}")
            
            # Stream the result details
            if streaming_callback:
                if isinstance(result, dict):
                    success = result.get("success", False)
                    message = result.get("message", "No message")
                    if success:
                        await streaming_callback(f"✅ {function_name} succeeded: {message}", "result")
                        # Also stream key data if available
                        data = result.get("data", {})
                        if isinstance(data, dict):
                            if "total_results" in data:
                                await streaming_callback(f"Found {data['total_results']} results", "result_detail")
                            elif "answer" in data:
                                await streaming_callback(f"Got answer: {str(data['answer'])[:100]}...", "result_detail")
                    else:
                        await streaming_callback(f"❌ {function_name} failed: {message}", "error")
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
        Assemble enriched prompt with context from local buffer
        
        Args:
            user_message: Current user message
            user_id: User identifier for buffer isolation
            
        Returns:
            Enriched message with conversation and execution context
        """
        logger.info(f"PROMPT-ASSEMBLY: Assembling context for user {user_id}")
        
        # Check if buffer is fresh (30-minute expiry)
        if not self._is_buffer_fresh(user_id):
            logger.info(f"PROMPT-ASSEMBLY: Buffer expired for user {user_id}, using message as-is")
            return user_message
        
        buffer = self.user_buffers.get(user_id, {})
        context_parts = [user_message]
        
        # Prioritize context based on relevance and recency
        conversations = buffer.get('conversations', [])
        executions = buffer.get('executions', [])
        recent_flow = buffer.get('recent_flow', {})
        important = buffer.get('important', {})
        
        # Only add context if it's meaningful and recent
        context_weight = 0
        
        # Add most recent conversation (high priority)
        if conversations:
            last_conv = conversations[-1]
            # Only include if recent (within last hour)
            if time.time() - last_conv.get('timestamp', 0) < 3600:
                context_parts.append("\n=== RECENT CONTEXT ===")
                context_parts.append(f"Previous: {last_conv['user_input'][:100]}{'...' if len(last_conv['user_input']) > 100 else ''}")
                context_parts.append(f"Response: {last_conv['agent_response'][:150]}{'...' if len(last_conv['agent_response']) > 150 else ''}")
                context_weight += 1
        
        # Add execution summary only if tools were used (medium priority)
        if executions and context_weight < 2:
            last_exec = executions[-1]
            if last_exec.get('tools_used'):
                context_parts.append("\n=== TOOLS USED ===")
                context_parts.append(f"Recent tools: {', '.join(last_exec['tools_used'][:3])}")
                if last_exec.get('execution_summary'):
                    context_parts.append(f"Summary: {last_exec['execution_summary'][:200]}...")
                context_weight += 1
        
        # Add flow pattern only if established (low priority)
        if recent_flow.get('summary') and context_weight < 2:
            context_parts.append("\n=== INTERACTION PATTERN ===")
            context_parts.append(f"Recent pattern: {recent_flow['summary'][:200]}...")
            context_weight += 0.5
        
        # Add only most critical important items (high priority)
        if important and context_weight < 2:
            sorted_important = sorted(
                important.items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            )
            # Only show if very recent (within last 30 minutes)
            recent_important = [
                item for item in sorted_important 
                if time.time() - item[1]['timestamp'] < 1800
            ]
            
            if recent_important:
                context_parts.append("\n=== KEY POINTS ===")
                for note_id, note_data in recent_important[:2]:  # Only top 2 most recent
                    notes_text = note_data['notes'][:150] + ('...' if len(note_data['notes']) > 150 else '')
                    context_parts.append(f"• {notes_text}")
                context_weight += 1
        
        enriched_message = "\n".join(context_parts)
        logger.info(f"PROMPT-ASSEMBLY: Assembled {len(context_parts)-1} context sections for user {user_id}")
        
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
        """Update local buffer with conversation and execution data"""
        logger.info(f"BUFFER: Updating buffer for user {user_id}")
        
        # Initialize user buffer if not exists
        if user_id not in self.user_buffers:
            logger.info(f"BUFFER-NEW-USER: Creating new buffer for user_id='{user_id}'")
            self.user_buffers[user_id] = {
                'conversations': [],
                'executions': [],
                'important': {},
                'recent_flow': {},
                'last_updated': time.time()
            }
        else:
            logger.info(f"BUFFER-EXISTING-USER: Using existing buffer for user_id='{user_id}'")
        
        # Add conversation to buffer
        conversation_entry = {
            'user_input': user_message,
            'agent_response': agent_response,
            'timestamp': time.time()
        }
        self.user_buffers[user_id]['conversations'].append(conversation_entry)
        
        # Keep only last 3 conversations
        if len(self.user_buffers[user_id]['conversations']) > 3:
            self.user_buffers[user_id]['conversations'].pop(0)
        
        # Add execution context if there were tools used
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            tool_failures = [
                f"{tool['tool']} error: {tool['result_preview'][:100]}" 
                for tool in tool_usage_log 
                if not tool['success']
            ]
            
            # Determine reasoning approach
            reasoning_approach = "direct_response"
            if len(tools_used) > 0:
                if any("reg_" in tool for tool in tools_used):
                    reasoning_approach = "tool_discovery"
                elif any("memory_conversation_retrieve" in tool or "memory_execution_retrieve" in tool for tool in tools_used):
                    reasoning_approach = "context_retrieval"
                else:
                    reasoning_approach = "tool_execution"
            
            execution_entry = {
                'execution_summary': "Processing execution summary...",  # Will be updated async
                'tools_used': tools_used,
                'errors': tool_failures,
                'observations': f"Used {reasoning_approach} approach with {len(tool_usage_log)} tool(s) and {len(thinking_content)} thinking blocks",
                'success': len(tool_failures) == 0,
                'timestamp': time.time(),
                'reasoning_approach': reasoning_approach
            }
            
            self.user_buffers[user_id]['executions'].append(execution_entry)
            
            # Keep only last 1 execution (since we only use executions[-1])
            if len(self.user_buffers[user_id]['executions']) > 1:
                self.user_buffers[user_id]['executions'].pop(0)
        
        # Update timestamp
        self.user_buffers[user_id]['last_updated'] = time.time()
        logger.info(f"BUFFER: Buffer updated for user {user_id}")
    
    async def _store_execution_memory(self, tool_usage_log: list, thinking_content: list, user_id: str, message_id: str = None):
        """Process execution details and store to MemoryOS"""
        if not tool_usage_log:  # No execution to store
            return
        
        try:
            logger.info(f"EXECUTION-MEMORY: Processing execution details for user {user_id}")
            
            # Generate execution summary + observations using Gemini Flash 2.5
            run_summary = await self._generate_run_summary(thinking_content, tool_usage_log)
            
            # Extract tool usage details
            tools_used = [tool['tool'] for tool in tool_usage_log]
            tool_failures = [
                f"{tool['tool']} error: {tool['result_preview'][:100]}" 
                for tool in tool_usage_log 
                if not tool['success']
            ]
            
            # Determine reasoning approach
            reasoning_approach = "direct_response"
            if len(tools_used) > 0:
                if any("reg_" in tool for tool in tools_used):
                    reasoning_approach = "tool_discovery"
                elif any("memory_conversation_retrieve" in tool or "memory_execution_retrieve" in tool for tool in tools_used):
                    reasoning_approach = "context_retrieval"
                else:
                    reasoning_approach = "tool_execution"
            
            # Store execution memory with detailed execution info
            start_time = getattr(self, '_conversation_start_time', None)
            duration_ms = int((time.time() - start_time) * 1000) if start_time else None
            
            await self.tool_executor.execute_command("memory.execution.add", {
                "message_id": message_id or f"auto_{user_id}_{int(time.time())}",
                "explanation": "Automatic execution details storage with tools and reasoning",
                "execution_summary": run_summary,
                "tools_used": tools_used,
                "errors": [{"error_type": "tool_failure", "error_message": f"Tool {tool} failed", "tool": tool} for tool in tool_failures],
                "observations": f"Used {reasoning_approach} approach with {len(tool_usage_log)} tool(s) and {len(thinking_content)} thinking blocks",
                "success": len(tool_failures) == 0,
                "duration_ms": duration_ms
            }, user_id=user_id)
            
            # Update buffer execution entry with generated summary
            if user_id in self.user_buffers and self.user_buffers[user_id]['executions']:
                self.user_buffers[user_id]['executions'][-1]['execution_summary'] = run_summary
            
            logger.info(f"EXECUTION-MEMORY: Successfully stored execution details for user {user_id}")
            
        except Exception as e:
            logger.error(f"EXECUTION-MEMORY: Failed to store execution details for user {user_id}: {e}")
            import traceback
            logger.error(f"EXECUTION-MEMORY: Traceback: {traceback.format_exc()}")
    
    async def _update_recent_flow(self, user_id: str):
        """Update recent flow summary with sliding window of multiple past turns"""
        try:
            logger.info(f"RECENT-FLOW: Updating recent flow summary for user {user_id}")
            
            if user_id not in self.user_buffers:
                return
            
            conversations = self.user_buffers[user_id].get('conversations', [])
            executions = self.user_buffers[user_id].get('executions', [])
            
            # Only generate recent flow if we have multiple conversations
            if len(conversations) < 2:
                return
            
            # Prepare conversation history for analysis (last 3 conversations)
            conversation_history = []
            for conv in conversations[-3:]:
                conversation_history.append(f"Q: {conv['user_input']}")
                conversation_history.append(f"A: {conv['agent_response']}")
            
            # Prepare execution patterns
            execution_patterns = []
            for exec_entry in executions[-3:] if executions else []:
                if exec_entry.get('tools_used'):
                    execution_patterns.append(f"Tools: {', '.join(exec_entry['tools_used'])}")
                if exec_entry.get('reasoning_approach'):
                    execution_patterns.append(f"Approach: {exec_entry['reasoning_approach']}")
            
            # Generate flow summary using Gemini Flash 2.5
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("RECENT-FLOW: GEMINI_API_KEY not set, using fallback summary")
                recent_flow_summary = self._generate_fallback_flow_summary(conversations, executions)
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                prompt = f"""Analyze this conversation flow and create a brief summary of the recent interaction pattern (2-3 sentences max):

RECENT CONVERSATIONS:
{chr(10).join(conversation_history)}

EXECUTION PATTERNS:
{chr(10).join(execution_patterns)}

Create a summary that captures:
- The overall conversation theme or topic progression
- Problem-solving approach or tool usage patterns
- Any evolving user needs or context

Format: Natural language summary focusing on flow and patterns, not individual details.

Generate summary:"""

                try:
                    response = model.generate_content(prompt)
                    recent_flow_summary = response.text.strip().replace('\n', ' ')
                except Exception as gemini_error:
                    logger.warning(f"RECENT-FLOW: Gemini API failed ({gemini_error}), using fallback")
                    recent_flow_summary = self._generate_fallback_flow_summary(conversations, executions)
            
            # Update buffer
            self.user_buffers[user_id]['recent_flow'] = {
                'summary': recent_flow_summary,
                'conversation_count': len(conversations),
                'last_updated': time.time()
            }
            
            logger.info(f"RECENT-FLOW: Updated recent flow for user {user_id}")
            
        except Exception as e:
            logger.error(f"RECENT-FLOW: Failed to update recent flow for user {user_id}: {e}")
            import traceback
            logger.error(f"RECENT-FLOW: Traceback: {traceback.format_exc()}")
    
    def _generate_fallback_flow_summary(self, conversations: list, executions: list) -> str:
        """Generate a simple flow summary without using Gemini API"""
        if not conversations:
            return "No conversation history available"
        
        # Analyze conversation patterns
        recent_convs = conversations[-3:]  # Last 3 conversations
        topics = []
        
        for conv in recent_convs:
            user_input = conv.get('user_input', '').lower()
            
            # Extract key topics based on common patterns
            if any(word in user_input for word in ['weather', 'temperature', 'forecast']):
                topics.append('weather')
            elif any(word in user_input for word in ['search', 'find', 'lookup', 'information']):
                topics.append('information_search')
            elif any(word in user_input for word in ['memory', 'remember', 'recall', 'context']):
                topics.append('memory_operations')
            elif any(word in user_input for word in ['help', 'assist', 'support']):
                topics.append('assistance')
            elif any(word in user_input for word in ['prompt', 'assembly', 'stack', 'context']):
                topics.append('system_inquiry')
            else:
                topics.append('general_inquiry')
        
        # Analyze execution patterns
        tool_usage = []
        if executions:
            recent_exec = executions[-1]
            tools_used = recent_exec.get('tools_used', [])
            if tools_used:
                if any('memory' in tool for tool in tools_used):
                    tool_usage.append('memory_retrieval')
                if any('perplexity' in tool for tool in tools_used):
                    tool_usage.append('web_search')
                if any('weather' in tool for tool in tools_used):
                    tool_usage.append('weather_data')
                if any('reg_' in tool for tool in tools_used):
                    tool_usage.append('tool_discovery')
        
        # Generate summary
        unique_topics = list(set(topics))
        topic_text = ', '.join(unique_topics[:2]) if unique_topics else 'general discussion'
        
        if tool_usage:
            tool_text = f" using {', '.join(set(tool_usage))}"
        else:
            tool_text = " with direct responses"
        
        conversation_count = len(conversations)
        
        return f"Recent pattern: {conversation_count} conversations focused on {topic_text}{tool_text}. User appears to be exploring system capabilities and memory functionality."
    
    async def _update_important(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list, user_id: str):
        """Generate important cliffnotes from Q&A pair and execution trace"""
        try:
            logger.info(f"IMPORTANT: Generating important cliffnotes for user {user_id}")
            
            # Only generate important notes if there's significant activity
            if not tool_usage_log and len(user_message.split()) < 10:
                return
            
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
            
            # Generate important notes using Gemini Flash 2.5
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("IMPORTANT: GEMINI_API_KEY not set, using fallback notes")
                important_notes = self._generate_fallback_important_notes(user_message, agent_response, tool_usage_log, thinking_content)
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                prompt = f"""Extract the most important highlights from this interaction for future reference (1-2 bullet points max):

USER QUESTION:
{user_message}

AGENT RESPONSE:
{agent_response}

EXECUTION DETAILS:
{execution_summary}

AGENT THINKING:
{thinking_summary}

Create concise cliffnotes that capture:
- Key user needs or specific requirements mentioned
- Important facts, preferences, or constraints discovered
- Critical errors or limitations encountered
- Significant achievements or successful approaches

Format: Bullet points, focus on what's worth remembering for future interactions.

Generate important notes:"""

                try:
                    response = model.generate_content(prompt)
                    important_notes = response.text.strip()
                except Exception as gemini_error:
                    logger.warning(f"IMPORTANT: Gemini API failed ({gemini_error}), using fallback")
                    important_notes = self._generate_fallback_important_notes(user_message, agent_response, tool_usage_log, thinking_content)
            
            # Update buffer (keep only recent important items)
            if user_id not in self.user_buffers:
                return
            
            current_important = self.user_buffers[user_id].get('important', {})
            
            # Add new important item with timestamp
            import uuid
            note_id = f"note_{uuid.uuid4().hex[:8]}"
            current_important[note_id] = {
                'notes': important_notes,
                'timestamp': time.time(),
                'related_question': user_message[:100]  # First 100 chars for context
            }
            
            # Keep only last 5 important items (sliding window)
            if len(current_important) > 5:
                oldest_key = min(current_important.keys(), key=lambda k: current_important[k]['timestamp'])
                del current_important[oldest_key]
            
            self.user_buffers[user_id]['important'] = current_important
            
            logger.info(f"IMPORTANT: Generated important notes for user {user_id}")
            
        except Exception as e:
            logger.error(f"IMPORTANT: Failed to generate important notes for user {user_id}: {e}")
            import traceback
            logger.error(f"IMPORTANT: Traceback: {traceback.format_exc()}")
    
    def _generate_fallback_important_notes(self, user_message: str, agent_response: str, tool_usage_log: list, thinking_content: list) -> str:
        """Generate important notes without using Gemini API"""
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
        if any(word in user_lower for word in ['prompt', 'memory', 'context', 'assembly', 'debug', 'issue']):
            notes.append(f"• **System Inquiry**: User investigating {user_message[:60]}...")
        
        # Analyze tool usage for important patterns
        if tool_usage_log:
            tools_used = [tool['tool'] for tool in tool_usage_log]
            failed_tools = [tool['tool'] for tool in tool_usage_log if not tool['success']]
            
            if failed_tools:
                notes.append(f"• **System Issue**: Tools failed: {', '.join(failed_tools[:2])}")
            
            if any('memory' in tool for tool in tools_used):
                notes.append(f"• **Memory Operations**: User accessing conversation history and context")
            
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
