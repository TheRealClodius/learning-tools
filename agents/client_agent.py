"""
Simple Client Agent using Anthropic Claude API

This agent provides a basic interface for processing user requests using
Anthropic's Claude language model with tool discovery capabilities and thinking streaming.
"""

import anthropic
import logging
import os
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
                "name": "memory_add",
                "description": "Store a conversation pair in MemoryOS for building persistent dialogue history and contextual memory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why this memory is being stored"
                        },
                        "user_input": {
                            "type": "string",
                            "description": "The user's input, question, or message to be stored"
                        },
                        "agent_response": {
                            "type": "string",
                            "description": "The agent's response or reply to the user input"
                        },
                        "timestamp": {
                            "type": "string",
                            "description": "Optional timestamp in YYYY-MM-DD HH:MM:SS format"
                        },
                        "meta_data": {
                            "type": "object",
                            "description": "Optional metadata about the conversation context"
                        }
                    },
                    "required": ["explanation", "user_input", "agent_response"]
                }
            },
            {
                "name": "memory_retrieve",
                "description": "Retrieve relevant memories and context from MemoryOS based on a query to provide historical context",
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
                        "relationship_with_user": {
                            "type": "string",
                            "enum": ["friend", "assistant", "colleague", "professional", "casual"],
                            "description": "The relationship context between agent and user"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return from each memory type"
                        }
                    },
                    "required": ["explanation", "query"]
                }
            },
            {
                "name": "memory_profile",
                "description": "Get user profile and knowledge extracted from conversation analysis by MemoryOS",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why the user profile is being requested"
                        },
                        "include_knowledge": {
                            "type": "boolean",
                            "description": "Whether to include user-specific knowledge entries"
                        },
                        "include_assistant_knowledge": {
                            "type": "boolean",
                            "description": "Whether to include assistant knowledge base entries"
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
            # Add context information if available
            full_message = message
            if context:
                platform = context.get('platform', 'unknown')
                timestamp = context.get('timestamp', 'unknown')
                full_message += f"\n\n[Context: Platform={platform}, Time={timestamp}]"
            
            # Run the agent loop with iterative tool calling
            response = await self.run_agent_loop(full_message, streaming_callback)
            
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
    
    async def _generate_response(self, prompt: str, streaming_callback=None) -> str:
        """Generate response using Claude API with function calling and thinking streaming"""
        try:
            # Create messages for Claude
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
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
            
            logger.info(f"Claude response: {response}")
            
            # Extract thinking content and stream it
            full_response = ""
            if response.content:
                for content_block in response.content:
                    if content_block.type == "text":
                        text = content_block.text
                        full_response += text
                        
                        # Extract and stream thinking content
                        if streaming_callback:
                            await self._extract_and_stream_thinking(text, streaming_callback)
                    
                    elif content_block.type == "tool_use":
                        # Execute tool calls
                        tool_result = await self._execute_claude_tool(content_block, streaming_callback)
                        full_response += f"\n\nTool result: {tool_result}"
            
            # Remove thinking tags from final response
            clean_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL)
            return clean_response.strip()
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    async def run_agent_loop(self, user_message: str, streaming_callback=None) -> str:
        """Run the agent loop with iterative tool calling"""
        messages = [{"role": "user", "content": user_message}]
        max_iterations = self.max_iterations  # Load from config
        
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}")
            
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
            
            # If no tool calls, we're done
            if not tool_calls:
                # Extract final text response
                final_text = ""
                for content in assistant_content:
                    if content["type"] == "text":
                        final_text += content["text"]
                
                # Remove thinking tags from final response
                clean_response = re.sub(r'<thinking>.*?</thinking>', '', final_text, flags=re.DOTALL)
                return clean_response.strip()
            
            # Execute tool calls and add results
            tool_results = []
            
            for tool_call in tool_calls:
                if streaming_callback:
                    await streaming_callback(f"🔧 Executing {tool_call.name}...", "thinking")
                
                result = await self._execute_claude_tool(tool_call, streaming_callback)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": str(result)
                })
                
                # Note: Tools are now discovered purely through registry search
                # All execution is handled dynamically by tool_executor
                
                if streaming_callback:
                    await streaming_callback(f"📋 Got result: {str(result)[:100]}...", "thinking")
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
        
        return "Maximum iterations reached. Unable to complete the request."
    
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
    
    async def _execute_claude_tool(self, tool_use_block, streaming_callback=None) -> str:
        """Execute a Claude tool call and return the result"""
        try:
            function_name = tool_use_block.name
            args = tool_use_block.input
            
            logger.info(f"Executing Claude tool: {function_name}, Args: {args}")
            
            # Handle registry tools directly, then try dynamic execution for others
            if function_name == "reg_search":
                args.setdefault("search_type", "description")
                args.setdefault("limit", 10)
                result = await self.tool_executor.execute_command("reg.search", args)
            elif function_name == "reg_describe":
                args.setdefault("include_schema", True)
                result = await self.tool_executor.execute_command("reg.describe", args)
            elif function_name == "reg_list":
                args.setdefault("limit", 50)
                result = await self.tool_executor.execute_command("reg.list", args)
            elif function_name == "reg_categories":
                result = await self.tool_executor.execute_command("reg.categories", args)
            elif function_name == "memory_add":
                result = await self.tool_executor.execute_command("memory.add", args)
            elif function_name == "memory_retrieve":
                args.setdefault("relationship_with_user", "friend")
                args.setdefault("max_results", 10)
                result = await self.tool_executor.execute_command("memory.retrieve", args)
            elif function_name == "memory_profile":
                args.setdefault("include_knowledge", True)
                args.setdefault("include_assistant_knowledge", False)
                result = await self.tool_executor.execute_command("memory.profile", args)
            elif function_name == "execute_tool":
                # Execute any discovered tool dynamically - completely generic
                tool_name = args["tool_name"]
                tool_args = args.get("tool_args", {})
                
                result = await self.tool_executor.execute_command(tool_name, tool_args)
            else:
                return f"Unknown function: {function_name}. Available tools: reg_search, reg_describe, reg_list, reg_categories, memory_add, memory_retrieve, memory_profile, execute_tool"
            
            logger.info(f"Tool executor result: {result}")
            
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
    
    async def _execute_function(self, function_call, streaming_callback=None) -> str:
        """Execute a function call and return JSON-serializable result"""
        try:
            # Debug logging
            logger.info(f"Executing function call: {function_call}")
            
            function_name = getattr(function_call, 'name', None)
            if not function_name:
                return "Error: Function call has no name"
                
            args = dict(getattr(function_call, 'args', {}))
            logger.info(f"Function: {function_name}, Args: {args}")
            
            # Stream thinking about what function is being executed
            if streaming_callback:
                tool_descriptions = {
                    "reg_search": "📋 Searching tool registry",
                    "reg_describe": "📖 Getting tool details", 
                    "reg_list": "📝 Listing all tools",
                    "reg_categories": "📂 Getting tool categories",
                    "execute_tool": "🔧 Executing discovered tool"
                }
                description = tool_descriptions.get(function_name, f"⚙️ Executing {function_name}")
                await streaming_callback(description, "thinking")
            
            # Only handle registry tools directly - all others should be discovered dynamically
            if function_name == "reg_search":
                args.setdefault("search_type", "description")
                args.setdefault("limit", 10)
                result = await self.tool_executor.execute_command("reg.search", args)
            elif function_name == "reg_describe":
                args.setdefault("include_schema", True)
                result = await self.tool_executor.execute_command("reg.describe", args)
            elif function_name == "reg_list":
                args.setdefault("limit", 50)
                result = await self.tool_executor.execute_command("reg.list", args)
            elif function_name == "reg_categories":
                result = await self.tool_executor.execute_command("reg.categories", args)
            elif function_name == "execute_tool":
                # Execute any discovered tool dynamically - completely generic
                tool_name = args["tool_name"]
                tool_args = args.get("tool_args", {})
                
                result = await self.tool_executor.execute_command(tool_name, tool_args)
            else:
                return f"Unknown function: {function_name}. Available tools: reg_search, reg_describe, reg_list, reg_categories, execute_tool"
            
            logger.info(f"Tool executor result: {result}")
            
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
            logger.error(f"Function execution error: {e}")
            function_name = getattr(function_call, 'name', 'unknown')
            return f"Error executing {function_name}: {str(e)}"
    

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_type": "ClientAgent",
            "model": "gemini-2.5-flash",
            "api_configured": bool(self.api_key),
            "tools_loaded": len(self.tool_executor.get_loaded_tools()),
            "status": "active"
        }
