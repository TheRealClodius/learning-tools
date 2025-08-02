from typing import Dict, Any, Callable, List, Optional, Set
import logging
import importlib
from pathlib import Path

from execute import execute_reg_search_input
from tools.registry import registry_search, registry_describe, registry_list, registry_categories

logger = logging.getLogger(__name__)

class ToolNotFoundError(Exception):
    """Raised when a requested tool is not available"""
    pass

class ToolExecutor:
    """
    Main tool execution engine that handles:
    - Dynamic tool discovery via registry
    - Tool loading and execution
    - Service management
    """
    
    def __init__(self):
        # Only registry tools loaded at startup
        self.available_tools: Dict[str, Callable] = {}
        self.loaded_services: Set[str] = set()
        
        # Tool type mappings for dynamic loading
        self.service_tool_mappings = {
            "weather": {
                "weather.search": "tools.weather:weather_search",
                "weather.current": "tools.weather:weather_current", 
                "weather.forecast": "tools.weather:weather_forecast"
            },
            "perplexity": {
                "perplexity.search": "tools.perplexity:perplexity_search",
                "perplexity.research": "tools.perplexity:perplexity_research"
            },
            "memory": {
                # Dual memory system functions
                "memory.conversation.add": "tools.memory:conversation_add",
                "memory.conversation.retrieve": "tools.memory:conversation_retrieve",
                "memory.execution.add": "tools.memory:execution_add",
                "memory.execution.retrieve": "tools.memory:execution_retrieve",
                "memory.get_profile": "tools.memory:profile_retrieve"
            },
            "slack": {
                "slack.send_message": "tools.slack:slack_send_message",
                "slack.search_channels": "tools.slack:slack_search_channels"
            }
        }
        
        # Load essential tools at startup (registry + memory)
        self._load_registry_tools()
        self._load_memory_tools()
    
    def _load_registry_tools(self):
        """Load registry tools that are always available"""
        try:
            from tools.registry import (
                registry_search, 
                registry_describe, 
                registry_list, 
                registry_categories
            )
            
            self.available_tools.update({
                "reg.search": registry_search,
                "reg.describe": registry_describe,
                "reg.list": registry_list,
                "reg.categories": registry_categories
            })
            
            self.loaded_services.add("registry")
            logger.info("Registry tools loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not load registry tools: {e}")
    
    def _load_memory_tools(self):
        """Load memory tools that are always available"""
        try:
            from tools.memory import (
                conversation_add,
                conversation_retrieve,
                execution_add,
                execution_retrieve,
                profile_retrieve
            )
            
            self.available_tools.update({
                "memory.conversation.add": conversation_add,
                "memory.conversation.retrieve": conversation_retrieve,
                "memory.execution.add": execution_add,
                "memory.execution.retrieve": execution_retrieve,
                "memory.get_profile": profile_retrieve
            })
            
            self.loaded_services.add("memory")
            logger.info("Dual memory system tools loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not load memory tools: {e}")
    
    async def execute_command(self, command: str, input_data: Any, user_id: str = None) -> Any:
        """
        Main command execution flow with dynamic discovery
        
        Args:
            command: Tool command like "weather.current" or "reg.search"  
            input_data: Input data (Pydantic model instance)
            user_id: Dynamic user ID for memory operations (optional)
            
        Returns:
            Tool execution result (Pydantic model instance)
        """
        logger.info(f"Executing command: {command}")
        
        # Check if tool is already loaded
        if command in self.available_tools:
            return await self._execute_tool(command, input_data, user_id)
        
        # Parse service and action
        try:
            service, action = command.split(".", 1)
        except ValueError:
            raise ToolNotFoundError(f"Invalid command format: {command}")
        
        # Dynamic discovery for unknown services
        if service not in self.loaded_services:
            await self._discover_and_load_service(service)
        
        # Execute the command after loading
        if command in self.available_tools:
            return await self._execute_tool(command, input_data, user_id)
        else:
            raise ToolNotFoundError(f"Command {command} not available after discovery")
    
    async def _execute_tool(self, command: str, input_data: Any, user_id: str = None) -> Any:
        """Execute a loaded tool with error handling"""
        try:
            tool_func = self.available_tools[command]
            
            # For memory tools, require user_id - no silent fallbacks
            if command.startswith("memory.") and command.endswith((".retrieve", ".add")):
                if not user_id:
                    raise ValueError(f"user_id is required for memory operation {command} - cannot proceed without user identification")
                logger.info(f"TOOL-EXECUTOR: Passing user_id='{user_id}' to {command}")
                result = await tool_func(input_data, user_id)
            else:
                result = await tool_func(input_data)
                
            logger.info(f"Successfully executed {command}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing {command}: {str(e)}")
            raise
    
    async def _discover_and_load_service(self, service: str):
        """Use registry to discover and load service tools"""
        logger.info(f"Discovering tools for service: {service}")
        
        if "reg.search" not in self.available_tools:
            logger.warning("Registry not available for discovery")
            return
        
        try:
            # Import registry input model (when execute.py exists)
            # from execute import registry_search_input
            
            # Search for tools by service category
            discovery_query = {
                "explanation": f"Agent needs to load {service} service tools",
                "query": service,
                "search_type": "category",  # Search by category for service name
                "categories": [service] if service else []
            }
            
            # Query registry for service capabilities
            registry_result = await self.available_tools["reg.search"](discovery_query)
            logger.info(f"Registry search result: {registry_result}")
            
            # Load discovered tools - handle both Pydantic and dict responses
            tools_to_load = []
            if hasattr(registry_result, 'data') and hasattr(registry_result.data, 'tools'):
                tools_to_load = registry_result.data.tools
            elif isinstance(registry_result, dict) and 'data' in registry_result:
                tools_to_load = registry_result['data'].get('tools', [])
            elif isinstance(registry_result, dict) and 'tools' in registry_result:
                tools_to_load = registry_result['tools']
            
            logger.info(f"Tools to load: {tools_to_load}")
            
            for tool in tools_to_load:
                tool_name = tool.get('name') if isinstance(tool, dict) else tool.name
                service_name = tool_name.split(".")[0]
                if service_name == service:
                    logger.info(f"Loading tool: {tool_name}")
                    self._load_tool_implementation(tool_name)
            
            self.loaded_services.add(service)
            logger.info(f"Service {service} tools loaded successfully")
            
        except Exception as e:
            logger.error(f"Error discovering service {service}: {str(e)}")
    
    def _load_tool_implementation(self, tool_name: str):
        """Load actual tool implementation dynamically"""
        service, action = tool_name.split(".", 1)
        
        # Check if we have a mapping for this service
        if service in self.service_tool_mappings:
            service_tools = self.service_tool_mappings[service]
            
            if tool_name in service_tools:
                module_path = service_tools[tool_name]
                
                try:
                    # Parse module:function format
                    module_name, function_name = module_path.split(":")
                    
                    # Import the module and get the function
                    module = importlib.import_module(module_name)
                    tool_func = getattr(module, function_name)
                    
                    # Add to available tools
                    self.available_tools[tool_name] = tool_func
                    logger.info(f"Loaded tool implementation: {tool_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading tool {tool_name}: {str(e)}")
    
    async def discover_tools(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover available tools based on query"""
        if "reg.search" not in self.available_tools:
            return []
        
        try:
            discovery_query = {
                "explanation": f"Discovering tools for query: {query}",
                "query": query,
                "search_type": "capability" if not category else "category"
            }
            
            if category:
                discovery_query["categories"] = [category]
            
            result = await self.available_tools["reg.search"](discovery_query)
            
            # Extract tool information
            if hasattr(result, 'data') and hasattr(result.data, 'tools'):
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.category,
                        "capabilities": tool.capabilities
                    }
                    for tool in result.data.tools
                ]
            
        except Exception as e:
            logger.error(f"Error discovering tools: {str(e)}")
        
        return []
    
    async def get_categories(self) -> Dict[str, Any]:
        """Get available tool categories"""
        if "registry.categories" not in self.available_tools:
            return {}
        
        try:
            query = {
                "explanation": "Getting all tool categories for web frontend"
            }
            
            result = await self.available_tools["registry.categories"](query)
            
            if hasattr(result, 'data') and hasattr(result.data, 'categories'):
                return result.data.categories
                
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
        
        return {}
    
    def get_loaded_tools(self) -> List[str]:
        """Get list of currently loaded tools"""
        return list(self.available_tools.keys())
    
    def is_tool_loaded(self, tool_name: str) -> bool:
        """Check if a specific tool is loaded"""
        return tool_name in self.available_tools 