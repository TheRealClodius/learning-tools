from typing import Dict, Any, Callable, List, Optional, Set
import logging
import importlib
from pathlib import Path
import time

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
    - Tool schema caching for performance
    """
    
    def __init__(self):
        # Only registry tools loaded at startup
        self.available_tools: Dict[str, Callable] = {}
        self.loaded_services: Set[str] = set()
        
        # Tool schema cache to avoid redundant registry calls
        self.tool_schema_cache: Dict[str, Dict[str, Any]] = {}
        self.search_result_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl: float = 300.0  # 5 minutes TTL for cache entries
        
        # Tool type mappings for dynamic loading
        self.service_tool_mappings = {
            "weather": {
                "weather.search": "tools.weather:weather_search",
                "weather.current": "tools.weather:weather_current", 
                "weather.forecast": "tools.weather:weather_forecast"
            },
            "perplexity": {
                "perplexity.search": "tools.perplexity:perplexity_search",
                "perplexity.research": "tools.perplexity:perplexity_research",
                "perplexity.research_stream": "tools.perplexity:perplexity_research_stream"
            },
            "memory": {
                "memory.add": "tools.memory_mcp:add_memory",
                "memory.retrieve": "tools.memory_mcp:retrieve_memory",
                "memory.get_user_profile": "tools.memory_mcp:get_user_profile"
            },
            "slack": {
                "slack.send_message": "tools.slack:slack_send_message",
                "slack.search_channels": "tools.slack:slack_search_channels",
                "slack.vector_search": "tools.slack_mcp:vector_search",
                "slack.get_channels": "tools.slack_mcp:get_channels",
                "slack.get_search_stats": "tools.slack_mcp:get_search_stats"
            }
        }
        
        # Load essential tools at startup
        self._load_registry_tools()
        self._load_memory_tools()
        self._load_slack_tools()
    
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
            from tools.memory_mcp import add_memory, retrieve_memory, get_user_profile
            
            self.available_tools.update({
                "memory.add": add_memory,
                "memory.retrieve": retrieve_memory,
                "memory.get_user_profile": get_user_profile
            })
            
            self.loaded_services.add("memory")
            logger.info("Memory tools loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not load memory tools: {e}")
    
    def _load_slack_tools(self):
        """Load Slack MCP tools that are always available"""
        try:
            from tools.slack_mcp import (
                vector_search,
                get_channels,
                get_search_stats
            )
            
            self.available_tools.update({
                "slack.vector_search": vector_search,
                "slack.get_channels": get_channels,
                "slack.get_search_stats": get_search_stats
            })
            
            self.loaded_services.add("slack")
            logger.info("Slack MCP tools loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not load Slack MCP tools: {e}")
    

    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid based on TTL"""
        if cache_key not in self.cache_timestamps:
            return False
        return (time.time() - self.cache_timestamps[cache_key]) < self.cache_ttl
    
    def _cache_tool_schema(self, tool_name: str, schema_data: Dict[str, Any]) -> None:
        """Cache tool schema data from reg.describe"""
        self.tool_schema_cache[tool_name] = schema_data
        self.cache_timestamps[f"schema_{tool_name}"] = time.time()
        logger.debug(f"Cached schema for tool: {tool_name}")
    
    def _cache_search_result(self, query: str, search_type: str, categories: List[str], result: Dict[str, Any]) -> None:
        """Cache search result from reg.search"""
        cache_key = f"{query}_{search_type}_{','.join(sorted(categories))}"
        self.search_result_cache[cache_key] = result
        self.cache_timestamps[f"search_{cache_key}"] = time.time()
        logger.debug(f"Cached search result for query: {query}")
    
    def _get_cached_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get cached tool schema if available and valid"""
        cache_key = f"schema_{tool_name}"
        if self._is_cache_valid(cache_key) and tool_name in self.tool_schema_cache:
            logger.debug(f"Using cached schema for tool: {tool_name}")
            return self.tool_schema_cache[tool_name]
        return None
    
    def _get_cached_search_result(self, query: str, search_type: str, categories: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached search result if available and valid"""
        cache_key = f"{query}_{search_type}_{','.join(sorted(categories))}"
        search_cache_key = f"search_{cache_key}"
        if self._is_cache_valid(search_cache_key) and cache_key in self.search_result_cache:
            logger.debug(f"Using cached search result for query: {query}")
            return self.search_result_cache[cache_key]
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.tool_schema_cache.clear()
        self.search_result_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Tool schema cache cleared")

    async def execute_command(self, command: str, input_data: Any, user_id: str = None, stream_callback: Optional[Callable] = None) -> Any:
        """
        Main command execution flow with dynamic discovery and streaming support.
        If a tool is not immediately available, it triggers a discovery process
        for the corresponding service and then re-attempts execution.
        """
        logger.info(f"Executing command: {command}")
        import time
        t_overall = time.time()

        # Handle registry commands with their specific caching logic first
        if command == "reg.search":
            return await self._execute_cached_search(input_data)
        elif command == "reg.describe":
            return await self._execute_cached_describe(input_data)

        # If tool is not available, try to discover and load its service
        if command not in self.available_tools:
            logger.warning(f"Tool '{command}' not loaded. Attempting discovery...")
            try:
                t_discover = time.time()
                service, _ = command.split(".", 1)
                # This will attempt to load all tools for the service.
                # Caching within discover_and_load_service prevents redundant registry calls.
                await self._discover_and_load_service(service)
                logger.info(f"TOOL-TIMING: discovery for service={service} took {int((time.time()-t_discover)*1000)} ms")
            except ValueError:
                raise ToolNotFoundError(f"Invalid command format for discovery: {command}")
            except Exception as e:
                # Log the discovery error but proceed to the final check,
                # as another concurrent task might have loaded the tool.
                logger.error(f"An error occurred during discovery for command '{command}': {e}")

        # Final attempt to execute the tool after potential discovery
        if command in self.available_tools:
            t_exec = time.time()
            result = await self._execute_tool(command, input_data, user_id, stream_callback)
            logger.info(f"TOOL-TIMING: execute {command} took {int((time.time()-t_exec)*1000)} ms (total {int((time.time()-t_overall)*1000)} ms)")
            return result
        else:
            # If the tool is still not available, raise a definitive error.
            # Check if the service was loaded to provide a more specific error message.
            service_name, _ = command.split('.', 1)
            if service_name in self.loaded_services:
                error_message = (
                    f"Command '{command}' is still not available, even though the '{service_name}' service "
                    f"was loaded. The tool may be invalid or failed to load. "
                    f"Loaded tools for this service: {[t for t in self.available_tools if t.startswith(service_name)]}"
                )
            else:
                error_message = f"Command '{command}' could not be found or loaded after discovery attempt."
            
            logger.error(error_message)
            raise ToolNotFoundError(error_message)
    
    async def _execute_tool(self, command: str, input_data: Any, user_id: str = None, stream_callback: Optional[Callable] = None) -> Any:
        """Execute a loaded tool with error handling, optional user_id, and streaming"""
        try:
            tool_func = self.available_tools[command]
            
            # Inspect the tool function to see if it accepts user_id and stream_callback
            import inspect
            sig = inspect.signature(tool_func)
            
            call_kwargs = {}
            if 'user_id' in sig.parameters:
                call_kwargs['user_id'] = user_id
            if 'stream_callback' in sig.parameters and stream_callback:
                call_kwargs['stream_callback'] = stream_callback
            
            result = await tool_func(input_data, **call_kwargs)
                
            logger.info(f"Successfully executed {command}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing {command}: {str(e)}")
            raise
    
    async def _execute_cached_search(self, input_data: Any) -> Any:
        """Execute reg.search with caching support"""
        # Extract search parameters
        query = input_data.get("query", "")
        search_type = input_data.get("search_type", "description")
        categories = input_data.get("categories", [])
        
        # Check cache first
        cached_result = self._get_cached_search_result(query, search_type, categories)
        if cached_result:
            logger.info(f"CACHE HIT: Using cached search result for query: '{query}' (type: {search_type})")
            return cached_result
        
        # Cache miss - execute actual search
        logger.info(f"CACHE MISS: Executing reg.search for query: '{query}' (type: {search_type})")
        result = await self._execute_tool("reg.search", input_data)
        
        # Cache the result
        self._cache_search_result(query, search_type, categories, result)
        
        return result
    
    async def _execute_cached_describe(self, input_data: Any) -> Any:
        """Execute reg.describe with caching support"""
        tool_name = input_data.get("tool_name", "")
        
        # Check cache first
        cached_schema = self._get_cached_tool_schema(tool_name)
        if cached_schema:
            logger.info(f"CACHE HIT: Using cached schema for tool: '{tool_name}'")
            return cached_schema
        
        # Cache miss - execute actual describe
        logger.info(f"CACHE MISS: Executing reg.describe for tool: '{tool_name}'")
        result = await self._execute_tool("reg.describe", input_data)
        
        # Cache the result
        self._cache_tool_schema(tool_name, result)
        
        return result

    async def _discover_and_load_service(self, service: str):
        """Use registry to discover and load service tools"""
        logger.info(f"Discovering tools for service: {service}")
        
        if "reg.search" not in self.available_tools:
            logger.warning("Registry not available for discovery")
            return
        
        try:
            # Search for tools by service category
            discovery_query = {
                "explanation": f"Agent needs to load {service} service tools",
                "query": service,
                "search_type": "category",  # Search by category for service name
                "categories": [service] if service else []
            }
            
            # Check cache first, then query registry
            cached_result = self._get_cached_search_result(
                service, "category", [service] if service else []
            )
            
            if cached_result:
                logger.info(f"Using cached search result for service discovery: {service}")
                registry_result = cached_result
            else:
                # Query registry for service capabilities
                registry_result = await self.available_tools["reg.search"](discovery_query)
                logger.info(f"Registry search result: {registry_result}")
                
                # Cache the search result
                self._cache_search_result(service, "category", [service] if service else [], registry_result)
            
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
            search_type = "capability" if not category else "category"
            categories = [category] if category else []
            
            # Check cache first
            cached_result = self._get_cached_search_result(query, search_type, categories)
            
            if cached_result:
                logger.info(f"Using cached discovery result for query: {query}")
                result = cached_result
            else:
                discovery_query = {
                    "explanation": f"Discovering tools for query: {query}",
                    "query": query,
                    "search_type": search_type
                }
                
                if category:
                    discovery_query["categories"] = [category]
                
                result = await self.available_tools["reg.search"](discovery_query)
                
                # Cache the result
                self._cache_search_result(query, search_type, categories, result)
            
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
    
    def get_tool_schema_from_cache(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema from cache without hitting registry"""
        return self._get_cached_tool_schema(tool_name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        current_time = time.time()
        valid_schema_count = sum(
            1 for key in self.cache_timestamps 
            if key.startswith("schema_") and (current_time - self.cache_timestamps[key]) < self.cache_ttl
        )
        valid_search_count = sum(
            1 for key in self.cache_timestamps 
            if key.startswith("search_") and (current_time - self.cache_timestamps[key]) < self.cache_ttl
        )
        
        return {
            "total_cached_schemas": len(self.tool_schema_cache),
            "total_cached_searches": len(self.search_result_cache),
            "valid_cached_schemas": valid_schema_count,
            "valid_cached_searches": valid_search_count,
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def populate_schema_cache(self, tool_name: str, schema_data: Dict[str, Any]) -> None:
        """Pre-populate the cache with a tool schema (useful for batch loading)"""
        self._cache_tool_schema(tool_name, schema_data)
        logger.info(f"Pre-populated cache with schema for: {tool_name}")
    
    def set_cache_ttl(self, ttl_seconds: float) -> None:
        """Update the cache TTL setting"""
        self.cache_ttl = ttl_seconds
        logger.info(f"Cache TTL updated to {ttl_seconds} seconds") 