import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import re

from execute import (
    execute_reg_search_input, execute_reg_search_output,
    execute_reg_describe_input, execute_reg_describe_output,
    execute_reg_list_input, execute_reg_list_output,
    execute_reg_categories_input, execute_reg_categories_output
)

logger = logging.getLogger(__name__)

class RegistryError(Exception):
    """Raised when registry operations fail"""
    pass

class ToolRegistry:
    """Tool registry that discovers tools dynamically from schema directory structure"""
    
    def __init__(self, schemas_path: str = "schemas/services"):
        self.schemas_path = Path(schemas_path)
        self._tools_cache = None
        self._categories_cache = None
        
        # Static category definitions
        self._static_categories = {
            "registry": {
                "name": "Tool Registry",
                "description": "Tools for discovering and managing other tools",
                "icon": "🔧"
            },
            "weather": {
                "name": "Weather & Climate",
                "description": "Tools for weather information and forecasts",
                "icon": "🌤️"
            },
            "memory": {
                "name": "Memory & Personalization",
                "description": "Tools for persistent memory and user profiling",
                "icon": "🧠"
            },
            "search": {
                "name": "Search & Research",
                "description": "Tools for web search and comprehensive research",
                "icon": "🔍"
            },
            "communication": {
                "name": "Communication",
                "description": "Tools for messaging and team communication",
                "icon": "💬"
            },
            "scratchpad": {
                "name": "Scratchpad & Discovery",
                "description": "Tools for progressive reduction, findings management, and evidence tracking",
                "icon": "📝"
            }
        }
    
    def _discover_tools(self) -> List[Dict[str, Any]]:
        """Discover all tools by scanning the schemas directory structure"""
        if self._tools_cache is not None:
            return self._tools_cache
            
        tools = []
        
        try:
            if not self.schemas_path.exists():
                logger.warning(f"Schemas directory not found: {self.schemas_path}")
                return []
            
            # Scan each service directory
            for service_dir in self.schemas_path.iterdir():
                if not service_dir.is_dir():
                    continue
                    
                service_name = service_dir.name
                logger.debug(f"Scanning service directory: {service_name}")
                
                # Find all input schema files
                input_files = list(service_dir.glob("*_input.json"))
                
                for input_file in input_files:
                    # Extract action name from filename (e.g., "search_input.json" -> "search")
                    action_name = input_file.stem.replace("_input", "")
                    tool_name = f"{service_name}.{action_name}"
                    
                    # Check if corresponding output file exists
                    output_file = service_dir / f"{action_name}_output.json"
                    if not output_file.exists():
                        logger.warning(f"No output schema found for {tool_name}")
                        continue
                    
                    try:
                        # Load and parse the input schema
                        with open(input_file, 'r') as f:
                            input_schema = json.load(f)
                        
                        # Extract tool information from schema
                        tool_info = self._extract_tool_info(tool_name, input_schema, service_name)
                        tools.append(tool_info)
                        
                        logger.debug(f"Discovered tool: {tool_name}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {tool_name}: {e}")
                        continue
            
            # Add registry tools (they're special and not in schemas)
            registry_tools = self._get_registry_tools()
            tools.extend(registry_tools)
            
            self._tools_cache = tools
            logger.info(f"Discovered {len(tools)} tools total")
            return tools
                
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
            return []
    
    def _extract_tool_info(self, tool_name: str, input_schema: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Extract tool information from input schema"""
        
        # Parse tool title and description from schema
        title = input_schema.get("title", tool_name.replace(".", " ").title())
        description = input_schema.get("description", "")
        
        # Convert description to array format if it's a string
        if isinstance(description, str):
            description = [description]
        elif not isinstance(description, list):
            description = [str(description)]
        
        # Determine complexity based on number of parameters
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        complexity = "basic"
        if len(properties) > 8:
            complexity = "complex"
        elif len(properties) > 4:
            complexity = "intermediate"
        
        # Extract capabilities from property descriptions
        capabilities = []
        use_cases = []
        
        for prop_name, prop_info in properties.items():
            prop_desc = prop_info.get("description", "")
            if prop_desc:
                # Extract key capabilities
                if "search" in prop_desc.lower():
                    capabilities.append("search")
                if "filter" in prop_desc.lower():
                    capabilities.append("filtering")
                if "real-time" in prop_desc.lower() or "current" in prop_desc.lower():
                    capabilities.append("real_time_data")
                if "forecast" in prop_desc.lower() or "prediction" in prop_desc.lower():
                    capabilities.append("forecasting")
        
        # Remove duplicates
        capabilities = list(set(capabilities))
        
        # Default capabilities if none found
        if not capabilities:
            if "search" in tool_name:
                capabilities = ["search", "data_retrieval"]
            elif "forecast" in tool_name:
                capabilities = ["forecasting", "weather_data"]
            else:
                capabilities = ["data_processing"]
        
        # Generate use cases based on tool name and capabilities
        action = tool_name.split(".")[-1]
        if action == "search":
            use_cases = ["Find information", "Lookup data", "Search queries"]
        elif action == "current":
            use_cases = ["Get current status", "Real-time data", "Current conditions"]
        elif action == "forecast":
            use_cases = ["Future predictions", "Planning", "Trend analysis"]
        elif action == "research":
            use_cases = ["Academic research", "Comprehensive analysis", "Investigation"]
        else:
            use_cases = [f"{action.title()} operations", "Data processing"]
        
        return {
            "name": tool_name,
            "display_name": title,
            "description": description,
            "category": service_name,
            "tags": self._generate_tags(tool_name, input_schema),
            "capabilities": capabilities,
            "use_cases": use_cases,
            "complexity": complexity,
            "implementation_type": "rest_api" if service_name not in ["registry"] else "internal",
            "input_schema": f"schemas/services/{service_name}/{action}_input.json",
            "output_schema": f"schemas/services/{service_name}/{action}_output.json"
        }
    
    def _generate_tags(self, tool_name: str, schema: Dict[str, Any]) -> List[str]:
        """Generate relevant tags for a tool"""
        tags = []
        
        # Add service name as tag
        service = tool_name.split(".")[0]
        tags.append(service)
        
        # Add action name as tag
        action = tool_name.split(".")[-1]
        tags.append(action)
        
        # Add tags based on schema properties
        properties = schema.get("properties", {})
        for prop_name, prop_info in properties.items():
            prop_desc = prop_info.get("description", "").lower()
            
            if "api" in prop_desc:
                tags.append("api")
            if "real-time" in prop_desc:
                tags.append("real-time")
            if "filter" in prop_desc:
                tags.append("filtering")
            if "search" in prop_desc:
                tags.append("search")
        
        # Add domain-specific tags
        if service == "weather":
            tags.extend(["weather", "forecast", "climate"])
        elif service == "perplexity":
            tags.extend(["ai", "research", "web"])
        elif service == "registry":
            tags.extend(["discovery", "tools", "registry"])
        
        return list(set(tags))  # Remove duplicates
    
    def _get_registry_tools(self) -> List[Dict[str, Any]]:
        """Get the built-in registry tools"""
        return [
            {
                "name": "reg.search",
                "display_name": "Search Tools",
                "description": [
                    "Search for tools in the registry using semantic queries, filters, and advanced search criteria.",
                    "Supports searching by description, capabilities, tags, use cases, and categories.",
                    "Essential for tool discovery and finding the right tool for specific tasks."
                ],
                "category": "registry",
                "tags": ["search", "discovery", "tools", "registry"],
                "capabilities": ["semantic_search", "filtering", "categorization"],
                "use_cases": ["Find weather tools", "Discover communication tools", "Search by capability"],
                "complexity": "basic",
                "implementation_type": "internal",
                "input_schema": "schemas/services/registry/search_input.json",
                "output_schema": "schemas/services/registry/search_output.json"
            },
            {
                "name": "reg.describe",
                "display_name": "Describe Tool",
                "description": [
                    "Get detailed information about a specific tool including its schema and usage examples.",
                    "Returns comprehensive tool documentation, input/output schemas, and implementation details.",
                    "Critical for understanding how to properly use discovered tools."
                ],
                "category": "registry",
                "tags": ["describe", "documentation", "schema", "details"],
                "capabilities": ["tool_details", "schema_retrieval", "documentation"],
                "use_cases": ["Get tool parameters", "Understand tool usage", "View examples"],
                "complexity": "basic",
                "implementation_type": "internal",
                "input_schema": "schemas/services/registry/describe_input.json",
                "output_schema": "schemas/services/registry/describe_output.json"
            },
            {
                "name": "reg.list",
                "display_name": "List All Tools",
                "description": [
                    "List all available tools with optional filtering and pagination.",
                    "Provides overview of the entire tool ecosystem with metadata and categories.",
                    "Useful for getting a comprehensive view of available capabilities."
                ],
                "category": "registry",
                "tags": ["list", "overview", "tools", "pagination"],
                "capabilities": ["tool_listing", "pagination", "filtering"],
                "use_cases": ["See all tools", "Browse by category", "Tool inventory"],
                "complexity": "basic",
                "implementation_type": "internal",
                "input_schema": "schemas/services/registry/list_input.json",
                "output_schema": "schemas/services/registry/list_output.json"
            },
            {
                "name": "reg.categories",
                "display_name": "List Categories",
                "description": [
                    "Get all available tool categories with their descriptions and metadata.",
                    "Provides an overview of tool organization and helps browse tools by functional area.",
                    "Essential for understanding the tool ecosystem structure and finding tools by domain."
                ],
                "category": "registry",
                "tags": ["categories", "organization", "browse", "structure"],
                "capabilities": ["category_listing", "tool_organization", "domain_browsing"],
                "use_cases": ["Browse tool categories", "Understand tool organization", "Find tools by domain"],
                "complexity": "basic",
                "implementation_type": "internal",
                "input_schema": "schemas/services/registry/categories_input.json",
                "output_schema": "schemas/services/registry/categories_output.json"
            }
        ]
    
    def search_tools(self, query: str, search_type: str = "description", 
                    categories: Optional[List[str]] = None, 
                    limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tools based on query and criteria"""
        
        tools = self._discover_tools()
        results = []
        
        # Filter by categories if specified
        if categories:
            tools = [tool for tool in tools if tool.get("category") in categories]
        
        # Search based on type
        query_lower = query.lower()
        
        # First pass: exact phrase matching
        for tool in tools:
            match_score = 0
            
            if search_type == "description":
                # Search in description and capabilities
                desc = " ".join(tool.get("description", [])).lower()
                caps = " ".join(tool.get("capabilities", [])).lower()
                tags = " ".join(tool.get("tags", [])).lower()
                if query_lower in desc or query_lower in caps or query_lower in tags:
                    match_score = 1
                    
            elif search_type == "capability":
                # Search in capabilities and use_cases
                caps = " ".join(tool.get("capabilities", [])).lower()
                use_cases = " ".join(tool.get("use_cases", [])).lower()
                if query_lower in caps or query_lower in use_cases:
                    match_score = 1
                    
            elif search_type == "name":
                # Search in name and display_name
                name = tool.get("name", "").lower()
                display_name = tool.get("display_name", "").lower()
                if query_lower in name or query_lower in display_name:
                    match_score = 1
                    
            elif search_type == "category":
                # Search by category
                if tool.get("category", "").lower() == query_lower:
                    match_score = 1
            
            if match_score > 0:
                results.append(tool)
        
        # If no results found with exact phrase, fallback to individual word matching
        if not results and len(query.split()) > 1:
            query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]  # Skip short words
            
            for tool in tools:
                match_score = 0
                
                if search_type == "description":
                    # Search in description, capabilities, and tags
                    desc = " ".join(tool.get("description", [])).lower()
                    caps = " ".join(tool.get("capabilities", [])).lower()
                    tags = " ".join(tool.get("tags", [])).lower()
                    search_text = f"{desc} {caps} {tags}"
                    
                    # Count how many words match
                    matched_words = sum(1 for word in query_words if word in search_text)
                    if matched_words > 0:
                        match_score = matched_words / len(query_words)  # Score based on word match ratio
                        
                elif search_type == "capability":
                    # Search in capabilities and use_cases
                    caps = " ".join(tool.get("capabilities", [])).lower()
                    use_cases = " ".join(tool.get("use_cases", [])).lower()
                    search_text = f"{caps} {use_cases}"
                    
                    matched_words = sum(1 for word in query_words if word in search_text)
                    if matched_words > 0:
                        match_score = matched_words / len(query_words)
                        
                elif search_type == "name":
                    # Search in name and display_name
                    name = tool.get("name", "").lower()
                    display_name = tool.get("display_name", "").lower()
                    search_text = f"{name} {display_name}"
                    
                    matched_words = sum(1 for word in query_words if word in search_text)
                    if matched_words > 0:
                        match_score = matched_words / len(query_words)
                
                if match_score > 0:
                    tool["_match_score"] = match_score
                    results.append(tool)
            
            # Sort by match score (best matches first)
            results.sort(key=lambda x: x.get("_match_score", 0), reverse=True)
            
            # Remove the temporary score field
            for tool in results:
                tool.pop("_match_score", None)
        
        # Sort by relevance and limit results
        return results[:limit]
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get specific tool by name"""
        tools = self._discover_tools()
            
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        
        return None
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools"""
        return self._discover_tools()
    
    def get_categories(self) -> Dict[str, Any]:
        """Get all tool categories with tool counts"""
        if self._categories_cache is not None:
            return self._categories_cache
            
        categories = dict(self._static_categories)
        tools = self._discover_tools()
        
        # Count tools in each category
        for category_key in categories.keys():
            tool_count = len([t for t in tools if t.get("category") == category_key])
            categories[category_key]["tool_count"] = tool_count
        
        # Add any new categories found in tools
        for tool in tools:
            tool_category = tool.get("category")
            if tool_category and tool_category not in categories:
                categories[tool_category] = {
                    "name": tool_category.title(),
                    "description": f"Tools for {tool_category} operations",
                    "icon": "🔧",
                    "tool_count": 1
                }
        
        self._categories_cache = categories
        return categories

# Global registry instance
_registry = ToolRegistry()

# Registry tool implementations
async def registry_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for tools in the registry
    
    This is the primary tool discovery mechanism when agents need to find
    tools that match specific capabilities, solve particular problems, or 
    work within certain constraints.
    """
    try:
        # Extract search parameters
        query = input_data.get("query", "")
        search_type = input_data.get("search_type", "description")
        categories = input_data.get("categories", [])
        limit = input_data.get("limit", 10)
        include_schemas = input_data.get("include_schemas", False)
        
        # Check if we would need fallback search (for logging purposes)
        query_words = query.lower().split()
        uses_fallback = len(query_words) > 1
        
        # Search for tools
        tools = _registry.search_tools(
            query=query,
            search_type=search_type,
            categories=categories,
            limit=limit
        )
        
        # Determine search strategy message
        if len(tools) == 0:
            message = f"Found 0 tools matching query '{query}'"
        elif uses_fallback and len(query_words) > 1:
            # Check if exact phrase would have returned results by doing a simple substring check
            all_tools = _registry._discover_tools()
            exact_matches = 0
            query_lower = query.lower()
            for tool in all_tools:
                desc = " ".join(tool.get("description", [])).lower()
                caps = " ".join(tool.get("capabilities", [])).lower()
                tags = " ".join(tool.get("tags", [])).lower()
                if query_lower in desc or query_lower in caps or query_lower in tags:
                    exact_matches += 1
                    break
            
            if exact_matches == 0:
                message = f"Found {len(tools)} tools using word-based search for '{query}'"
            else:
                message = f"Found {len(tools)} tools matching query '{query}'"
        else:
            message = f"Found {len(tools)} tools matching query '{query}'"
        
        # Format response
        response_data = {
            "success": True,
            "message": message,
            "data": {
                "query": query,
                "total_results": len(tools),
                "tools": tools
            }
        }
        
        # Add schemas if requested
        if include_schemas:
            for tool in response_data["data"]["tools"]:
                # Add full input/output schemas here
                tool["schemas"] = {
                    "input_schema": tool.get("input_schema", ""),
                    "output_schema": tool.get("output_schema", "")
                }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in registry_search: {e}")
        return {
            "success": False,
            "message": f"Registry search failed: {str(e)}",
            "data": {"tools": []}
        }

async def registry_describe(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific tool
    
    Use this when you need complete information about a tool including
    its capabilities, usage patterns, input/output schemas, and examples.
    """
    try:
        tool_name = input_data.get("tool_name", "")
        include_schema = input_data.get("include_schema", True)
        
        if not tool_name:
            return {
                "success": False,
                "message": "tool_name is required",
                "data": {}
            }
        
        # Get tool information
        tool = _registry.get_tool_by_name(tool_name)
        
        if not tool:
            return {
                "success": False,
                "message": f"Tool '{tool_name}' not found in registry",
                "data": {}
            }
        
        response_data = {
            "success": True,
            "message": f"Retrieved information for tool '{tool_name}'",
            "data": {
                "tool": tool
            }
        }
        
        # Add schema information if requested
        if include_schema:
            schemas = {}
            
            # Load actual schema content
            service, action = tool_name.split('.')
            input_schema_path = f"schemas/services/{service}/{action}_input.json"
            output_schema_path = f"schemas/services/{service}/{action}_output.json"
            
            try:
                # Load input schema
                if os.path.exists(input_schema_path):
                    with open(input_schema_path, 'r') as f:
                        input_schema_data = json.load(f)
                        
                        # Extract the actual schema - it's nested under "input_schema" key
                        nested_schema = input_schema_data.get("input_schema", input_schema_data)
                        
                        # Extract the actual schema properties for agent use
                        agent_schema = {
                            "type": nested_schema.get("type", "object"),
                            "properties": nested_schema.get("properties", {}),
                            "required": nested_schema.get("required", [])
                        }
                        
                        # Filter out implementation details that are handled automatically
                        implementation_fields = ["appid", "api_key", "auth_token", "authorization"]
                        if 'properties' in agent_schema:
                            for field in implementation_fields:
                                agent_schema['properties'].pop(field, None)
                        if 'required' in agent_schema:
                            agent_schema['required'] = [req for req in agent_schema['required'] 
                                                       if req not in implementation_fields]
                        
                        schemas["input_schema"] = agent_schema
                
                # Load output schema
                if os.path.exists(output_schema_path):
                    with open(output_schema_path, 'r') as f:
                        output_schema_data = json.load(f)
                        schemas["output_schema"] = output_schema_data
                        
            except Exception as e:
                logger.error(f"Error loading schemas for {tool_name}: {e}")
                schemas["error"] = f"Failed to load schemas: {str(e)}"
            
            response_data["data"]["schemas"] = schemas
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in registry_describe: {e}")
        return {
            "success": False,
            "message": f"Tool description failed: {str(e)}",
            "data": {}
        }

async def registry_list(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all available tools with optional filtering
    
    Use this to get an overview of all available tools, optionally
    filtered by category, complexity, or implementation type.
    """
    try:
        # Extract filtering parameters
        category = input_data.get("category")
        complexity = input_data.get("complexity") 
        implementation_type = input_data.get("implementation_type")
        limit = input_data.get("limit", 50)
        
        # Get all tools
        all_tools = _registry.get_all_tools()
        filtered_tools = all_tools
        
        # Apply filters
        if category:
            filtered_tools = [t for t in filtered_tools if t.get("category") == category]
        
        if complexity:
            filtered_tools = [t for t in filtered_tools if t.get("complexity") == complexity]
        
        if implementation_type:
            filtered_tools = [t for t in filtered_tools if t.get("implementation_type") == implementation_type]
        
        # Apply limit
        filtered_tools = filtered_tools[:limit]
        
        return {
            "success": True,
            "message": f"Retrieved {len(filtered_tools)} tools",
            "data": {
                "total_available": len(all_tools),
                "returned": len(filtered_tools),
                "tools": filtered_tools
            }
        }
        
    except Exception as e:
        logger.error(f"Error in registry_list: {e}")
        return {
            "success": False,
            "message": f"Tool listing failed: {str(e)}",
            "data": {"tools": []}
        }

async def registry_categories(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get all available tool categories
    
    Use this to understand the organization of tools and to present
    category-based navigation to users or for filtered discovery.
    """
    try:
        # Get categories from registry
        categories = _registry.get_categories()
        
        return {
            "success": True,
            "message": f"Retrieved {len(categories)} tool categories",
            "data": {
                "categories": categories,
                "total_categories": len(categories)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in registry_categories: {e}")
        return {
            "success": False,
            "message": f"Category retrieval failed: {str(e)}",
            "data": {"categories": {}}
        } 