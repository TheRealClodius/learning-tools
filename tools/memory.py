import os
import logging
import json
import time
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Load environment variables from .env.local
load_dotenv('.env.local')

from execute import (
    # Unified memory system models
    execute_memory_conversation_add_input, execute_memory_conversation_add_output,
    execute_memory_conversation_retrieve_input, execute_memory_conversation_retrieve_output,
    execute_memory_get_profile_input, execute_memory_get_profile_output
    # DEPRECATED: Removed execution models - use conversation models with execution data instead
)

logger = logging.getLogger(__name__)

class MemoryError(Exception):
    """Raised when memory operations fail"""
    pass

class McpMemoryClient:
    """MCP client for connecting to Railway-deployed MemoryOS server via SSE"""
    
    def __init__(self):
        self.server_url = os.getenv("MEMORYOS_SERVER_URL", "http://localhost:5000/sse")
        self.user_id = os.getenv("MEMORY_USER_ID", "default_user")
        self.timeout = int(os.getenv("MEMORYOS_TIMEOUT", "30"))
        
        # SSE-based MCP client components
        self.session: Optional[ClientSession] = None
        self.transport = None
        self._connected = False
        
        # Ensure URL has /sse endpoint
        if not self.server_url.endswith('/sse'):
            if self.server_url.endswith('/'):
                self.server_url += 'sse'
            else:
                self.server_url += '/sse'
        
        logger.info(f"🧠 MemoryOS Client initialized for: {self.server_url}")
    
    async def connect(self) -> bool:
        """Connect to the MemoryOS MCP server via SSE"""
        try:
            logger.info(f"🌐 Connecting to MemoryOS server: {self.server_url}")
            
            self.transport = sse_client(self.server_url)
            read_stream, write_stream = await self.transport.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            
            await self.session.initialize()
            self._connected = True
            
            logger.info("✅ Connected to MemoryOS MCP Server via SSE")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MemoryOS: {e}")
            self._connected = False
            return False
    
    async def close(self):
        """Properly close the SSE connection"""
        if self.session:
            await self.session.close()
        if self.transport:
            await self.transport.__aexit__(None, None, None)
        self._connected = False
        logger.info("🔌 Disconnected from MemoryOS")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._connected:
            logger.warning("MEMORY-CLIENT: Connection not properly closed - use await close()")
    
    def _ensure_connected(self):
        """Ensure client is connected before making requests"""
        if not self._connected or not self.session:
            raise RuntimeError("Not connected to MemoryOS. Call connect() first.")
    
    async def _make_mcp_request(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Make an MCP tool call request to the MemoryOS server with timing monitoring"""
        
        # Start timing for monitoring
        start_time = time.time()
        
        try:
            self._ensure_connected()
            
            logger.info(f"MEMORY-REQUEST: Starting {tool_name} to {self.server_url}")
            
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # Calculate and log timing
            duration = time.time() - start_time
            logger.info(f"MEMORY-TIMING: {tool_name} completed in {duration:.3f}s")
            
            # Extract response content
            if result.content and len(result.content) > 0:
                response_text = result.content[0].text
                try:
                    # Try to parse as JSON for structured responses
                    parsed_response = json.loads(response_text)
                    logger.info(f"MEMORY-SUCCESS: {tool_name} completed successfully in {duration:.3f}s")
                    return parsed_response
                except json.JSONDecodeError:
                    # Return raw text if not JSON
                    logger.info(f"MEMORY-SUCCESS: {tool_name} completed successfully in {duration:.3f}s (text response)")
                    return {"content": response_text}
            else:
                logger.warning(f"MEMORY-WARNING: {tool_name} returned empty content")
                return {}
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"MEMORY-ERROR: {tool_name} failed after {duration:.3f}s: {str(e)}")
            raise MemoryError(f"Failed to execute {tool_name}: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if the MemoryOS server is healthy"""
        try:
            # Remove '/sse' from URL and add '/health'
            health_url = self.server_url.replace('/sse', '/health')
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                is_healthy = response.status_code == 200
                
            logger.info(f"💚 Health check: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
            return is_healthy
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

# Global MCP client instance
_mcp_client: Optional[McpMemoryClient] = None

async def _get_mcp_client() -> McpMemoryClient:
    """Get or create the MCP client instance and ensure it's connected"""
    global _mcp_client
    
    if _mcp_client is None:
        # Force reload environment variables before creating client
        load_dotenv('.env.local', override=True)
        _mcp_client = McpMemoryClient()
        logger.info(f"MCP MemoryOS client initialized for server: {_mcp_client.server_url}")
        logger.info(f"MCP MemoryOS client config - User ID: {_mcp_client.user_id}")
        logger.info(f"MCP MemoryOS client config - Timeout: {_mcp_client.timeout}")
    
    # Ensure client is connected
    if not _mcp_client._connected:
        await _mcp_client.connect()
    
    return _mcp_client

async def reset_mcp_client():
    """Reset the global MCP client to force reloading of environment variables"""
    global _mcp_client
    if _mcp_client is not None:
        # Close existing client if needed
        await _mcp_client.close()
        _mcp_client = None
        logger.info("MCP client reset - will reload environment variables on next use")

# =============================================================================
# DUAL MEMORY SYSTEM FUNCTIONS
# =============================================================================

async def conversation_add(input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Add a conversation memory to MemoryOS via MCP.
    
    Args:
        input_data: Dictionary conforming to the add_conversation_input.json schema.
        user_id: User identifier for memory isolation.
        
    Returns:
        Dictionary conforming to the add_conversation_output.json schema.
    """
    logger.info(f"Adding conversation memory for user_id='{user_id}'")
    
    try:
        client = await _get_mcp_client()
        
        # Validate required fields
        required_fields = ["user_input", "agent_response"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {"status": "error", "message_id": "", "timestamp": ""}
                }

        # Format conversation content for storage
        conversation_content = f"User: {input_data['user_input']}\nAgent: {input_data['agent_response']}"
        
        # Add metadata if available
        if input_data.get("meta_data") or input_data.get("execution_details"):
            metadata_parts = []
            if input_data.get("meta_data"):
                metadata_parts.append(f"Metadata: {json.dumps(input_data['meta_data'])}")
            if input_data.get("execution_details"):
                metadata_parts.append(f"Execution: {json.dumps(input_data['execution_details'])}")
            conversation_content += f"\n{' | '.join(metadata_parts)}"

        # Prepare arguments for new MCP server
        arguments = {
            "user_id": user_id,
            "content": conversation_content,
            "importance": 0.7  # Default importance for conversations
        }

        # Call new MCP server
        logger.info(f"MEMORY-CONVERSATION-ADD: Sending MCP request with user_id='{user_id}'")
        result = await client._make_mcp_request("add_memory", arguments)
        logger.info(f"MEMORY-CONVERSATION-ADD: MCP response: {result}")
        
        # Return success response in expected format
        return {
            "success": True,
            "message": "Conversation memory stored via MCP server.",
            "data": {
                "status": "success",
                "message_id": input_data.get("message_id", ""),
                "timestamp": input_data.get("timestamp", ""),
                "details": result
            }
        }

    except Exception as e:
        logger.error(f"Unexpected error in conversation_add: {e}")
        return {
            "success": False,
            "message": f"Conversation memory storage failed: {str(e)}",
            "data": {"status": "error", "message_id": "", "timestamp": ""}
        }

async def conversation_retrieve(input_data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """
    Retrieve memories from MemoryOS via MCP.
    
    Args:
        input_data: Dictionary conforming to the retrieve_conversation_input.json schema.
        user_id: User identifier for memory isolation.
        
    Returns:
        Dictionary conforming to the retrieve_conversation_output.json schema.
    """
    logger.info(f"Retrieving conversation memory for user_id='{user_id}'")
    
    try:
        client = await _get_mcp_client()
        
        # Validate required fields
        required_fields = ["query"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "status": "error",
                    "query": input_data.get("query", ""),
                    "results": []
                }

        if not user_id:
            raise ValueError("user_id is required for memory operations")

        # Prepare arguments for new MCP server
        arguments = {
            "user_id": user_id,
            "query": input_data["query"],
            "limit": input_data.get("max_results", 10)
        }

        # Call new MCP server
        logger.info(f"MEMORY-CONVERSATION-RETRIEVE: Sending MCP request with user_id='{user_id}'")
        result = await client._make_mcp_request("retrieve_memory", arguments)
        logger.info(f"MEMORY-CONVERSATION-RETRIEVE: MCP response: {result}")
        
        # Parse server response - new server returns a list of memories
        memories = []
        if isinstance(result, list):
            memories = result
        elif isinstance(result, dict) and "content" in result:
            # Handle raw text response
            memories = [{"content": result["content"], "importance": 0.5}]
        
        # Convert to expected format
        formatted_results = []
        for memory in memories:
            if isinstance(memory, dict):
                formatted_results.append({
                    "content": memory.get("content", ""),
                    "importance": memory.get("importance", 0.5),
                    "timestamp": memory.get("timestamp", ""),
                    "metadata": memory.get("metadata", {})
                })
            else:
                # Handle string responses
                formatted_results.append({
                    "content": str(memory),
                    "importance": 0.5,
                    "timestamp": "",
                    "metadata": {}
                })
        
        return {
            "status": "success",
            "query": input_data["query"],
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Unexpected error in conversation_retrieve: {e}")
        return {
            "status": "error",
            "query": input_data.get("query", ""),
            "results": []
        }


async def get_profile(input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Retrieve user profile information from MemoryOS via MCP
    
    IMPORTANT: Profile memories are agent-generated embeddings from past interactions.
    This function performs semantic/embeddings search only - no real-time user data.
    Profile content is built by AI agents based on conversation patterns and user preferences.
    
    Args:
        input_data: Dictionary with explanation, include_knowledge (optional), 
                   include_assistant_knowledge (optional)
        user_id: Dynamic user ID to use (required for user-specific profile)
        
    Returns:
        Dictionary with agent-generated profile data via embeddings search
    """
    logger.info(f"Retrieving agent-generated profile via embeddings search: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = await _get_mcp_client()
        
        # Validate required fields
        required_fields = ["explanation"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {
                        "status": "error",
                        "timestamp": "",
                        "user_id": user_id,
                        "assistant_id": "signal",
                        "user_profile": "",
                        "user_knowledge": [],
                        "user_knowledge_count": 0,
                        "assistant_knowledge": [],
                        "assistant_knowledge_count": 0
                    }
                }
        
        # Prepare arguments - require explicit user_id, no fallbacks
        if not user_id:
            raise ValueError("user_id is required for memory operations - cannot retrieve profile without user identification")
        
        logger.info(f"MEMORY-PROFILE-RETRIEVE: Using user_id='{user_id}' (explicit)")
        
        # Prepare arguments for new MCP server
        arguments = {
            "user_id": user_id
        }
        
        # Call new MCP server
        logger.info(f"MEMORY-PROFILE-RETRIEVE: Sending MCP request with user_id='{user_id}'")
        result = await client._make_mcp_request("get_user_profile", arguments)
        logger.info(f"MEMORY-PROFILE-RETRIEVE: MCP response: {result}")
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Extract profile data from MCP result
        if isinstance(result, dict):
            profile_text = result.get("content", "")
            total_memories = result.get("total_memories", 0)
            memory_stats = result.get("memory_stats", {})
            
            # Create structured profile data
            user_knowledge = [{"content": profile_text, "source": "profile"}] if profile_text else []
            
            return {
                "success": True,
                "message": f"Retrieved agent-generated profile with {total_memories} total memories via MCP server",
                "data": {
                    "status": "success",
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "assistant_id": "signal",
                    "user_profile": profile_text,
                    "user_knowledge": user_knowledge,
                    "user_knowledge_count": len(user_knowledge),
                    "assistant_knowledge": [],
                    "assistant_knowledge_count": 0,
                    "memory_stats": memory_stats,
                    "total_memories": total_memories
                }
            }
        else:
            # Handle simple text response
            profile_text = str(result) if result else ""
            user_knowledge = [{"content": profile_text, "source": "profile"}] if profile_text else []
            
            return {
                "success": True,
                "message": f"Retrieved agent-generated profile via MCP server",
                "data": {
                    "status": "success",
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "assistant_id": "signal",
                    "user_profile": profile_text,
                    "user_knowledge": user_knowledge,
                    "user_knowledge_count": len(user_knowledge),
                    "assistant_knowledge": [],
                    "assistant_knowledge_count": 0
                }
            }
        
    except MemoryError as e:
        logger.error(f"MCP User Profile error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "timestamp": "",
                "user_id": user_id,
                "assistant_id": "signal",
                "user_profile": "",
                "user_knowledge": [],
                "user_knowledge_count": 0,
                "assistant_knowledge": [],
                "assistant_knowledge_count": 0
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in get_profile: {e}")
        return {
            "success": False,
            "message": f"User profile retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "timestamp": "",
                "user_id": user_id,
                "assistant_id": "signal",
                "user_profile": "",
                "user_knowledge": [],
                "user_knowledge_count": 0,
                "assistant_knowledge": [],
                "assistant_knowledge_count": 0
            }
        }


# =============================================================================
# CONTEXT MANAGER FOR AUTOMATIC CONNECTION HANDLING  
# =============================================================================

class MemoryOSContext:
    """Context manager for automatic MemoryOS connection handling."""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self) -> McpMemoryClient:
        self.client = await _get_mcp_client()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def health_check() -> bool:
    """
    Check if the MemoryOS server is healthy.
    
    Returns:
        bool: True if server is healthy, False otherwise
    """
    try:
        client = await _get_mcp_client()
        return await client.health_check()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


async def add_simple_memory(user_id: str, content: str, importance: float = 0.5) -> bool:
    """
    Simple function to add a memory (compatible with template usage).
    
    Args:
        user_id: User identifier
        content: Memory content
        importance: Memory importance (0.0 to 1.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        input_data = {
            "user_input": content,
            "agent_response": f"Remembered: {content[:50]}..."
        }
        result = await conversation_add(input_data, user_id)
        return result.get("success", False)
    except Exception as e:
        logger.error(f"Failed to add simple memory: {e}")
        return False


async def retrieve_simple_memory(user_id: str, query: str, limit: int = 5) -> list:
    """
    Simple function to retrieve memories (compatible with template usage).
    
    Args:
        user_id: User identifier
        query: Search query
        limit: Maximum results
        
    Returns:
        list: List of memory objects
    """
    try:
        input_data = {
            "query": query,
            "max_results": limit
        }
        result = await conversation_retrieve(input_data, user_id)
        return result.get("results", [])
    except Exception as e:
        logger.error(f"Failed to retrieve simple memory: {e}")
        return []