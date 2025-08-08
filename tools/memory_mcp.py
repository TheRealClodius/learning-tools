"""
MemoryOS MCP Client

This module provides an MCP (Model Context Protocol) client for connecting to 
the MemoryOS server to enable persistent memory capabilities for AI agents.

The client supports three main operations:
- add_memory: Store conversations in the memory system
- retrieve_memory: Query the memory system for relevant historical context
- get_user_profile: Generate user profiles from conversation history

Configuration via environment variables:
- MEMORYOS_MCP_HOST: MCP server host (default: localhost)
- MEMORYOS_MCP_PORT: MCP server port (default: 8000)
- MEMORYOS_MCP_PATH: MCP server path (default: /mcp)
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

class MemoryMCPClient:
    """
    MCP client for MemoryOS server operations
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None, 
                 path: str = None,
                 timeout: int = 30):
        """
        Initialize MemoryOS MCP client
        
        Args:
            host: MCP server hostname (defaults to env MEMORYOS_MCP_HOST or 'localhost')
            port: MCP server port (defaults to env MEMORYOS_MCP_PORT or 8000)
            path: MCP server path (defaults to env MEMORYOS_MCP_PATH or '/mcp')
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv('MEMORYOS_MCP_HOST', 'localhost')
        self.port = port or int(os.getenv('MEMORYOS_MCP_PORT', '8000'))
        self.path = path or os.getenv('MEMORYOS_MCP_PATH', '/mcp')
        self.timeout = timeout
        
        self.server_url = f"http://{self.host}:{self.port}{self.path}"
        logger.info(f"MemoryOS MCP Client initialized for {self.server_url}")
    
    @asynccontextmanager
    async def _get_session(self):
        """
        Get an MCP session with proper connection management
        """
        try:
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except Exception as e:
            logger.error(f"Failed to create MCP session: {e}")
            raise
    
    async def add_memory(self, user_input: str, agent_response: str, user_id: str = None) -> Dict[str, Any]:
        """
        Add a conversation to the memory system
        
        Args:
            user_input: The user's input to be stored
            agent_response: The agent's response to be stored
            user_id: Optional user identifier (for multi-user scenarios)
            
        Returns:
            Dictionary with success status and message
        """
        try:
            logger.info(f"Adding memory for user_id={user_id}")
            
            params = {
                "user_input": user_input,
                "agent_response": agent_response
            }
            
            # Add user_id if provided (some MCP servers support multi-user)
            if user_id:
                params["user_id"] = user_id
            
            async with self._get_session() as session:
                result = await session.call_tool("add_memory", params)
                
            logger.info(f"Memory added successfully for user_id={user_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to add memory: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }
    
    async def retrieve_memory(self, 
                              query: str, 
                              user_id: str = None,
                              relationship_with_user: str = "friend",
                              style_hint: str = "",
                              max_results: int = 10) -> Dict[str, Any]:
        """
        Retrieve memories based on a query
        
        Args:
            query: The query to search for in memory
            user_id: Optional user identifier
            relationship_with_user: Relationship type with the user
            style_hint: Response style hint for the retrieval
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with retrieved memory information
        """
        try:
            logger.info(f"Retrieving memory for query='{query}' user_id={user_id}")
            
            params = {
                "query": query,
                "relationship_with_user": relationship_with_user,
                "style_hint": style_hint,
                "max_results": max_results
            }
            
            # Add user_id if provided
            if user_id:
                params["user_id"] = user_id
            
            async with self._get_session() as session:
                result = await session.call_tool("retrieve_memory", params)
                
            logger.info(f"Memory retrieved successfully for user_id={user_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to retrieve memory: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "response": error_msg,
                "error": str(e)
            }
    
    async def get_user_profile(self, 
                              user_id: str = None,
                              include_knowledge: bool = True,
                              include_assistant_knowledge: bool = False) -> Dict[str, Any]:
        """
        Get user profile generated from conversation history
        
        Args:
            user_id: Optional user identifier
            include_knowledge: Include user knowledge items in the response
            include_assistant_knowledge: Include assistant knowledge items in the response
            
        Returns:
            Dictionary with user profile information
        """
        try:
            logger.info(f"Getting user profile for user_id={user_id}")
            
            params = {
                "include_knowledge": include_knowledge,
                "include_assistant_knowledge": include_assistant_knowledge
            }
            
            # Add user_id if provided
            if user_id:
                params["user_id"] = user_id
                
            async with self._get_session() as session:
                result = await session.call_tool("get_user_profile", params)
                
            logger.info(f"User profile retrieved successfully for user_id={user_id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to get user profile: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "user_profile": error_msg,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy and responsive
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with self._get_session() as session:
                # Try to list available tools as a health check
                await session.list_tools()
                logger.info("MemoryOS MCP server health check passed")
                return True
        except Exception as e:
            logger.warning(f"MemoryOS MCP server health check failed: {e}")
            return False

# Global MCP client instance
_mcp_client = None

def get_mcp_client() -> MemoryMCPClient:
    """
    Get the global MCP client instance (singleton pattern)
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MemoryMCPClient()
    return _mcp_client

# Tool functions for integration with tool executor
async def add_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for adding memory via MCP
    """
    client = get_mcp_client()
    
    user_input = input_data.get("user_input", "")
    agent_response = input_data.get("agent_response", "")
    user_id = input_data.get("user_id")
    
    if not user_input or not agent_response:
        return {
            "success": False,
            "message": "Both user_input and agent_response are required"
        }
    
    return await client.add_memory(user_input, agent_response, user_id)

async def retrieve_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for retrieving memory via MCP
    """
    client = get_mcp_client()
    
    query = input_data.get("query", "")
    user_id = input_data.get("user_id")
    relationship_with_user = input_data.get("relationship_with_user", "friend")
    style_hint = input_data.get("style_hint", "")
    max_results = input_data.get("max_results", 10)
    
    if not query:
        return {
            "success": False,
            "response": "Query parameter is required"
        }
    
    return await client.retrieve_memory(
        query=query,
        user_id=user_id,
        relationship_with_user=relationship_with_user,
        style_hint=style_hint,
        max_results=max_results
    )

async def get_user_profile(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting user profile via MCP
    """
    client = get_mcp_client()
    
    user_id = input_data.get("user_id")
    include_knowledge = input_data.get("include_knowledge", True)
    include_assistant_knowledge = input_data.get("include_assistant_knowledge", False)
    
    return await client.get_user_profile(
        user_id=user_id,
        include_knowledge=include_knowledge,
        include_assistant_knowledge=include_assistant_knowledge
    )

# Test function for development
async def test_mcp_client():
    """
    Test function to verify MCP client functionality
    """
    client = get_mcp_client()
    
    print("Testing MemoryOS MCP Client...")
    
    # Health check
    print("1. Health check...")
    healthy = await client.health_check()
    print(f"   Server healthy: {healthy}")
    
    if not healthy:
        print("   Skipping tests - server not available")
        return
    
    # Test add memory
    print("2. Testing add_memory...")
    add_result = await client.add_memory(
        user_input="Hi! I'm Tom, I work as a data scientist in SF.",
        agent_response="Hello Tom! Nice to meet you. Data science is such an exciting field."
    )
    print(f"   Result: {add_result}")
    
    # Test retrieve memory
    print("3. Testing retrieve_memory...")
    retrieve_result = await client.retrieve_memory(
        query="What do you remember about Tom?",
        relationship_with_user="assistant",
        max_results=5
    )
    print(f"   Result: {retrieve_result}")
    
    # Test recent history retrieval
    print("4. Testing recent history retrieval...")
    recent_result = await client.retrieve_memory(
        query="recent conversation history",
        max_results=3
    )
    print(f"   Recent history: {recent_result}")
    
    # Test get user profile
    print("5. Testing get_user_profile...")
    profile_result = await client.get_user_profile()
    print(f"   Result: {profile_result}")
    
    print("Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_mcp_client())
