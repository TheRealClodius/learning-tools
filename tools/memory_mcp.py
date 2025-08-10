"""
MemoryOS MCP Client

This module provides an MCP (Model Context Protocol) client for connecting to 
the MemoryOS server to enable persistent memory capabilities for AI agents.

The client supports three main operations:
- add_memory: Store conversations in the memory system
- retrieve_memory: Query the memory system for relevant historical context
- get_user_profile: Generate user profiles from conversation history

Configuration via environment variables:
- MEMORYOS_MCP_HOST: MCP server host (default: memory-signal-production.up.railway.app)
- MEMORYOS_MCP_PORT: MCP server port (default: 443)
- MEMORYOS_MCP_PATH: MCP server path (default: /mcp)
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable
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
                 timeout: int = 30,
                 user_id: Optional[str] = None):
        """
        Initialize MemoryOS MCP client
        
        Args:
            host: MCP server hostname (defaults to env MEMORYOS_MCP_HOST or 'memory-signal-production.up.railway.app')
            port: MCP server port (defaults to env MEMORYOS_MCP_PORT or 443)
            path: MCP server path (defaults to env MEMORYOS_MCP_PATH or '/mcp')
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv('MEMORYOS_MCP_HOST', 'memory-signal-production.up.railway.app')
        self.port = port or int(os.getenv('MEMORYOS_MCP_PORT', '443'))
        self.path = path or os.getenv('MEMORYOS_MCP_PATH', '/mcp')
        self.timeout = timeout
        # Optional: bind this client session to a specific user for isolation
        self.session_user_id: Optional[str] = user_id
        # Track which user id was applied to the currently open session (if any)
        self._applied_user_id: Optional[str] = None
        
        # Use HTTPS for production Railway deployment
        protocol = "https" if self.host.endswith('.railway.app') else "http"
        if self.host.endswith('.railway.app') and self.port == 443:
            # For HTTPS on standard port, don't include port in URL
            self.server_url = f"{protocol}://{self.host}{self.path}"
        else:
            self.server_url = f"{protocol}://{self.host}:{self.port}{self.path}"
        
        # Performance optimizations
        self._retry_attempts = 2  # Retry failed operations once
        self._fast_timeout = 5.0  # Much faster timeout
        logger.info(f"MemoryOS MCP Client initialized for {self.server_url}")

        # Persistent session management
        self._connection_lock: Optional[asyncio.Lock] = asyncio.Lock()
        self._session_use_lock: Optional[asyncio.Lock] = asyncio.Lock()
        self._client_cm = None
        self._session_cm = None
        self._read = None
        self._write = None
        self._session: Optional[ClientSession] = None
        self._session_ready: bool = False
    
    async def _ensure_session(self) -> None:
        """Ensure a persistent MCP session is established and initialized."""
        # If a session exists but the desired user header changed, recreate the session
        if self._session_ready and self._session is not None:
            if self._applied_user_id != self.session_user_id:
                await self._close_session(silent=True)
            else:
                return
        async with self._connection_lock:
            if self._session_ready and self._session is not None:
                return
            # Tear down any half-open state first
            await self._close_session(silent=True)
            try:
                # Hybrid headers required for ALL HTTP requests (GET for SSE + POST for JSON-RPC)
                # MCP clients need both: text/event-stream for SSE, application/json for JSON-RPC calls
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Cache-Control": "no-cache", 
                    "Connection": "keep-alive"
                }
                # Propagate per-user header when available to help server isolate sessions
                if self.session_user_id:
                    headers["X-User-Id"] = self.session_user_id
                self._client_cm = streamablehttp_client(self.server_url, headers=headers)
                # Connection with timeout
                try:
                    self._read, self._write, _ = await asyncio.wait_for(
                        self._client_cm.__aenter__(), timeout=self._fast_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Connection timeout after {self._fast_timeout}s")
                    raise

                # Open MCP client session
                self._session_cm = ClientSession(self._read, self._write)
                try:
                    self._session = await asyncio.wait_for(
                        self._session_cm.__aenter__(), timeout=self._fast_timeout
                    )
                    await asyncio.wait_for(
                        self._session.initialize(), timeout=self._fast_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Session initialization timeout after {self._fast_timeout}s")
                    raise

                self._session_ready = True
                self._applied_user_id = self.session_user_id
            except asyncio.TimeoutError:
                logger.error(f"MCP session creation timed out after {self._fast_timeout}s")
                await self._close_session(silent=True)
                raise
            except Exception as e:
                logger.error(f"Failed to create MCP session: {e}")
                await self._close_session(silent=True)
                raise

    async def _close_session(self, silent: bool = False) -> None:
        """Close and reset the persistent session and transport safely."""
        errors = []
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                errors.append(e)
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception as e:
                errors.append(e)
        self._session_cm = None
        self._client_cm = None
        self._read = None
        self._write = None
        self._session = None
        self._session_ready = False
        if errors and not silent:
            # Log but do not raise to avoid propagating teardown noise
            logger.warning(f"Errors during MCP session close: {[type(e).__name__ for e in errors]}")

    async def close(self) -> None:
        """Public API to gracefully close the persistent MCP session."""
        async with self._connection_lock:
            await self._close_session(silent=True)

    @asynccontextmanager
    async def _get_session(self):
        """Yield a ready persistent session; serialize usage with a lock."""
        await self._ensure_session()
        async with self._session_use_lock:
            try:
                yield self._session
            except Exception as e:
                # If the active session becomes unhealthy, reset it
                logger.warning(f"MCP session error during use; resetting session: {e}")
                await self._close_session(silent=True)
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
            import time
            t0 = time.time()
            
            params = {
                "user_input": user_input,
                "agent_response": agent_response
            }
            
            # Add user_id if provided (some MCP servers support multi-user)
            if user_id:
                params["user_id"] = user_id
            
            # Add retry logic with exponential backoff
            for attempt in range(self._retry_attempts):
                try:
                    t_ensure_start = time.time()
                    async with self._get_session() as session:
                        t_ensure_ms = int((time.time() - t_ensure_start) * 1000)
                        t_call_start = time.time()
                        # Invoke tool without per-call cancellation to avoid transport teardown races
                        result = await session.call_tool("add_memory", params)
                        t_call_ms = int((time.time() - t_call_start) * 1000)
                        logger.info(f"MCP-TIMING add_memory: ensure_session={t_ensure_ms} ms, call_tool={t_call_ms} ms")
                        break  # Success, exit retry loop
                except (asyncio.TimeoutError, Exception) as e:
                    if attempt == self._retry_attempts - 1:  # Last attempt
                        logger.error(f"Memory add failed after {self._retry_attempts} attempts: {e}")
                        raise
                    else:
                        wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Memory add attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                
            t_after_call = time.time()
            
            # Parse the MCP CallToolResult to extract the actual JSON data
            if hasattr(result, 'structuredContent') and result.structuredContent:
                parse_ms = int((time.time() - t_after_call) * 1000)
                total_ms = int((time.time() - t0) * 1000)
                logger.info(f"MCP-TIMING add_memory: parse={parse_ms} ms, total={total_ms} ms")
                logger.info(f"Memory added successfully for user_id={user_id} in {total_ms} ms")
                return result.structuredContent.get('result', result.structuredContent)
            elif hasattr(result, 'content') and result.content:
                # Try to parse JSON from text content
                import json
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        try:
                            parsed = json.loads(content_item.text)
                            parse_ms = int((time.time() - t_after_call) * 1000)
                            total_ms = int((time.time() - t0) * 1000)
                            logger.info(f"MCP-TIMING add_memory: parse={parse_ms} ms, total={total_ms} ms")
                            logger.info(f"Memory added successfully for user_id={user_id} in {total_ms} ms")
                            return parsed
                        except json.JSONDecodeError:
                            pass
            
            # Fallback: return the raw result if we can't parse it
            total_ms = int((time.time() - t0) * 1000)
            logger.info(f"MCP-TIMING add_memory: total={total_ms} ms (fallback parse)")
            logger.info(f"Memory added successfully for user_id={user_id} in {total_ms} ms")
            return {"status": "success", "message": "Memory added but could not parse response format"}
            
        except Exception as e:
            error_msg = f"Failed to add memory: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
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
            import time
            start_time = time.time()
            
            params = {
                "query": query,
                "relationship_with_user": relationship_with_user,
                "style_hint": style_hint,
                "max_results": max_results
            }
            
            # Add user_id if provided
            if user_id:
                params["user_id"] = user_id
            
            # Add retry logic with exponential backoff
            for attempt in range(self._retry_attempts):
                try:
                    async with self._get_session() as session:
                        # Invoke tool without per-call cancellation to avoid transport teardown races
                        result = await session.call_tool("retrieve_memory", params)
                        break  # Success, exit retry loop
                except (asyncio.TimeoutError, Exception) as e:
                    if attempt == self._retry_attempts - 1:  # Last attempt
                        logger.error(f"Memory retrieve failed after {self._retry_attempts} attempts: {e}")
                        raise
                    else:
                        wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Memory retrieve attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Memory retrieved successfully for user_id={user_id} in {duration_ms} ms")
            
            # Parse the MCP CallToolResult to extract the actual JSON data
            parsed = None
            if hasattr(result, 'structuredContent') and result.structuredContent:
                parsed = result.structuredContent.get('result', result.structuredContent)
            elif hasattr(result, 'content') and result.content:
                # Try to parse JSON from text content
                import json
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        try:
                            parsed = json.loads(content_item.text)
                            break
                        except json.JSONDecodeError:
                            pass

            if not isinstance(parsed, dict):
                # Fallback: return error format if we can't parse it
                return {
                    "status": "error",
                    "query": query,
                    "timestamp": "",
                    "short_term_memory": [],
                    "short_term_count": 0,
                    "error": "Could not parse response format"
                }

            # Ensure a helpful message is present for streaming UIs
            try:
                stm = parsed.get("short_term_memory", []) if isinstance(parsed, dict) else []
                pages = parsed.get("retrieved_pages", []) if isinstance(parsed, dict) else []
                if "message" not in parsed:
                    parsed["message"] = f"Retrieved {len(stm)} recent and {len(pages)} historical entries"
            except Exception:
                # Do not fail if shape is unexpected
                parsed.setdefault("message", "Retrieved conversation history")

            return parsed
            
        except Exception as e:
            error_msg = f"Failed to retrieve memory: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "query": query,
                "timestamp": "",
                "short_term_memory": [],
                "short_term_count": 0,
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
            import time
            start_time = time.time()
            
            params = {
                "include_knowledge": include_knowledge,
                "include_assistant_knowledge": include_assistant_knowledge
            }
            
            # Add user_id if provided
            if user_id:
                params["user_id"] = user_id
                
            async with self._get_session() as session:
                result = await session.call_tool("get_user_profile", params)
                
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"User profile retrieved successfully for user_id={user_id} in {duration_ms} ms")
            
            # Parse the MCP CallToolResult to extract the actual JSON data
            if hasattr(result, 'structuredContent') and result.structuredContent:
                return result.structuredContent.get('result', result.structuredContent)
            elif hasattr(result, 'content') and result.content:
                # Try to parse JSON from text content
                import json
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        try:
                            return json.loads(content_item.text)
                        except json.JSONDecodeError:
                            pass
            
            # Fallback: return error format if we can't parse it
            return {
                "status": "error",
                "user_profile": "Could not parse response format",
                "error": "Response format parsing failed"
            }
            
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

def get_mcp_client(user_id: Optional[str] = None) -> MemoryMCPClient:
    """
    Get the global MCP client instance (singleton pattern)
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MemoryMCPClient(user_id=user_id)
    # If a new user_id is provided and differs, update header for future connections
    if user_id and getattr(_mcp_client, 'session_user_id', None) != user_id:
        _mcp_client.session_user_id = user_id
    return _mcp_client

async def close_mcp_client() -> None:
    """Close the global MCP client if it exists."""
    global _mcp_client
    if _mcp_client is not None:
        try:
            await _mcp_client.close()
        finally:
            _mcp_client = None

# Tool functions for integration with tool executor
async def add_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for adding memory via MCP with robust error handling
    """
    try:
        # Add overall timeout to prevent hanging (Python 3.10 compatible)
        async def _add_memory_operation():
            client = get_mcp_client(user_id=input_data.get("user_id"))
            
            user_input = input_data.get("user_input", "")
            agent_response = input_data.get("agent_response", "")
            user_id = input_data.get("user_id")
            
            if not user_input or not agent_response:
                return {
                    "status": "error",
                    "message": "Both user_input and agent_response are required"
                }
            
            return await client.add_memory(user_input, agent_response, user_id)
        
        return await asyncio.wait_for(_add_memory_operation(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("Memory add operation timed out after 30 seconds")
        return {
            "status": "error",
            "message": "Memory operation timed out",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return {
            "status": "error", 
            "message": f"Failed to add memory: {str(e)[:100]}", 
            "error_type": type(e).__name__
        }

async def retrieve_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for retrieving memory via MCP with robust error handling
    """
    try:
        # Add overall timeout to prevent hanging (Python 3.10 compatible)
        async def _retrieve_memory_operation():
            client = get_mcp_client(user_id=input_data.get("user_id"))
            
            query = input_data.get("query", "")
            user_id = input_data.get("user_id")
            relationship_with_user = input_data.get("relationship_with_user", "friend")
            style_hint = input_data.get("style_hint", "")
            max_results = input_data.get("max_results", 10)
            
            if not query:
                return {
                    "status": "error",
                    "query": "",
                    "timestamp": "",
                    "short_term_memory": [],
                    "short_term_count": 0,
                    "error": "Query parameter is required"
                }
            
            return await client.retrieve_memory(
                query=query,
                user_id=user_id,
                relationship_with_user=relationship_with_user,
                style_hint=style_hint,
                max_results=max_results
            )
        
        return await asyncio.wait_for(_retrieve_memory_operation(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("Memory retrieve operation timed out after 30 seconds")
        return {
            "status": "error",
            "query": input_data.get("query", ""),
            "timestamp": "",
            "short_term_memory": [],
            "short_term_count": 0,
            "error": "Memory operation timed out",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        logger.error(f"Failed to retrieve memory: {e}")
        return {
            "status": "error",
            "query": query if 'query' in locals() else "",
            "timestamp": "",
            "short_term_memory": [],
            "short_term_count": 0,
            "error": f"Failed to retrieve memory: {str(e)[:100]}",
            "error_type": type(e).__name__
        }



async def get_user_profile(input_data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """
    Tool function for getting user profile via MCP
    """
    client = get_mcp_client(user_id=input_data.get("user_id"))
    
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
    # Cleanly close the persistent session to avoid exit-time warnings
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_mcp_client())
