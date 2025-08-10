"""
Slack Chatter MCP Client

This module provides an MCP (Model Context Protocol) client for connecting to 
the Slack Chatter Service to enable semantic search of Slack messages.

The client supports three main operations:
- search_slack_messages: Search through Slack messages using semantic search
- get_slack_channels: Get list of available Slack channels
- get_search_stats: Get statistics about the indexed Slack messages

Configuration via environment variables:
- SLACK_CHATTER_MCP_HOST: MCP server host (default: slack-chatter-service.andreiclodius.repl.co)
- SLACK_CHATTER_MCP_PORT: MCP server port (default: 443)
- SLACK_CHATTER_MCP_PATH: MCP server path (default: /mcp)
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

class SlackChatterMCPClient:
    """
    MCP client for Slack Chatter Service operations
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None, 
                 path: str = None,
                 timeout: int = 30):
        """
        Initialize Slack Chatter MCP client
        
        Args:
            host: MCP server hostname (defaults to env SLACK_CHATTER_MCP_HOST or Replit URL)
            port: MCP server port (defaults to env SLACK_CHATTER_MCP_PORT or 443)
            path: MCP server path (defaults to env SLACK_CHATTER_MCP_PATH or '/mcp')
            timeout: Request timeout in seconds
        """
        # Default to the Replit deployment URL
        self.host = host or os.getenv('SLACK_CHATTER_MCP_HOST', 'slack-chatter-service.andreiclodius.repl.co')
        self.port = port or int(os.getenv('SLACK_CHATTER_MCP_PORT', '443'))
        self.path = path or os.getenv('SLACK_CHATTER_MCP_PATH', '/mcp')
        self.timeout = timeout
        
        # Use HTTPS for Replit deployment
        protocol = "https" if self.host.endswith('.repl.co') or self.port == 443 else "http"
        if self.port == 443:
            # For HTTPS on standard port, don't include port in URL
            self.server_url = f"{protocol}://{self.host}{self.path}"
        else:
            self.server_url = f"{protocol}://{self.host}:{self.port}{self.path}"
        
        # Performance optimizations
        self._retry_attempts = 2  # Retry failed operations once
        self._fast_timeout = 10.0  # Faster timeout for Slack search
        logger.info(f"Slack Chatter MCP Client initialized for {self.server_url}")

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
        if self._session_ready and self._session is not None:
            return
            
        async with self._connection_lock:
            if self._session_ready and self._session is not None:
                return
            # Tear down any half-open state first
            await self._close_session(silent=True)
            try:
                # Headers for HTTP requests
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Cache-Control": "no-cache", 
                    "Connection": "keep-alive"
                }
                
                # Create the client connection
                self._client_cm = streamablehttp_client(
                    url=self.server_url,
                    headers=headers,
                    sse_read_timeout=self._fast_timeout
                )
                # The streamablehttp_client returns a tuple of (read, write, _)
                self._read, self._write, _ = await self._client_cm.__aenter__()
                
                # Create the session
                self._session_cm = ClientSession(self._read, self._write)
                self._session = await self._session_cm.__aenter__()
                
                # Initialize the session
                init_result = await self._session.initialize()
                
                if init_result.server_info:
                    logger.info(f"Connected to Slack Chatter MCP server: {init_result.server_info.name} v{init_result.server_info.version}")
                else:
                    logger.info("Connected to Slack Chatter MCP server (no server info provided)")
                
                self._session_ready = True
                
            except Exception as e:
                logger.error(f"Failed to establish MCP session: {e}")
                await self._close_session(silent=True)
                raise
    
    async def _close_session(self, silent: bool = False) -> None:
        """Close the MCP session and clean up resources."""
        self._session_ready = False
        
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                if not silent:
                    logger.error(f"Error closing session: {e}")
            finally:
                self._session_cm = None
                self._session = None
        
        if self._client_cm:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception as e:
                if not silent:
                    logger.error(f"Error closing client: {e}")
            finally:
                self._client_cm = None
                self._read = None
                self._write = None
    
    async def search_slack_messages(self, 
                                   query: str,
                                   top_k: int = 10,
                                   channel_filter: Optional[str] = None,
                                   user_filter: Optional[str] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Search through Slack messages using semantic search
        
        Args:
            query: Search query for finding relevant messages
            top_k: Number of results to return (1-50, default 10)
            channel_filter: Filter results by channel name
            user_filter: Filter results by user name
            date_from: Filter messages from this date (YYYY-MM-DD)
            date_to: Filter messages to this date (YYYY-MM-DD)
            
        Returns:
            Dictionary with search results
        """
        try:
            async with self._session_use_lock:
                await self._ensure_session()
                
                # Build arguments
                arguments = {"query": query}
                if top_k is not None:
                    arguments["top_k"] = min(max(top_k, 1), 50)
                if channel_filter:
                    arguments["channel_filter"] = channel_filter
                if user_filter:
                    arguments["user_filter"] = user_filter
                if date_from:
                    arguments["date_from"] = date_from
                if date_to:
                    arguments["date_to"] = date_to
                
                # Call the search tool
                result = await self._session.call_tool(
                    "search_slack_messages",
                    arguments=arguments
                )
                
                # Process and return the result
                if result.content:
                    content_text = ""
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text + "\n"
                    
                    return {
                        "success": True,
                        "results": content_text,
                        "raw_content": result.content
                    }
                else:
                    return {
                        "success": False,
                        "message": "No results returned from search"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Search timeout after {self._fast_timeout}s")
            return {
                "success": False,
                "message": f"Search timed out after {self._fast_timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "success": False,
                "message": f"Search failed: {str(e)}"
            }
    
    async def get_slack_channels(self) -> Dict[str, Any]:
        """
        Get list of available Slack channels
        
        Returns:
            Dictionary with channel list
        """
        try:
            async with self._session_use_lock:
                await self._ensure_session()
                
                # Call the get channels tool
                result = await self._session.call_tool(
                    "get_slack_channels",
                    arguments={}
                )
                
                # Process and return the result
                if result.content:
                    content_text = ""
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text + "\n"
                    
                    return {
                        "success": True,
                        "channels": content_text,
                        "raw_content": result.content
                    }
                else:
                    return {
                        "success": False,
                        "message": "No channels returned"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Get channels timeout after {self._fast_timeout}s")
            return {
                "success": False,
                "message": f"Request timed out after {self._fast_timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Get channels error: {e}")
            return {
                "success": False,
                "message": f"Failed to get channels: {str(e)}"
            }
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed Slack messages
        
        Returns:
            Dictionary with search statistics
        """
        try:
            async with self._session_use_lock:
                await self._ensure_session()
                
                # Call the get stats tool
                result = await self._session.call_tool(
                    "get_search_stats",
                    arguments={}
                )
                
                # Process and return the result
                if result.content:
                    content_text = ""
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text + "\n"
                    
                    return {
                        "success": True,
                        "stats": content_text,
                        "raw_content": result.content
                    }
                else:
                    return {
                        "success": False,
                        "message": "No stats returned"
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Get stats timeout after {self._fast_timeout}s")
            return {
                "success": False,
                "message": f"Request timed out after {self._fast_timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {
                "success": False,
                "message": f"Failed to get stats: {str(e)}"
            }
    
    async def close(self):
        """Close the MCP client connection"""
        await self._close_session()


# Global client instance
_slack_chatter_client: Optional[SlackChatterMCPClient] = None

def get_slack_chatter_client() -> SlackChatterMCPClient:
    """Get or create the global Slack Chatter MCP client instance"""
    global _slack_chatter_client
    if _slack_chatter_client is None:
        _slack_chatter_client = SlackChatterMCPClient()
    return _slack_chatter_client

async def close_slack_chatter_client():
    """Close and cleanup the global Slack Chatter MCP client"""
    global _slack_chatter_client
    if _slack_chatter_client:
        await _slack_chatter_client.close()
        _slack_chatter_client = None


# Tool function wrappers for integration with the execute system
async def search_slack_messages(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for searching Slack messages via MCP
    """
    try:
        client = get_slack_chatter_client()
        
        query = input_data.get("query", "")
        if not query:
            return {
                "success": False,
                "message": "Search query is required"
            }
        
        top_k = input_data.get("top_k", 10)
        channel_filter = input_data.get("channel_filter")
        user_filter = input_data.get("user_filter")
        date_from = input_data.get("date_from")
        date_to = input_data.get("date_to")
        
        return await client.search_slack_messages(
            query=query,
            top_k=top_k,
            channel_filter=channel_filter,
            user_filter=user_filter,
            date_from=date_from,
            date_to=date_to
        )
        
    except Exception as e:
        logger.error(f"Search Slack messages error: {e}")
        return {
            "success": False,
            "message": f"Search failed: {str(e)}"
        }

async def get_slack_channels(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting Slack channels via MCP
    """
    try:
        client = get_slack_chatter_client()
        return await client.get_slack_channels()
        
    except Exception as e:
        logger.error(f"Get Slack channels error: {e}")
        return {
            "success": False,
            "message": f"Failed to get channels: {str(e)}"
        }

async def get_slack_search_stats(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting Slack search statistics via MCP
    """
    try:
        client = get_slack_chatter_client()
        return await client.get_search_stats()
        
    except Exception as e:
        logger.error(f"Get Slack search stats error: {e}")
        return {
            "success": False,
            "message": f"Failed to get stats: {str(e)}"
        }