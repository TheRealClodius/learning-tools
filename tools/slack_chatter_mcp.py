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
- SLACK_CHATTER_MCP_PORT: MCP server port (default: 5000 for Replit)
- SLACK_CHATTER_API_KEY: Optional API key for authentication
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)

class SlackChatterMCPClient:
    """
    MCP client for Slack Chatter Service operations
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None,
                 api_key: str = None,
                 timeout: int = 30):
        """
        Initialize Slack Chatter MCP client
        
        Args:
            host: MCP server hostname (defaults to env SLACK_CHATTER_MCP_HOST or Replit URL)
            port: MCP server port (defaults to env SLACK_CHATTER_MCP_PORT or 5000)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        # Default to the Replit deployment URL
        self.host = host or os.getenv('SLACK_CHATTER_MCP_HOST', 'slack-chatter-service.andreiclodius.repl.co')
        self.port = port or int(os.getenv('SLACK_CHATTER_MCP_PORT', '5000'))
        self.api_key = api_key or os.getenv('SLACK_CHATTER_API_KEY')
        self.timeout = timeout
        
        # Use HTTPS for Replit deployment
        protocol = "https" if self.host.endswith('.repl.co') else "http"
        if protocol == "https" and self.port == 443:
            # For HTTPS on standard port, don't include port in URL
            self.base_url = f"{protocol}://{self.host}"
        else:
            self.base_url = f"{protocol}://{self.host}:{self.port}"
        
        logger.info(f"Slack Chatter MCP Client initialized for {self.base_url}")
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None
    
    async def _ensure_session(self) -> None:
        """Ensure an HTTP session is available."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def _make_mcp_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the MCP server."""
        await self._ensure_session()
        
        # Prepare JSON-RPC request
        request_id = 1
        json_rpc_request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authentication if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add session ID if available
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        
        # Make the request
        url = f"{self.base_url}/mcp"
        
        try:
            async with self._session.post(url, json=json_rpc_request, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract session ID if provided
                    if "session_info" in result and "session_id" in result["session_info"]:
                        self._session_id = result["session_info"]["session_id"]
                    
                    # Check for JSON-RPC error
                    if "error" in result:
                        raise Exception(f"MCP error: {result['error'].get('message', 'Unknown error')}")
                    
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            raise Exception(f"Failed to connect to MCP server: {str(e)}")
    
    async def _close_session(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            self._session_id = None
    
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
            
            # Make MCP request
            params = {
                "name": "search_slack_messages",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                content_text = ""
                for content_item in result["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        content_text += content_item["text"] + "\n"
                
                return {
                    "success": True,
                    "results": content_text,
                    "raw_content": result["content"]
                }
            else:
                return {
                    "success": False,
                    "message": "No results returned from search"
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
            # Make MCP request
            params = {
                "name": "get_slack_channels",
                "arguments": {}
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                content_text = ""
                for content_item in result["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        content_text += content_item["text"] + "\n"
                
                return {
                    "success": True,
                    "channels": content_text,
                    "raw_content": result["content"]
                }
            else:
                return {
                    "success": False,
                    "message": "No channels returned"
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
            # Make MCP request
            params = {
                "name": "get_search_stats",
                "arguments": {}
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                content_text = ""
                for content_item in result["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        content_text += content_item["text"] + "\n"
                
                return {
                    "success": True,
                    "stats": content_text,
                    "raw_content": result["content"]
                }
            else:
                return {
                    "success": False,
                    "message": "No stats returned"
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