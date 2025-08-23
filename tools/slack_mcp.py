"""
Slack MCP Client

This module provides an MCP (Model Context Protocol) client for connecting to 
the Slack Service to enable comprehensive Slack operations including search,
canvas management, and discovery capabilities.

The client supports 11 main operations:

Search Operations (4):
- vector_search: Search through Slack messages using semantic search
- get_channels: Get list of available Slack channels
- search_info: Search for information using assistant.search.info
- search_context: Search for context using assistant.search.context

Canvas Operations (6):
- canvas_create: Create collaborative documents/summaries
- canvas_edit: Update canvas content with markdown/tables
- canvas_delete: Remove canvases
- canvas_access_set: Manage canvas permissions
- canvas_access_delete: Remove canvas permissions
- canvas_sections_lookup: Search within canvas sections

Statistics (1):
- get_search_stats: Get statistics about the indexed Slack messages

Configuration via environment variables:
- SLACK_MCP_HOST: MCP server host (default: slack-chronicler-andreiclodius.replit.app)
- SLACK_MCP_PORT: MCP server port (default: 443 for Replit)
- SLACK_API_KEY: Optional API key for authentication
- SLACK_WORKSPACE_DOMAIN: Enterprise workspace domain (default: uipath.enterprise.slack.com)
- SLACK_WORKSPACE_ID: Workspace ID for deep links (default: TLXCE0J2Z)
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, List
import aiohttp

from interfaces.slack.services.progressive_reducer import ProgressiveReducer
from interfaces.slack.services.scratchpad import Scratchpad

logger = logging.getLogger(__name__)

# Global scratchpad instance
_scratchpad: Optional[Scratchpad] = None

def get_scratchpad() -> Scratchpad:
    """Get or create the global scratchpad instance"""
    global _scratchpad
    if _scratchpad is None:
        _scratchpad = Scratchpad()
    return _scratchpad

class SlackMCPClient:
    """
    MCP client for Slack Service operations
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None,
                 api_key: str = None,
                 timeout: int = 30):
        self.result_reducer = ProgressiveReducer()
        """
        Initialize Slack MCP client
        
        Args:
            host: MCP server hostname (defaults to env SLACK_MCP_HOST or Replit URL)
            port: MCP server port (defaults to env SLACK_MCP_PORT or 443)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        # Default to the correct Slack Chronicler URL
        self.host = host or os.getenv('SLACK_MCP_HOST', 'slack-chronicler-andreiclodius.replit.app')
        self.port = port or int(os.getenv('SLACK_MCP_PORT', '443'))
        # Use the discovered API key as default if not provided
        self.api_key = api_key or os.getenv('SLACK_API_KEY', 'mcp_key_035e8af1ff630a5dac461a150e27c4ad0ab07bc4fb1a7bbd')
        self.timeout = timeout
        
        # Use HTTPS for Replit deployment
        protocol = "https" if (self.host.endswith('.repl.co') or self.host.endswith('.replit.app')) else "http"
        if protocol == "https" and self.port == 443:
            # For HTTPS on standard port, don't include port in URL
            self.base_url = f"{protocol}://{self.host}"
        else:
            self.base_url = f"{protocol}://{self.host}:{self.port}"
        
        logger.info(f"Slack MCP Client initialized for {self.base_url}")
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None
    
    async def _ensure_session(self) -> None:
        """Ensure an HTTP session is available."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            # Create connector with HTTP/1.1 to avoid potential HTTP/2 issues
            connector = aiohttp.TCPConnector(force_close=True)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
    
    async def _make_mcp_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the MCP server."""
        # Don't use persistent session due to HTTP/2 issues with Replit
        # await self._ensure_session()
        
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
            # Create a new session for each request to avoid HTTP/2 issues
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=json_rpc_request, headers=headers) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                        except Exception as json_error:
                            # Try to read as text if JSON parsing fails
                            text = await response.text()
                            logger.error(f"Failed to parse JSON response: {json_error}")
                            logger.error(f"Response text: {text[:500]}")
                            raise Exception(f"Invalid JSON response from server")
                        
                        # Extract session ID if provided
                        if "session_info" in result and "session_id" in result["session_info"]:
                            self._session_id = result["session_info"]["session_id"]
                        
                        # Check for JSON-RPC error
                        if "error" in result:
                            raise Exception(f"MCP error: {result['error'].get('message', 'Unknown error')}")
                        
                        return result.get("result", {})
                    elif response.status == 401:
                        raise Exception("Authentication failed - API key may be invalid or expired")
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            # Check if it's a specific error
            if "Server disconnected" in str(e):
                # This might be due to HTTP/2 issues, let's recreate the session
                await self._close_session()
                raise Exception(f"Server connection issue - please retry")
            raise Exception(f"Failed to connect to MCP server: {str(e)}")
    
    async def _close_session(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            self._session_id = None
    
    async def vector_search(self, 
                           query: str,
                           top_k: int = 10,
                           channel_filter: Optional[str] = None,
                           user_filter: Optional[str] = None,
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Search through Slack messages using semantic/vector search
        
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
    
    async def get_channels(self) -> Dict[str, Any]:
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
    
    async def search_context(self,
                          query: str,
                          thread_ts: Optional[str] = None,
                          channel_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for context using assistant.search.context
        
        Args:
            query: Search query for finding relevant context
            thread_ts: Optional thread timestamp to get context for a specific thread
            channel_id: Optional channel ID to scope the context search
            
        Returns:
            Dictionary with context search results
        """
        try:
            # Build arguments
            arguments = {"query": query}
            if thread_ts:
                arguments["thread_ts"] = thread_ts
            if channel_id:
                arguments["channel_id"] = channel_id
            
            # Make MCP request
            params = {
                "name": "assistant.search.context",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "context": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No context returned from search"
                }
                
        except Exception as e:
            logger.error(f"Context search error: {e}")
            return {
                "success": False,
                "message": f"Context search failed: {str(e)}"
            }

    async def search_info(self,
                         query: str,
                         include_pinned: bool = True,
                         include_canvases: bool = True) -> Dict[str, Any]:
        """
        Search for information using assistant.search.info
        
        Args:
            query: Search query for finding relevant information
            include_pinned: Whether to include pinned items
            include_canvases: Whether to include canvas items
            
        Returns:
            Dictionary with information search results
        """
        try:
            # Build arguments
            arguments = {
                "query": query,
                "include_pinned": include_pinned,
                "include_canvases": include_canvases
            }
            
            # Make MCP request
            params = {
                "name": "assistant.search.info",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and reduce the results
            if result and "data" in result:
                # Use structured data for processing, not the text content
                structured_data = result["data"]
                
                # Fix data type issues that cause comparison and conversion errors
                if "matches" in structured_data:
                    for match in structured_data["matches"]:
                        # Fix timestamp fields - ensure they're valid floats or convert to float
                        for ts_field in ["ts", "thread_ts"]:
                            ts_value = match.get(ts_field)
                            if ts_value == "" or ts_value is None:
                                # Use current time as fallback
                                import time
                                match[ts_field] = str(time.time())
                            elif isinstance(ts_value, str):
                                try:
                                    # Try to convert string to float to validate
                                    float(ts_value)
                                except (ValueError, TypeError):
                                    # Invalid timestamp, use current time
                                    import time
                                    match[ts_field] = str(time.time())
                            elif isinstance(ts_value, (int, float)):
                                # Convert to string for consistency
                                match[ts_field] = str(float(ts_value))
                        
                        # Ensure reply_count is a number
                        reply_count = match.get("reply_count", 0)
                        if isinstance(reply_count, str):
                            try:
                                match["reply_count"] = float(reply_count)
                            except (ValueError, TypeError):
                                match["reply_count"] = 0.0
                        elif not isinstance(reply_count, (int, float)):
                            match["reply_count"] = 0.0
                
                reduced_results = await self.result_reducer.reduce_results(
                    structured_data,
                    query
                )
                return {
                    "success": True,
                    "info": reduced_results
                }
            else:
                return {
                    "success": False,
                    "message": "No information returned from search"
                }
                
        except Exception as e:
            logger.error(f"Info search error: {e}")
            return {
                "success": False,
                "message": f"Info search failed: {str(e)}"
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

    async def canvas_create(self, 
                          name: str,
                          content: str,
                          channel_id: Optional[str] = None,
                          description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new Slack canvas
        
        Args:
            name: Name of the canvas to create
            content: Initial content for the canvas in markdown format
            channel_id: Optional channel ID to create the canvas in
            description: Optional description for the canvas
            
        Returns:
            Dictionary with canvas creation results
        """
        try:
            # Build arguments
            arguments = {
                "title": name,
                "content": content
            }
            if channel_id:
                arguments["channel_id"] = channel_id
            if description:
                arguments["description"] = description
            
            # Make MCP request
            params = {
                "name": "canvas.create",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "canvas_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No canvas data returned from creation"
                }
                
        except Exception as e:
            logger.error(f"Canvas create error: {e}")
            return {
                "success": False,
                "message": f"Canvas creation failed: {str(e)}"
            }

    async def canvas_edit(self,
                        canvas_id: str,
                        content: str,
                        operation: str = "replace",
                        section_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Edit content of an existing Slack canvas
        
        Args:
            canvas_id: ID of the canvas to edit
            content: New content for the canvas in markdown format
            operation: How to apply the content (append, replace, prepend)
            section_id: Optional specific section to edit
            
        Returns:
            Dictionary with canvas edit results
        """
        try:
            # Build arguments
            arguments = {
                "canvas_id": canvas_id,
                "content": content,
                "operation": operation
            }
            if section_id:
                arguments["section_id"] = section_id
            
            # Make MCP request
            params = {
                "name": "canvas.edit",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "canvas_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No canvas data returned from edit"
                }
                
        except Exception as e:
            logger.error(f"Canvas edit error: {e}")
            return {
                "success": False,
                "message": f"Canvas edit failed: {str(e)}"
            }

    async def canvas_delete(self, canvas_id: str) -> Dict[str, Any]:
        """
        Delete an existing Slack canvas
        
        Args:
            canvas_id: ID of the canvas to delete
            
        Returns:
            Dictionary with canvas deletion results
        """
        try:
            # Build arguments
            arguments = {"canvas_id": canvas_id}
            
            # Make MCP request
            params = {
                "name": "canvas.delete",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "deletion_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No deletion confirmation returned"
                }
                
        except Exception as e:
            logger.error(f"Canvas delete error: {e}")
            return {
                "success": False,
                "message": f"Canvas deletion failed: {str(e)}"
            }

    async def canvas_access_set(self,
                              canvas_id: str,
                              access_level: str,
                              user_ids: Optional[List[str]] = None,
                              channel_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Set access permissions for a Slack canvas
        
        Args:
            canvas_id: ID of the canvas to modify access for
            access_level: Access level to grant (read, write, admin)
            user_ids: List of user IDs to grant access to
            channel_ids: List of channel IDs to grant access to
            
        Returns:
            Dictionary with access setting results
        """
        try:
            # Build arguments
            arguments = {
                "canvas_id": canvas_id,
                "access_level": access_level
            }
            if user_ids:
                arguments["user_ids"] = user_ids
            if channel_ids:
                arguments["channel_ids"] = channel_ids
            
            # Make MCP request
            params = {
                "name": "canvas.access.manage",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "access_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No access data returned from setting"
                }
                
        except Exception as e:
            logger.error(f"Canvas access set error: {e}")
            return {
                "success": False,
                "message": f"Canvas access setting failed: {str(e)}"
            }

    async def canvas_access_delete(self,
                                 canvas_id: str,
                                 user_ids: Optional[List[str]] = None,
                                 channel_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Remove access permissions from a Slack canvas
        
        Args:
            canvas_id: ID of the canvas to remove access from
            user_ids: List of user IDs to remove access from
            channel_ids: List of channel IDs to remove access from
            
        Returns:
            Dictionary with access removal results
        """
        try:
            # Build arguments
            arguments = {"canvas_id": canvas_id}
            if user_ids:
                arguments["user_ids"] = user_ids
            if channel_ids:
                arguments["channel_ids"] = channel_ids
            
            # Make MCP request
            params = {
                "name": "canvas.access.manage",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "access_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No access data returned from removal"
                }
                
        except Exception as e:
            logger.error(f"Canvas access delete error: {e}")
            return {
                "success": False,
                "message": f"Canvas access removal failed: {str(e)}"
            }

    async def canvas_sections_lookup(self,
                                   canvas_id: str,
                                   query: str,
                                   section_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for content within sections of a Slack canvas
        
        Args:
            canvas_id: ID of the canvas to search within
            query: Search query to find content in canvas sections
            section_types: Filter by specific section types
            
        Returns:
            Dictionary with canvas section search results
        """
        try:
            # Build arguments
            arguments = {
                "canvas_id": canvas_id,
                "query": query
            }
            if section_types:
                arguments["section_types"] = section_types
            
            # Make MCP request
            params = {
                "name": "canvases.sections.lookup",
                "arguments": arguments
            }
            
            result = await self._make_mcp_request("tools/call", params)
            
            # Process and return the result
            if result and "content" in result:
                return {
                    "success": True,
                    "sections_data": result["content"],
                    "raw_content": result
                }
            else:
                return {
                    "success": False,
                    "message": "No sections data returned from search"
                }
                
        except Exception as e:
            logger.error(f"Canvas sections lookup error: {e}")
            return {
                "success": False,
                "message": f"Canvas sections lookup failed: {str(e)}"
            }
    
    async def close(self):
        """Close the MCP client connection"""
        await self._close_session()


# Global client instance
_slack_client: Optional[SlackMCPClient] = None

def get_slack_client() -> SlackMCPClient:
    """Get or create the global Slack MCP client instance"""
    global _slack_client
    if _slack_client is None:
        _slack_client = SlackMCPClient()
    return _slack_client

async def close_slack_client():
    """Close and cleanup the global Slack MCP client"""
    global _slack_client
    if _slack_client:
        await _slack_client.close()
        _slack_client = None


# Tool function wrappers for integration with the execute system
async def vector_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for vector searching Slack messages via MCP
    """
    try:
        client = get_slack_client()
        
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
        
        return await client.vector_search(
            query=query,
            top_k=top_k,
            channel_filter=channel_filter,
            user_filter=user_filter,
            date_from=date_from,
            date_to=date_to
        )
        
    except Exception as e:
        logger.error(f"Vector search Slack messages error: {e}")
        return {
            "success": False,
            "message": f"Search failed: {str(e)}"
        }

async def get_channels(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting Slack channels via MCP
    """
    try:
        client = get_slack_client()
        return await client.get_channels()
        
    except Exception as e:
        logger.error(f"Get Slack channels error: {e}")
        return {
            "success": False,
            "message": f"Failed to get channels: {str(e)}"
        }

async def get_search_stats(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting Slack search statistics via MCP
    """
    try:
        client = get_slack_client()
        return await client.get_search_stats()
        
    except Exception as e:
        logger.error(f"Get Slack search stats error: {e}")
        return {
            "success": False,
            "message": f"Failed to get stats: {str(e)}"
        }

# Scratchpad tool functions
async def scratchpad_add_finding(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for adding a finding to the scratchpad
    """
    try:
        scratchpad = get_scratchpad()
        
        finding_id = await scratchpad.add_finding(
            summary=input_data["summary"],
            evidence_ids=input_data["evidence_ids"],
            source=input_data.get("source", "slack")
        )
        
        return {
            "success": True,
            "finding_id": finding_id
        }
        
    except Exception as e:
        logger.error(f"Add finding error: {e}")
        return {
            "success": False,
            "message": f"Failed to add finding: {str(e)}"
        }

async def scratchpad_get_findings(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for retrieving findings from the scratchpad
    """
    try:
        scratchpad = get_scratchpad()
        
        findings = await scratchpad.get_context_findings(
            query=input_data.get("query"),
            source=input_data.get("source")
        )
        
        summary = await scratchpad.summarize_findings(
            max_findings=input_data.get("max_findings", 5),
            query=input_data.get("query")
        )
        
        return {
            "success": True,
            "findings": summary
        }
        
    except Exception as e:
        logger.error(f"Get findings error: {e}")
        return {
            "success": False,
            "message": f"Failed to get findings: {str(e)}"
        }

async def scratchpad_get_evidence(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for retrieving evidence from the scratchpad
    """
    try:
        scratchpad = get_scratchpad()
        
        evidence = await scratchpad.get_evidence(
            evidence_id=input_data["evidence_id"]
        )
        
        if evidence:
            return {
                "success": True,
                "evidence": {
                    "content": evidence.content,
                    "metadata": evidence.metadata,
                    "fetch_time": evidence.fetch_time
                }
            }
        else:
            return {
                "success": False,
                "message": "Evidence not found or expired"
            }
        
    except Exception as e:
        logger.error(f"Get evidence error: {e}")
        return {
            "success": False,
            "message": f"Failed to get evidence: {str(e)}"
        }

async def search_context(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for searching Slack context via assistant.search.context
    """
    try:
        client = get_slack_client()
        
        query = input_data.get("query", "")
        if not query:
            return {
                "success": False,
                "message": "Search query is required"
            }
        
        thread_ts = input_data.get("thread_ts")
        channel_id = input_data.get("channel_id")
        
        return await client.search_context(
            query=query,
            thread_ts=thread_ts,
            channel_id=channel_id
        )
        
    except Exception as e:
        logger.error(f"Search context error: {e}")
        return {
            "success": False,
            "message": f"Context search failed: {str(e)}"
        }

async def search_info(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for searching Slack info via assistant.search.info
    """
    try:
        client = get_slack_client()
        
        query = input_data.get("query", "")
        if not query:
            return {
                "success": False,
                "message": "Search query is required"
            }
        
        include_pinned = input_data.get("include_pinned", True)
        include_canvases = input_data.get("include_canvases", True)
        
        return await client.search_info(
            query=query,
            include_pinned=include_pinned,
            include_canvases=include_canvases
        )
        
    except Exception as e:
        logger.error(f"Search info error: {e}")
        return {
            "success": False,
            "message": f"Info search failed: {str(e)}"
        }

# Canvas tool function wrappers
async def canvas_create(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for creating Slack canvases via MCP
    """
    try:
        client = get_slack_client()
        
        name = input_data.get("name", "")
        if not name:
            return {
                "success": False,
                "message": "Canvas name is required"
            }
        
        content = input_data.get("content", "")
        if not content:
            return {
                "success": False,
                "message": "Canvas content is required"
            }
        
        channel_id = input_data.get("channel_id")
        description = input_data.get("description")
        
        result = await client.canvas_create(
            name=name,
            content=content,
            channel_id=channel_id,
            description=description
        )
        
        # Enhance the result with better user guidance if successful
        if result.get("success") and result.get("raw_content", {}).get("data", {}).get("canvas_id"):
            canvas_id = result["raw_content"]["data"]["canvas_id"]
            
            # Generate the proper enterprise workspace deep link
            # Format: https://{workspace_domain}/docs/{workspace_id}/{canvas_id}
            workspace_domain = os.getenv("SLACK_WORKSPACE_DOMAIN", "uipath.enterprise.slack.com")
            workspace_id = os.getenv("SLACK_WORKSPACE_ID", "TLXCE0J2Z")
            deep_link = f"https://{workspace_domain}/docs/{workspace_id}/{canvas_id}"
            
            # Provide multiple access methods for better UX
            access_instructions = {
                "canvas_created": f"✅ Canvas '{name}' created successfully!",
                "canvas_id": canvas_id,
                "access_methods": {
                    "slack_app": f"🔍 **In Slack App**: Search for '{name}' or use canvas ID: {canvas_id}",
                    "browser": result["raw_content"]["data"].get("url", deep_link),
                    "deep_link": deep_link,
                    "sidebar": "📋 **In Sidebar**: Check your 'Canvas' section for the new canvas"
                },
                "tip": "💡 **Tip**: The canvas should appear in your Slack sidebar under 'Canvas' or you can search for it by name."
            }
            
            # Merge the enhanced info with the original result
            result["user_guidance"] = access_instructions
            result["formatted_message"] = f"""Canvas '{name}' created successfully! 

**Access your canvas:**
• 🔍 Search for '{name}' in Slack
• 📋 Check the 'Canvas' section in your sidebar  
• 🔗 Direct link: {deep_link}
• 🆔 Canvas ID: {canvas_id}

The canvas should appear in your workspace momentarily."""
        
        return result
        
    except Exception as e:
        logger.error(f"Canvas create error: {e}")
        return {
            "success": False,
            "message": f"Canvas creation failed: {str(e)}"
        }

async def canvas_edit(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for editing Slack canvases via MCP
    """
    try:
        client = get_slack_client()
        
        canvas_id = input_data.get("canvas_id", "")
        if not canvas_id:
            return {
                "success": False,
                "message": "Canvas ID is required"
            }
        
        content = input_data.get("content", "")
        if not content:
            return {
                "success": False,
                "message": "Canvas content is required"
            }
        
        operation = input_data.get("operation", "replace")
        section_id = input_data.get("section_id")
        
        return await client.canvas_edit(
            canvas_id=canvas_id,
            content=content,
            operation=operation,
            section_id=section_id
        )
        
    except Exception as e:
        logger.error(f"Canvas edit error: {e}")
        return {
            "success": False,
            "message": f"Canvas edit failed: {str(e)}"
        }

async def canvas_delete(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for deleting Slack canvases via MCP
    """
    try:
        client = get_slack_client()
        
        canvas_id = input_data.get("canvas_id", "")
        if not canvas_id:
            return {
                "success": False,
                "message": "Canvas ID is required"
            }
        
        return await client.canvas_delete(canvas_id=canvas_id)
        
    except Exception as e:
        logger.error(f"Canvas delete error: {e}")
        return {
            "success": False,
            "message": f"Canvas deletion failed: {str(e)}"
        }

async def canvas_access_set(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for setting Slack canvas access permissions via MCP
    """
    try:
        client = get_slack_client()
        
        canvas_id = input_data.get("canvas_id", "")
        if not canvas_id:
            return {
                "success": False,
                "message": "Canvas ID is required"
            }
        
        access_level = input_data.get("access_level", "")
        if not access_level:
            return {
                "success": False,
                "message": "Access level is required"
            }
        
        user_ids = input_data.get("user_ids")
        channel_ids = input_data.get("channel_ids")
        
        return await client.canvas_access_set(
            canvas_id=canvas_id,
            access_level=access_level,
            user_ids=user_ids,
            channel_ids=channel_ids
        )
        
    except Exception as e:
        logger.error(f"Canvas access set error: {e}")
        return {
            "success": False,
            "message": f"Canvas access setting failed: {str(e)}"
        }

async def canvas_access_delete(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for removing Slack canvas access permissions via MCP
    """
    try:
        client = get_slack_client()
        
        canvas_id = input_data.get("canvas_id", "")
        if not canvas_id:
            return {
                "success": False,
                "message": "Canvas ID is required"
            }
        
        user_ids = input_data.get("user_ids")
        channel_ids = input_data.get("channel_ids")
        
        return await client.canvas_access_delete(
            canvas_id=canvas_id,
            user_ids=user_ids,
            channel_ids=channel_ids
        )
        
    except Exception as e:
        logger.error(f"Canvas access delete error: {e}")
        return {
            "success": False,
            "message": f"Canvas access removal failed: {str(e)}"
        }

async def canvas_sections_lookup(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for searching Slack canvas sections via MCP
    """
    try:
        client = get_slack_client()
        
        canvas_id = input_data.get("canvas_id", "")
        if not canvas_id:
            return {
                "success": False,
                "message": "Canvas ID is required"
            }
        
        query = input_data.get("query", "")
        if not query:
            return {
                "success": False,
                "message": "Search query is required"
            }
        
        section_types = input_data.get("section_types")
        
        return await client.canvas_sections_lookup(
            canvas_id=canvas_id,
            query=query,
            section_types=section_types
        )
        
    except Exception as e:
        logger.error(f"Canvas sections lookup error: {e}")
        return {
            "success": False,
            "message": f"Canvas sections lookup failed: {str(e)}"
        }