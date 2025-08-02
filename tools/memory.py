import os
import logging
import json
import httpx
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

from execute import (
    # Dual memory system models
    execute_memory_conversation_add_input, execute_memory_conversation_add_output,
    execute_memory_conversation_retrieve_input, execute_memory_conversation_retrieve_output,
    execute_memory_execution_add_input, execute_memory_execution_add_output,
    execute_memory_execution_retrieve_input, execute_memory_execution_retrieve_output,
    execute_memory_get_profile_input, execute_memory_get_profile_output
)

logger = logging.getLogger(__name__)

class MemoryError(Exception):
    """Raised when memory operations fail"""
    pass

class McpMemoryClient:
    """MCP client for connecting to remote MemoryOS server"""
    
    def __init__(self):
        self.server_url = os.getenv("MEMORYOS_SERVER_URL", "http://localhost:5000")
        self.api_key = os.getenv("MEMORYOS_API_KEY")
        self.user_id = os.getenv("MEMORY_USER_ID", "default_user")
        self.timeout = int(os.getenv("MEMORYOS_TIMEOUT", "30"))
        
        if not self.api_key:
            logger.warning("MEMORYOS_API_KEY not set - memory operations may fail")
    
    async def _make_mcp_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an MCP JSON-RPC request to the MemoryOS server"""
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.server_url}/mcp/",
                    headers=headers,
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise MemoryError(f"MCP server returned status {response.status_code}: {response.text}")
                
                result = response.json()
                
                if "error" in result:
                    raise MemoryError(f"MCP error: {result['error']}")
                
                return result.get("result", {})
                
        except httpx.TimeoutException:
            raise MemoryError(f"Timeout connecting to MemoryOS server at {self.server_url}")
        except httpx.RequestError as e:
            raise MemoryError(f"Failed to connect to MemoryOS server: {str(e)}")
        except json.JSONDecodeError as e:
            raise MemoryError(f"Invalid JSON response from MemoryOS server: {str(e)}")

# Global MCP client instance
_mcp_client: Optional[McpMemoryClient] = None

def _get_mcp_client() -> McpMemoryClient:
    """Get or create the MCP client instance"""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = McpMemoryClient()
        logger.info(f"MCP MemoryOS client initialized for server: {_mcp_client.server_url}")
    
    return _mcp_client

# =============================================================================
# DUAL MEMORY SYSTEM FUNCTIONS
# =============================================================================

async def conversation_add(input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Store a conversation pair in MemoryOS dual memory system via MCP
    
    Args:
        input_data: Dictionary with message_id, explanation, user_input, agent_response, 
                   timestamp (optional), meta_data (optional)
        
    Returns:
        Dictionary with operation result matching add_conversation_output.json schema
    """
    logger.info(f"Adding conversation memory via MCP: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = _get_mcp_client()
        
        # Validate required fields
        required_fields = ["message_id", "explanation", "user_input", "agent_response"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {
                        "status": "error",
                        "message_id": input_data.get("message_id", ""),
                        "timestamp": input_data.get("timestamp", "")
                    }
                }
        
        # Prepare MCP parameters - require explicit user_id, no fallbacks
        if not user_id:
            raise ValueError("user_id is required for memory operations - cannot store conversations without user identification")
        
        mcp_params = {
            "arguments": {
                "params": {
                    "user_id": user_id,
                    "message_id": input_data["message_id"],
                    "explanation": input_data["explanation"],
                    "user_input": input_data["user_input"],
                    "agent_response": input_data["agent_response"]
                }
            }
        }
        
        # Optional fields
        if input_data.get("timestamp"):
            mcp_params["arguments"]["params"]["timestamp"] = input_data["timestamp"]
        if input_data.get("meta_data"):
            mcp_params["arguments"]["params"]["meta_data"] = input_data["meta_data"]
        
        # Call MCP server
        result = await client._make_mcp_request("tools/call", {
            "name": "add_conversation_memory",
            **mcp_params
        })
        
        # Parse server response from MCP content format
        if result.get("content") and isinstance(result["content"][0], dict) and "text" in result["content"][0]:
            try:
                server_response = json.loads(result["content"][0]["text"])
                return server_response
            except json.JSONDecodeError:
                # Fallback if server response is malformed
                pass
        
        # Fallback response if server response parsing fails
        from datetime import datetime
        timestamp = input_data.get("timestamp") or datetime.now().isoformat()
        
        return {
            "success": True,
            "message": "Conversation memory stored via MCP MemoryOS server",
            "data": {
                "status": "success",
                "message_id": input_data["message_id"],
                "timestamp": timestamp,
                "details": {
                    "has_meta_data": bool(input_data.get("meta_data")),
                    "memory_processing": "Conversation memory added via MCP server"
                }
            }
        }
        
    except MemoryError as e:
        logger.error(f"MCP Conversation Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "message_id": input_data.get("message_id", ""),
                "timestamp": input_data.get("timestamp", "")
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in conversation_add: {e}")
        return {
            "success": False,
            "message": f"Conversation memory storage failed: {str(e)}",
            "data": {
                "status": "error",
                "message_id": input_data.get("message_id", ""),
                "timestamp": input_data.get("timestamp", "")
            }
        }

async def conversation_retrieve(input_data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """
    Retrieve conversation memories from MemoryOS dual memory system via MCP
    
    Args:
        input_data: Dictionary with explanation, query, message_id (optional), 
                   time_range (optional), max_results (optional)
        user_id: Dynamic user ID to use (overrides client default)
        
    Returns:
        Dictionary with retrieved conversations matching retrieve_conversation_output.json schema
    """
    logger.info(f"Retrieving conversation memory via MCP: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = _get_mcp_client()
        
        # Validate required fields
        required_fields = ["explanation", "query"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {
                        "status": "error",
                        "query": input_data.get("query", ""),
                        "explanation": input_data.get("explanation", ""),
                        "query_type": "general",
                        "retrieval_timestamp": "",
                        "conversations": [],
                        "total_found": 0,
                        "returned_count": 0,
                        "max_results_applied": False
                    }
                }
        
        # Prepare MCP parameters - require explicit user_id, no fallbacks
        if not user_id:
            raise ValueError("user_id is required for memory operations - cannot retrieve conversations without user identification")
        
        logger.info(f"MEMORY-CONVERSATION-RETRIEVE: Using user_id='{user_id}' (explicit)")
        
        mcp_params = {
            "arguments": {
                "params": {
                    "user_id": user_id,
                    "explanation": input_data["explanation"],
                    "query": input_data["query"],
                    "max_results": input_data.get("max_results", 10)
                }
            }
        }
        
        # Optional fields
        if input_data.get("message_id"):
            mcp_params["arguments"]["params"]["message_id"] = input_data["message_id"]
        if input_data.get("time_range"):
            mcp_params["arguments"]["params"]["time_range"] = input_data["time_range"]
        
        # Call MCP server
        result = await client._make_mcp_request("tools/call", {
            "name": "retrieve_conversation_memory",
            **mcp_params
        })
        
        # Get current timestamp
        from datetime import datetime
        retrieval_timestamp = datetime.now().isoformat()
        
        # Extract conversation data from MCP result
        conversation_data = result.get("content", [])
        if conversation_data and isinstance(conversation_data[0], dict) and "text" in conversation_data[0]:
            try:
                parsed_conversations = json.loads(conversation_data[0]["text"])
            except json.JSONDecodeError:
                parsed_conversations = {"conversations": [], "total_found": 0}
        else:
            parsed_conversations = {"conversations": [], "total_found": 0}
        
        conversations = parsed_conversations.get("conversations", [])
        total_found = parsed_conversations.get("total_found", len(conversations))
        max_results = input_data.get("max_results", 10)
        
        # Determine query type
        query_type = "general"
        if input_data.get("message_id"):
            query_type = "specific_message"
        elif input_data.get("time_range"):
            query_type = "time_filtered"
        
        return {
            "success": True,
            "message": f"Retrieved {len(conversations)} conversation(s) via MCP server",
            "data": {
                "status": "success",
                "query": input_data["query"],
                "explanation": input_data["explanation"],
                "query_type": query_type,
                "requested_message_id": input_data.get("message_id"),
                "retrieval_timestamp": retrieval_timestamp,
                "time_range": input_data.get("time_range"),
                "conversations": conversations,
                "total_found": total_found,
                "returned_count": len(conversations),
                "max_results_applied": total_found > max_results
            }
        }
        
    except MemoryError as e:
        logger.error(f"MCP Conversation Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "explanation": input_data.get("explanation", ""),
                "query_type": "general",
                "retrieval_timestamp": "",
                "conversations": [],
                "total_found": 0,
                "returned_count": 0,
                "max_results_applied": False
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in conversation_retrieve: {e}")
        return {
            "success": False,
            "message": f"Conversation memory retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "explanation": input_data.get("explanation", ""),
                "query_type": "general", 
                "retrieval_timestamp": "",
                "conversations": [],
                "total_found": 0,
                "returned_count": 0,
                "max_results_applied": False
            }
        }

async def execution_add(input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Store execution details in MemoryOS dual memory system via MCP
    
    Args:
        input_data: Dictionary with message_id, explanation, execution_summary, tools_used, 
                   errors, observations, success, duration_ms (optional), timestamp (optional)
        
    Returns:
        Dictionary with operation result matching add_execution_output.json schema
    """
    logger.info(f"Adding execution memory via MCP: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = _get_mcp_client()
        
        # Validate required fields - flat structure
        required_fields = ["message_id", "explanation", "execution_summary", "tools_used", "errors", "observations", "success"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {
                        "status": "error",
                        "message_id": input_data.get("message_id", ""),
                        "timestamp": input_data.get("timestamp", "")
                    }
                }
            # Special validation for success boolean field
            if field == "success" and input_data[field] is None:
                return {
                    "success": False,
                    "message": f"Field '{field}' cannot be None",
                    "data": {
                        "status": "error",
                        "message_id": input_data.get("message_id", ""),
                        "timestamp": input_data.get("timestamp", "")
                    }
                }
        
        # Prepare MCP parameters - require explicit user_id, no fallbacks
        if not user_id:
            raise ValueError("user_id is required for memory operations - cannot store executions without user identification")
        
        mcp_params = {
            "arguments": {
                "params": {
                    "user_id": user_id,
                    "message_id": input_data["message_id"],
                    "explanation": input_data["explanation"],
                    # Flat structure for better LLM processing
                    "execution_summary": input_data["execution_summary"],
                    "tools_used": input_data["tools_used"],
                    "errors": input_data["errors"],
                    "observations": input_data["observations"],
                    "success": input_data["success"]
                }
            }
        }
        
        # Optional fields
        if input_data.get("duration_ms") is not None:
            mcp_params["arguments"]["params"]["duration_ms"] = input_data["duration_ms"]
        if input_data.get("timestamp"):
            mcp_params["arguments"]["params"]["timestamp"] = input_data["timestamp"]
        if input_data.get("meta_data"):
            mcp_params["arguments"]["params"]["meta_data"] = input_data["meta_data"]
        
        # Call MCP server
        result = await client._make_mcp_request("tools/call", {
            "name": "add_execution_memory",
            **mcp_params
        })
        
        # Parse server response from MCP content format
        if result.get("content") and isinstance(result["content"][0], dict) and "text" in result["content"][0]:
            try:
                server_response = json.loads(result["content"][0]["text"])
                return server_response
            except json.JSONDecodeError:
                # Fallback if server response is malformed
                pass
        
        # Fallback response if server response parsing fails
        from datetime import datetime
        timestamp = input_data.get("timestamp") or datetime.now().isoformat()
        
        return {
            "success": True,
            "message": "Execution memory stored via MCP MemoryOS server",
            "data": {
                "status": "success",
                "message_id": input_data["message_id"],
                "timestamp": timestamp,
                "details": {
                    "duration_ms": input_data.get("duration_ms"),
                    "success": input_data["success"],
                    "memory_processing": "Execution memory added via MCP server"
                }
            }
        }
        
    except MemoryError as e:
        logger.error(f"MCP Execution Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "message_id": input_data.get("message_id", ""),
                "timestamp": input_data.get("timestamp", "")
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in execution_add: {e}")
        return {
            "success": False,
            "message": f"Execution memory storage failed: {str(e)}",
            "data": {
                "status": "error",
                "message_id": input_data.get("message_id", ""),
                "timestamp": input_data.get("timestamp", "")
            }
        }

async def execution_retrieve(input_data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
    """
    Retrieve execution memories from MemoryOS dual memory system via MCP
    
    Args:
        input_data: Dictionary with explanation, query, message_id (optional), 
                   max_results (optional)
        user_id: Dynamic user ID to use (overrides client default)
        
    Returns:
        Dictionary with retrieved executions matching retrieve_execution_output.json schema
    """
    logger.info(f"Retrieving execution memory via MCP: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = _get_mcp_client()
        
        # Validate required fields
        required_fields = ["explanation", "query"]
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return {
                    "success": False,
                    "message": f"Missing required field: {field}",
                    "data": {
                        "status": "error",
                        "query": input_data.get("query", ""),
                        "explanation": input_data.get("explanation", ""),
                        "query_type": "general",
                        "retrieval_timestamp": "",
                        "executions": [],
                        "total_found": 0,
                        "returned_count": 0,
                        "max_results_applied": False
                    }
                }
        
        # Prepare MCP parameters - require explicit user_id, no fallbacks
        if not user_id:
            raise ValueError("user_id is required for memory operations - cannot retrieve executions without user identification")
        
        logger.info(f"MEMORY-EXECUTION-RETRIEVE: Using user_id='{user_id}' (explicit)")
        
        mcp_params = {
            "arguments": {
                "params": {
                    "user_id": user_id,
                    "explanation": input_data["explanation"],
                    "query": input_data["query"],
                    "max_results": input_data.get("max_results", 10)
                }
            }
        }
        
        # Optional fields
        if input_data.get("message_id"):
            mcp_params["arguments"]["params"]["message_id"] = input_data["message_id"]
        
        # Call MCP server
        result = await client._make_mcp_request("tools/call", {
            "name": "retrieve_execution_memory",
            **mcp_params
        })
        
        # Get current timestamp
        from datetime import datetime
        retrieval_timestamp = datetime.now().isoformat()
        
        # Extract execution data from MCP result
        execution_data = result.get("content", [])
        if execution_data and isinstance(execution_data[0], dict) and "text" in execution_data[0]:
            try:
                parsed_executions = json.loads(execution_data[0]["text"])
            except json.JSONDecodeError:
                parsed_executions = {"executions": [], "total_found": 0}
        else:
            parsed_executions = {"executions": [], "total_found": 0}
        
        executions = parsed_executions.get("executions", [])
        total_found = parsed_executions.get("total_found", len(executions))
        max_results = input_data.get("max_results", 10)
        
        # Determine query type
        query_type = "general"
        if input_data.get("message_id"):
            query_type = "specific_message"
        else:
            query_type = "pattern_search"
        
        return {
            "success": True,
            "message": f"Retrieved {len(executions)} execution record(s) via MCP server",
            "data": {
                "status": "success",
                "query": input_data["query"],
                "explanation": input_data["explanation"],
                "query_type": query_type,
                "requested_message_id": input_data.get("message_id"),
                "retrieval_timestamp": retrieval_timestamp,
                "executions": executions,
                "total_found": total_found,
                "returned_count": len(executions),
                "max_results_applied": total_found > max_results
            }
        }
        
    except MemoryError as e:
        logger.error(f"MCP Execution Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "explanation": input_data.get("explanation", ""),
                "query_type": "general",
                "retrieval_timestamp": "",
                "executions": [],
                "total_found": 0,
                "returned_count": 0,
                "max_results_applied": False
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in execution_retrieve: {e}")
        return {
            "success": False,
            "message": f"Execution memory retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "explanation": input_data.get("explanation", ""),
                "query_type": "general",
                "retrieval_timestamp": "",
                "executions": [],
                "total_found": 0,
                "returned_count": 0,
                "max_results_applied": False
            }
        }


async def profile_retrieve(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve user profile information from MemoryOS via MCP
    
    Args:
        input_data: Dictionary with explanation, include_knowledge (optional), 
                   include_assistant_knowledge (optional)
        
    Returns:
        Dictionary with user profile data matching profile_output.json schema
    """
    logger.info(f"Retrieving user profile via MCP: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        client = _get_mcp_client()
        
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
                        "user_id": client.user_id,
                        "assistant_id": "signal",
                        "user_profile": "",
                        "user_knowledge": [],
                        "user_knowledge_count": 0,
                        "assistant_knowledge": [],
                        "assistant_knowledge_count": 0
                    }
                }
        
        # Prepare MCP parameters - only what MCP server expects
        mcp_params = {
            "arguments": {
                "params": {
                    "user_id": client.user_id,  # Required by MCP server
                    "include_knowledge": input_data.get("include_knowledge", True),
                    "include_assistant_knowledge": input_data.get("include_assistant_knowledge", False)
                }
            }
        }
        
        # Call MCP server
        result = await client._make_mcp_request("tools/call", {
            "name": "get_user_profile",
            **mcp_params
        })
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Extract profile data from MCP result
        profile_data = result.get("content", [])
        if profile_data and isinstance(profile_data[0], dict) and "text" in profile_data[0]:
            try:
                parsed_profile = json.loads(profile_data[0]["text"])
            except json.JSONDecodeError:
                parsed_profile = {
                    "user_profile": "",
                    "user_knowledge": [],
                    "assistant_knowledge": []
                }
        else:
            parsed_profile = {
                "user_profile": "",
                "user_knowledge": [],
                "assistant_knowledge": []
            }
        
        user_knowledge = parsed_profile.get("user_knowledge", [])
        assistant_knowledge = parsed_profile.get("assistant_knowledge", [])
        
        return {
            "success": True,
            "message": f"Retrieved user profile with {len(user_knowledge)} knowledge items via MCP server",
            "data": {
                "status": "success",
                "timestamp": timestamp,
                "user_id": client.user_id,
                "assistant_id": "signal",
                "user_profile": parsed_profile.get("user_profile", ""),
                "user_knowledge": user_knowledge,
                "user_knowledge_count": len(user_knowledge),
                "assistant_knowledge": assistant_knowledge,
                "assistant_knowledge_count": len(assistant_knowledge)
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
                "user_id": client.user_id,
                "assistant_id": "signal",
                "user_profile": "",
                "user_knowledge": [],
                "user_knowledge_count": 0,
                "assistant_knowledge": [],
                "assistant_knowledge_count": 0
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in profile_retrieve: {e}")
        return {
            "success": False,
            "message": f"User profile retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "timestamp": "",
                "user_id": client.user_id,
                "assistant_id": "signal",
                "user_profile": "",
                "user_knowledge": [],
                "user_knowledge_count": 0,
                "assistant_knowledge": [],
                "assistant_knowledge_count": 0
            }
        } 