import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

from execute import (
    execute_memory_add_input, execute_memory_add_output,
    execute_memory_retrieve_input, execute_memory_retrieve_output,
    execute_memory_profile_input, execute_memory_profile_output
)

logger = logging.getLogger(__name__)

class MemoryError(Exception):
    """Raised when memory operations fail"""
    pass

# Global MemoryOS instance - initialized on first use
_memoryos_instance: Optional[object] = None

def _get_memoryos_instance():
    """Get or create the MemoryOS instance"""
    global _memoryos_instance
    
    if _memoryos_instance is None:
        try:
            # Import MemoryOS here to avoid dependency issues if not installed
            from memoryos.memoryos import Memoryos
            
            # Get configuration from environment variables
            user_id = os.getenv("MEMORY_USER_ID", "default_user")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_base_url = os.getenv("OPENAI_BASE_URL")
            data_storage_path = os.getenv("MEMORY_DATA_PATH", "./memory_data")
            assistant_id = os.getenv("MEMORY_ASSISTANT_ID", "learning_tools_assistant")
            llm_model = os.getenv("MEMORY_LLM_MODEL", "gpt-4o-mini")
            embedding_model = os.getenv("MEMORY_EMBEDDING_MODEL", "BAAI/bge-m3")
            
            if not openai_api_key:
                raise MemoryError("OpenAI API key is required for MemoryOS. Set OPENAI_API_KEY environment variable.")
            
            _memoryos_instance = Memoryos(
                user_id=user_id,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                data_storage_path=data_storage_path,
                assistant_id=assistant_id,
                short_term_capacity=7,
                mid_term_capacity=1000,
                long_term_knowledge_capacity=100,
                retrieval_queue_capacity=7,
                mid_term_heat_threshold=5.0,
                mid_term_similarity_threshold=0.6,
                llm_model=llm_model,
                embedding_model_name=embedding_model
            )
            
            logger.info(f"MemoryOS initialized for user '{user_id}' with assistant '{assistant_id}'")
            
        except ImportError:
            raise MemoryError("MemoryOS package not installed. Please install the dependencies from the MemoryOS repository.")
        except Exception as e:
            raise MemoryError(f"Failed to initialize MemoryOS: {str(e)}")
    
    return _memoryos_instance

async def memory_add(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store a conversation pair in MemoryOS
    
    Args:
        input_data: Dictionary with user_input, agent_response, timestamp (optional), meta_data (optional)
        
    Returns:
        Dictionary with operation result and details
    """
    logger.info(f"Adding memory: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        memoryos = _get_memoryos_instance()
        
        user_input = input_data.get("user_input", "")
        agent_response = input_data.get("agent_response", "")
        timestamp = input_data.get("timestamp")
        meta_data = input_data.get("meta_data", {})
        
        if not user_input or not agent_response:
            return {
                "success": False,
                "message": "Both user_input and agent_response are required",
                "data": {
                    "status": "error",
                    "timestamp": timestamp or ""
                }
            }
        
        # Store the memory
        memoryos.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp,
            meta_data=meta_data
        )
        
        # Get current timestamp if not provided
        if not timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "message": "Memory successfully stored in MemoryOS",
            "data": {
                "status": "success",
                "timestamp": timestamp,
                "details": {
                    "user_input_length": len(user_input),
                    "agent_response_length": len(agent_response),
                    "has_meta_data": bool(meta_data),
                    "memory_processing": "Memory added to short-term storage"
                }
            }
        }
        
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "timestamp": input_data.get("timestamp", "")
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in memory_add: {e}")
        return {
            "success": False,
            "message": f"Memory storage failed: {str(e)}",
            "data": {
                "status": "error",
                "timestamp": input_data.get("timestamp", "")
            }
        }

async def memory_retrieve(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant memories from MemoryOS based on a query
    
    Args:
        input_data: Dictionary with query, relationship_with_user, style_hint, max_results
        
    Returns:
        Dictionary with retrieved memories from all memory types
    """
    logger.info(f"Retrieving memory: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        memoryos = _get_memoryos_instance()
        
        query = input_data.get("query", "")
        relationship_with_user = input_data.get("relationship_with_user", "friend")
        style_hint = input_data.get("style_hint", "")
        max_results = input_data.get("max_results", 10)
        
        if not query:
            return {
                "success": False,
                "message": "Query parameter is required for memory retrieval",
                "data": {
                    "status": "error",
                    "query": "",
                    "timestamp": "",
                    "user_profile": "",
                    "short_term_memory": []
                }
            }
        
        # Retrieve context using MemoryOS retriever
        retrieval_results = memoryos.retriever.retrieve_context(
            user_query=query,
            user_id=memoryos.user_id
        )
        
        # Get short-term memory
        short_term_history = memoryos.short_term_memory.get_all()
        
        # Get user profile
        user_profile = memoryos.get_user_profile_summary()
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "message": f"Retrieved {len(retrieval_results['retrieved_pages'])} relevant memories",
            "data": {
                "status": "success",
                "query": query,
                "timestamp": timestamp,
                "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile",
                "short_term_memory": short_term_history,
                "short_term_count": len(short_term_history),
                "retrieved_pages": [{
                    'user_input': page['user_input'],
                    'agent_response': page['agent_response'],
                    'timestamp': page['timestamp'],
                    'meta_info': page['meta_info']
                } for page in retrieval_results["retrieved_pages"][:max_results]],
                "retrieved_user_knowledge": [{
                    'knowledge': k['knowledge'],
                    'timestamp': k['timestamp']
                } for k in retrieval_results["retrieved_user_knowledge"][:max_results]],
                "retrieved_assistant_knowledge": [{
                    'knowledge': k['knowledge'],
                    'timestamp': k['timestamp']
                } for k in retrieval_results["retrieved_assistant_knowledge"][:max_results]],
                "total_pages_found": len(retrieval_results["retrieved_pages"]),
                "total_user_knowledge_found": len(retrieval_results["retrieved_user_knowledge"]),
                "total_assistant_knowledge_found": len(retrieval_results["retrieved_assistant_knowledge"])
            }
        }
        
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "timestamp": "",
                "user_profile": "",
                "short_term_memory": []
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in memory_retrieve: {e}")
        return {
            "success": False,
            "message": f"Memory retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "query": input_data.get("query", ""),
                "timestamp": "",
                "user_profile": "",
                "short_term_memory": []
            }
        }

async def memory_profile(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get user profile and knowledge from MemoryOS
    
    Args:
        input_data: Dictionary with include_knowledge, include_assistant_knowledge flags
        
    Returns:
        Dictionary with user profile and optional knowledge entries
    """
    logger.info(f"Getting user profile: {input_data.get('explanation', 'No explanation provided')}")
    
    try:
        memoryos = _get_memoryos_instance()
        
        include_knowledge = input_data.get("include_knowledge", True)
        include_assistant_knowledge = input_data.get("include_assistant_knowledge", False)
        
        # Get user profile
        user_profile = memoryos.get_user_profile_summary()
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = {
            "success": True,
            "message": "User profile retrieved successfully",
            "data": {
                "status": "success",
                "timestamp": timestamp,
                "user_id": memoryos.user_id,
                "assistant_id": memoryos.assistant_id,
                "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile"
            }
        }
        
        # Add user knowledge if requested
        if include_knowledge:
            user_knowledge = memoryos.user_long_term_memory.get_user_knowledge()
            result["data"]["user_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in user_knowledge
            ]
            result["data"]["user_knowledge_count"] = len(user_knowledge)
        
        # Add assistant knowledge if requested
        if include_assistant_knowledge:
            assistant_knowledge = memoryos.get_assistant_knowledge_summary()
            result["data"]["assistant_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in assistant_knowledge
            ]
            result["data"]["assistant_knowledge_count"] = len(assistant_knowledge)
        
        return result
        
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {
                "status": "error",
                "timestamp": "",
                "user_id": "",
                "assistant_id": "",
                "user_profile": ""
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in memory_profile: {e}")
        return {
            "success": False,
            "message": f"Profile retrieval failed: {str(e)}",
            "data": {
                "status": "error",
                "timestamp": "",
                "user_id": "",
                "assistant_id": "",
                "user_profile": ""
            }
        } 