
import os
import logging

logger = logging.getLogger(__name__)

# Try to import MemoryOS, but handle gracefully if not available
try:
    from memos.mem_os.main import MOS
    from memos.configs.mem_os import MOSConfig
    MEMORYOS_AVAILABLE = True
except ImportError:
    logger.warning("MemoryOS package not available. Memory functions will be disabled.")
    MOS = None
    MOSConfig = None
    MEMORYOS_AVAILABLE = False

# --- Configuration ---
ASSISTANT_ID = "signal_assistant"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DATA_STORAGE_PATH = "./memory_data"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def get_memo_instance(user_id: str):
    """Initializes and returns a MemoryOS instance for a given user."""
    if not MEMORYOS_AVAILABLE:
        raise RuntimeError("MemoryOS package is not available. Please install MemoryOS to use memory functions.")
    
    if not user_id:
        raise ValueError("user_id is required to initialize MemoryOS.")
    
    # Per-user data path
    user_data_path = os.path.join(DATA_STORAGE_PATH, user_id)
    
    try:
        # Create MemoryOS configuration
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": API_KEY,
                    "model": LLM_MODEL
                }
            },
            "history_db_path": os.path.join(user_data_path, "history.db")
        }
        
        # Initialize MemoryOS
        mos_config = MOSConfig.from_dict(config)
        memory = MOS(mos_config)
        
        # Create user if doesn't exist
        memory.create_user(user_id=user_id)
        
        return memory
    except Exception as e:
        logger.error(f"Error initializing MemoryOS for user {user_id}: {e}")
        raise

async def add_memory(input_data: dict, user_id: str):
    """Adds a new memory to the user's memory store."""
    if not MEMORYOS_AVAILABLE:
        return {"success": False, "error": "Memory functionality is not available. MemoryOS package is not installed."}
    
    try:
        memory = get_memo_instance(user_id)
        
        # Format messages for MemoryOS
        messages = []
        if input_data.get("user_input"):
            messages.append({"role": "user", "content": input_data.get("user_input")})
        if input_data.get("agent_response"):
            messages.append({"role": "assistant", "content": input_data.get("agent_response")})
        
        # Add memory using MemoryOS API
        memory.add(messages=messages, user_id=user_id)
        
        return {"success": True, "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding memory for user {user_id}: {e}")
        return {"success": False, "error": str(e)}

async def query_memory(input_data: dict, user_id: str):
    """Queries the user's memory store and returns a response."""
    if not MEMORYOS_AVAILABLE:
        return {"success": False, "error": "Memory functionality is not available. MemoryOS package is not installed."}
    
    try:
        memo = get_memo_instance(user_id)
        response = memo.get_response(
            query=input_data.get("query"),
        )
        return {"success": True, "response": response}
    except Exception as e:
        logger.error(f"Error querying memory for user {user_id}: {e}")
        return {"success": False, "error": str(e)}
