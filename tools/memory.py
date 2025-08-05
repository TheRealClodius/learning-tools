
import os
from memoryos import Memoryos
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
ASSISTANT_ID = "signal_assistant"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DATA_STORAGE_PATH = "./memory_data"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def get_memo_instance(user_id: str):
    """Initializes and returns a Memoryos instance for a given user."""
    if not user_id:
        raise ValueError("user_id is required to initialize MemoryOS.")
    
    # Per-user data path
    user_data_path = os.path.join(DATA_STORAGE_PATH, user_id)
    
    try:
        memo = Memoryos(
            user_id=user_id,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=user_data_path,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            embedding_model_name=EMBEDDING_MODEL_NAME,
        )
        return memo
    except Exception as e:
        logger.error(f"Error initializing MemoryOS for user {user_id}: {e}")
        raise

async def add_memory(input_data: dict, user_id: str):
    """Adds a new memory to the user's memory store."""
    try:
        memo = get_memo_instance(user_id)
        memo.add_memory(
            user_input=input_data.get("user_input"),
            agent_response=input_data.get("agent_response")
        )
        return {"success": True, "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding memory for user {user_id}: {e}")
        return {"success": False, "error": str(e)}

async def query_memory(input_data: dict, user_id: str):
    """Queries the user's memory store and returns a response."""
    try:
        memo = get_memo_instance(user_id)
        response = memo.get_response(
            query=input_data.get("query"),
        )
        return {"success": True, "response": response}
    except Exception as e:
        logger.error(f"Error querying memory for user {user_id}: {e}")
        return {"success": False, "error": str(e)}
