from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

# Import agent and runtime components
from agents.research_agent import ResearchAgent
from runtime.tool_executor import ToolExecutor
from runtime.rate_limit_handler import RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Backend API",
    description="Universal backend API for AI agent interactions",
    version="1.0.0"
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite dev server
        "https://your-web-frontend.com"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent and executor
agent = ResearchAgent()
tool_executor = ToolExecutor()

# Request/Response models
class ChatMessage(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = {}
    agent_type: Optional[str] = "research"

class ChatResponse(BaseModel):
    response: str
    tool_calls: Optional[list] = []
    context: Optional[Dict[str, Any]] = {}

class ToolQuery(BaseModel):
    query: str
    category: Optional[str] = None

# Main API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """Main chat endpoint for web frontend and external clients"""
    try:
        logger.info(f"Processing chat message: {message.text[:50]}...")
        
        # Process request through agent
        response = await agent.process_request(message.text, context=message.context)
        
        return ChatResponse(
            response=response.get("message", ""),
            tool_calls=response.get("tool_calls", []),
            context=response.get("context", {})
        )
        
    except RateLimitError as e:
        logger.warning(f"Rate limit error: {str(e)}")
        raise HTTPException(status_code=429, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        # Check if it's a rate limit error that wasn't caught
        if 'rate_limit_error' in str(e).lower() or '429' in str(e):
            raise HTTPException(
                status_code=429, 
                detail="Service is experiencing high demand. Please try again in a moment."
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/chat/stream")
async def chat_stream(websocket: WebSocket):
    """Real-time streaming chat for web interface"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {message[:50]}...")
            
            # Process through agent (streaming response)
            async for chunk in agent.stream_response(message):
                await websocket.send_text(chunk)
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/api/agent/status")
async def agent_status():
    """Health check and agent status for web frontend"""
    return {
        "status": "healthy",
        "agent_type": agent.__class__.__name__,
        "available_tools": len(tool_executor.available_tools),
        "loaded_services": list(tool_executor.loaded_services)
    }

@app.post("/api/tools/discover")
async def discover_tools(query: ToolQuery):
    """Web frontend can query available tools"""
    try:
        tools = await tool_executor.discover_tools(
            query=query.query,
            category=query.category
        )
        return {"tools": tools}
        
    except Exception as e:
        logger.error(f"Error discovering tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools/categories")
async def get_tool_categories():
    """Web frontend shows available tool categories"""
    try:
        categories = await tool_executor.get_categories()
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools/list")
async def list_available_tools():
    """List all currently loaded tools"""
    return {
        "loaded_tools": list(tool_executor.available_tools.keys()),
        "loaded_services": list(tool_executor.loaded_services)
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "agent-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 