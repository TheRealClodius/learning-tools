from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import os
import hashlib
from interfaces.slack.core_slack_orchestration import create_slack_app

# Import necessary components from API interface  
from agents.client_agent import ClientAgent
from runtime.rate_limit_handler import RateLimitHandler, RateLimitConfig, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a FastAPI instance
app = FastAPI(
    title="Signal AI - Slack & API Server",
    description="Combined Slack webhook server and public API",
    version="1.0.0"
)

# Initialize agent and rate limiter for API
agent = ClientAgent()
api_rate_config = RateLimitConfig(
    max_retries=2,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    tokens_per_minute=20000,
    burst_size=4000,
    max_queue_size=50,
    queue_timeout=120.0
)
api_rate_limiter = RateLimitHandler(api_rate_config)

# API Key handling (Railway mode - accept any key)
security = HTTPBearer(auto_error=False)
RAILWAY_MODE = True  # Always true on Railway

def hash_api_key(api_key: str) -> str:
    """Hash API key for privacy while maintaining uniqueness"""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]

async def get_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Extract API key from Authorization header"""
    if authorization and authorization.credentials:
        return authorization.credentials
    return None

# Request models
class SimpleMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}
    agent_type: Optional[str] = "research"

# Get the Slack handler
slack_handler = create_slack_app()

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    await api_rate_limiter.start()
    logger.info("🚂 Railway combined service started")

@app.on_event("shutdown")
async def shutdown_event():
    await api_rate_limiter.stop()
    logger.info("Railway combined service stopped")

# Slack webhook endpoints (keep existing functionality)
@app.post("/slack/events")
async def slack_events_endpoint(request: Request):
    """Endpoint for Slack Events API (messages, mentions, etc.)"""
    return await slack_handler.handle(request)

@app.post("/slack/interactive")
async def slack_interactive_endpoint(request: Request):
    """Endpoint for Slack Interactivity (buttons, modals, shortcuts, etc.)"""
    return await slack_handler.handle(request)

@app.post("/slack/actions")
async def slack_actions_endpoint(request: Request):
    """An alternative common endpoint for Slack Interactivity."""
    return await slack_handler.handle(request)

# API Endpoints
@app.post("/api/send-message")
async def send_message(
    message: SimpleMessage,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Main API endpoint for testing platforms - Railway mode accepts any key"""
    try:
        logger.info(f"Processing API message: {message.message[:50]}...")
        
        # Estimate tokens for rate limiting
        text_to_estimate = message.message
        if message.context:
            text_to_estimate += str(message.context)
        estimated_tokens = api_rate_limiter.estimate_tokens(text_to_estimate) * 2
        
        # Railway mode - accept any API key but track usage
        api_key_hash = hash_api_key(api_key or "default_anonymous")
        
        # Process request through agent with rate limiting
        async def rate_limited_request():
            return await agent.process_request(message.message, context=message.context)
        
        response = await api_rate_limiter.execute_with_retry(
            rate_limited_request,
            estimate_tokens=estimated_tokens
        )
        
        return {
            "status": "success",
            "response": response.get("message", ""),
            "tool_calls": response.get("tool_calls", []),
            "context": response.get("context", {}),
            "metadata": {
                "request_id": str(hash(message.message)),
                "agent_type": agent.__class__.__name__,
                "estimated_tokens": estimated_tokens,
                "api_key_hash": api_key_hash,
                "agent_type_requested": message.agent_type,
                "railway_mode": True
            }
        }
        
    except RateLimitError as e:
        logger.warning(f"Rate limit error: {str(e)}")
        return {
            "status": "rate_limited",
            "error": str(e),
            "retry_after": "60",
            "type": "rate_limit_error"
        }
        
    except Exception as e:
        logger.error(f"Error processing API message: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "estimated_tokens": locals().get('estimated_tokens', 0)
        }

@app.get("/api/agent/status")
async def agent_status():
    """API agent status and health check"""
    bucket = api_rate_limiter.token_bucket
    return {
        "status": "healthy",
        "agent_type": agent.__class__.__name__,
        "railway_mode": True,
        "rate_limiting": {
            "available_tokens": int(bucket.tokens),
            "tokens_per_minute": bucket.tokens_per_minute,
            "burst_size": bucket.burst_size,
            "queue_size": len(api_rate_limiter.request_queue)
        }
    }

# Add a health check for the combined service
@app.get("/health")
async def health_check():
    """Health check for the combined Slack + API service"""
    return {"status": "healthy", "services": ["slack", "api"]}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Signal AI Combined Server",
        "slack_endpoints": ["/slack/events", "/slack/interactive", "/slack/actions"],
        "api_endpoints": ["/api/send-message", "/api/agent/status"],
        "health": "/health",
        "docs": "/docs"
    }
