#!/usr/bin/env python3
"""
FastAPI webhook server for Slack integration
Receives Slack events via webhooks and processes them through ClientAgent
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Import our Slack interface
from interfaces.slack_interface import create_slack_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Agent Slack Webhook Server",
    description="Webhook server for Slack integration with AI Agent",
    version="1.0.0"
)

# Create Slack handler
slack_handler = create_slack_app()

@app.post("/slack/events")
async def slack_events(request: Request):
    """Handle Slack event subscriptions"""
    try:
        # Use Slack Bolt FastAPI handler
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/slack/interactive")
async def slack_interactive(request: Request):
    """Handle Slack interactive components (buttons, modals, etc.)"""
    try:
        # Use Slack Bolt FastAPI handler for interactive components
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Error handling Slack interactive event: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/slack/commands")
async def slack_commands(request: Request):
    """Handle Slack slash commands"""
    try:
        # Use Slack Bolt FastAPI handler for slash commands
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Error handling Slack command: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "slack-webhook-server",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with setup information"""
    return {
        "message": "AI Agent Slack Webhook Server",
        "endpoints": {
            "events": "/slack/events",
            "interactive": "/slack/interactive", 
            "commands": "/slack/commands",
            "health": "/health"
        },
        "setup_instructions": {
            "1": "Configure your Slack app with webhook URL: https://your-domain.com/slack/events",
            "2": "Set Interactive Components URL: https://your-domain.com/slack/interactive",
            "3": "Set Slash Commands URL: https://your-domain.com/slack/commands",
            "4": "Subscribe to bot events: app_mention, message.im",
            "5": "Required scopes: app_mentions:read, channels:history, chat:write, im:history, im:read, users:read"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env.local file")
        exit(1)
    
    logger.info("Starting Slack webhook server...")
    logger.info("Webhook endpoints will be available at:")
    logger.info("  - Events: http://localhost:8000/slack/events")
    logger.info("  - Interactive: http://localhost:8000/slack/interactive")
    logger.info("  - Commands: http://localhost:8000/slack/commands")
    
    # Run the server
    uvicorn.run(
        "slack_webhook_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )