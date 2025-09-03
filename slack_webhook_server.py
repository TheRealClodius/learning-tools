from fastapi import FastAPI, Request
from interfaces.slack.core_slack_orchestration import create_slack_app

# Import API interface app to mount it
from interfaces.api_interface import app as api_app

# Create a FastAPI instance
app = FastAPI(
    title="Signal AI - Slack & API Server",
    description="Combined Slack webhook server and public API",
    version="1.0.0"
)

# Get the Slack handler
slack_handler = create_slack_app()

# Mount the API interface as a sub-application
app.mount("/api", api_app)

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
        "api_endpoints": "Available at /api/*",
        "health": "/health",
        "docs": "/docs"
    }
