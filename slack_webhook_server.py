from fastapi import FastAPI, Request
from interfaces.slack.core_slack_orchestration import create_slack_app

# Create a FastAPI instance
app = FastAPI()

# Get the Slack handler
slack_handler = create_slack_app()

# This single handler can process all incoming requests from Slack
# (Events, Interactivity, Commands, etc.)
# We just need to expose the endpoints that Slack is configured to call.

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
