from fastapi import FastAPI, Request
from interfaces.slack_interface import create_slack_app

# Create a FastAPI instance
app = FastAPI()

# Get the Slack handler
slack_handler = create_slack_app()

# Create a route for Slack events
@app.post("/slack/events")
async def slack_events(request: Request):
    """Pass incoming requests to the Slack handler"""
    return await slack_handler.handle(request)
