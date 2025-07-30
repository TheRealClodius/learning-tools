import os
import logging
from typing import Dict, Any, Optional
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

# Import agent and runtime components
from agents.research_agent import ResearchAgent
from runtime.tool_executor import ToolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackInterface:
    """
    Slack bot interface for agent conversations
    
    Handles user interactions through Slack, providing a conversational
    interface to the agent's capabilities.
    """
    
    def __init__(self):
        # Initialize Slack app
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Initialize agent and executor
        self.agent = ResearchAgent()
        self.tool_executor = ToolExecutor()
        
        # Setup Slack event handlers
        self._setup_handlers()
        
        # Socket mode handler for development
        self.socket_handler = AsyncSocketModeHandler(
            self.app, 
            os.environ.get("SLACK_APP_TOKEN")
        )
    
    def _setup_handlers(self):
        """Setup Slack event handlers"""
        
        # Handle direct messages and mentions
        @self.app.event("message")
        async def handle_message(event, say, logger):
            await self._handle_user_message(event, say, logger)
        
        # Handle app mentions (@botname)
        @self.app.event("app_mention")
        async def handle_mention(event, say, logger):
            await self._handle_user_message(event, say, logger)
        
        # Handle slash commands
        @self.app.command("/agent")
        async def handle_agent_command(ack, respond, command, logger):
            await self._handle_slash_command(ack, respond, command, logger)
        
        # Handle interactive buttons/actions
        @self.app.action("agent_action")
        async def handle_agent_action(ack, body, respond, logger):
            await self._handle_interactive_action(ack, body, respond, logger)
    
    async def _handle_user_message(self, event: Dict[str, Any], say, logger):
        """Handle incoming user messages"""
        try:
            # Extract message details
            user_id = event.get("user")
            channel_id = event.get("channel")
            message_text = event.get("text", "")
            thread_ts = event.get("thread_ts") or event.get("ts")
            
            # Skip bot messages
            if event.get("bot_id"):
                return
            
            # Remove bot mention if present
            message_text = self._clean_message_text(message_text)
            
            if not message_text.strip():
                await say(
                    text="Hi! I'm your AI agent. How can I help you today?",
                    thread_ts=thread_ts
                )
                return
            
            logger.info(f"Processing message from user {user_id}: {message_text[:50]}...")
            
            # Show typing indicator
            await self._show_typing(channel_id)
            
            # Process message through agent
            try:
                response = await self.agent.process_request(
                    message_text,
                    context={
                        "platform": "slack",
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "thread_ts": thread_ts
                    }
                )
                
                # Send response back to Slack
                await self._send_response(say, response, thread_ts)
                
            except Exception as e:
                logger.error(f"Error processing agent request: {e}")
                await say(
                    text="Sorry, I encountered an error processing your request. Please try again.",
                    thread_ts=thread_ts
                )
                
        except Exception as e:
            logger.error(f"Error handling Slack message: {e}")
    
    async def _handle_slash_command(self, ack, respond, command, logger):
        """Handle slash commands like /agent"""
        await ack()
        
        try:
            user_id = command.get("user_id")
            command_text = command.get("text", "")
            
            logger.info(f"Processing slash command from {user_id}: {command_text}")
            
            if not command_text.strip():
                await respond({
                    "text": "Usage: `/agent <your message>`\nExample: `/agent What's the weather in London?`",
                    "response_type": "ephemeral"
                })
                return
            
            # Process command through agent
            response = await self.agent.process_request(
                command_text,
                context={
                    "platform": "slack",
                    "user_id": user_id,
                    "interaction_type": "slash_command"
                }
            )
            
            # Format response for slash command
            response_text = response.get("message", "No response generated")
            
            await respond({
                "text": response_text,
                "response_type": "in_channel"  # Make response visible to all
            })
            
        except Exception as e:
            logger.error(f"Error handling slash command: {e}")
            await respond({
                "text": "Sorry, I encountered an error processing your command.",
                "response_type": "ephemeral"
            })
    
    async def _handle_interactive_action(self, ack, body, respond, logger):
        """Handle interactive buttons and actions"""
        await ack()
        
        try:
            user_id = body.get("user", {}).get("id")
            action_value = body.get("actions", [{}])[0].get("value", "")
            
            logger.info(f"Processing interactive action from {user_id}: {action_value}")
            
            # Process action through agent
            response = await self.agent.process_request(
                f"User clicked: {action_value}",
                context={
                    "platform": "slack",
                    "user_id": user_id,
                    "interaction_type": "button_click"
                }
            )
            
            response_text = response.get("message", "Action processed")
            
            await respond({
                "text": response_text,
                "replace_original": False
            })
            
        except Exception as e:
            logger.error(f"Error handling interactive action: {e}")
            await respond({
                "text": "Sorry, I encountered an error processing your action."
            })
    
    def _clean_message_text(self, text: str) -> str:
        """Remove bot mentions and clean message text"""
        import re
        
        # Remove bot mentions like <@U1234567890>
        text = re.sub(r'<@[UW][A-Z0-9]{8,}>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def _show_typing(self, channel_id: str):
        """Show typing indicator in Slack"""
        try:
            # Note: Slack doesn't have a direct typing indicator API
            # This is a placeholder for potential future implementation
            pass
        except Exception as e:
            logger.warning(f"Could not show typing indicator: {e}")
    
    async def _send_response(self, say, response: Dict[str, Any], thread_ts: str):
        """Send formatted response back to Slack"""
        
        message_text = response.get("message", "No response generated")
        tool_calls = response.get("tool_calls", [])
        
        # Basic text response
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message_text
                }
            }
        ]
        
        # Add tool call information if present
        if tool_calls:
            tool_info = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool", "unknown")
                tool_info.append(f"• Used tool: `{tool_name}`")
            
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "\n".join(tool_info)
                    }
                ]
            })
        
        # Add action buttons if relevant
        if "weather" in message_text.lower():
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Get Forecast"
                        },
                        "action_id": "agent_action",
                        "value": "get_weather_forecast"
                    }
                ]
            })
        
        await say(
            blocks=blocks,
            text=message_text,  # Fallback for notifications
            thread_ts=thread_ts
        )
    
    async def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack interface...")
        
        # Verify environment variables
        required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "SLACK_APP_TOKEN"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing Slack configuration: {missing_vars}")
        
        try:
            # Start socket mode handler
            await self.socket_handler.start_async()
            logger.info("Slack bot started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Slack bot: {e}")
            raise
    
    async def stop(self):
        """Stop the Slack bot"""
        logger.info("Stopping Slack interface...")
        try:
            await self.socket_handler.close_async()
            logger.info("Slack bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Slack bot: {e}")

# For standalone running
if __name__ == "__main__":
    import asyncio
    
    async def main():
        slack_interface = SlackInterface()
        await slack_interface.start()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await slack_interface.stop()
    
    asyncio.run(main()) 