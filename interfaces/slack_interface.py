import os
import logging
import time
from typing import Dict, Any, Optional
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

# Import agent and runtime components
from agents.client_agent import ClientAgent
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
        # Initialize Slack app for webhook mode
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Initialize agent and executor
        self.agent = ClientAgent()
        self.tool_executor = ToolExecutor()
        
        # Setup Slack event handlers
        self._setup_handlers()
        
        # Create FastAPI handler for webhooks
        self.handler = AsyncSlackRequestHandler(self.app)
    
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
        
        @self.app.action("view_execution_details")
        async def handle_execution_details(ack, body, respond, logger):
            await self._handle_execution_details_view(ack, body, respond, logger)
    
    async def _handle_user_message(self, event: Dict[str, Any], say, logger):
        """Handle incoming user messages"""
        try:
            # Extract message details
            user_id = event.get("user")
            channel_id = event.get("channel")
            message_text = event.get("text", "")
            thread_ts = event.get("thread_ts") or event.get("ts")
            
            # DEBUG: Log user_id extraction
            logger.info(f"SLACK-USER-ID: Extracted user_id='{user_id}' from event")
            logger.info(f"SLACK-MESSAGE: User '{user_id}' sent: '{message_text[:50]}...'")
            
            # Skip bot messages
            if event.get("bot_id"):
                logger.debug(f"SLACK-SKIP: Ignoring bot message from {event.get('bot_id')}")
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
            
            # Process message through agent with modal execution tracking
            try:
                # Open execution modal (need trigger_id from event)
                modal_response = await self._open_execution_modal(user_id, message_text, event.get("ts"))
                modal_view_id = modal_response.get("view", {}).get("id") if modal_response else None
                
                # Create streaming callback for modal updates
                async def slack_streaming_callback(text: str, msg_type: str):
                    await self._update_execution_modal(modal_view_id, text, msg_type)
                
                # Process with live modal updates
                context = {
                    "platform": "slack",
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "thread_ts": thread_ts
                }
                
                # DEBUG: Log context being passed to agent
                logger.info(f"SLACK-CONTEXT: Passing user_id='{user_id}' to ClientAgent")
                logger.info(f"SLACK-CONTEXT: Full context={context}")
                
                start_time = time.time()
                response = await self.agent.process_request(
                    message_text,
                    context=context,
                    streaming_callback=slack_streaming_callback
                )
                
                # Mark execution complete in modal
                duration_ms = int((time.time() - start_time) * 1000)
                await self._complete_execution_modal(modal_view_id, duration_ms)
                
                # Send only the final response to conversation (clean!)
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
        
        # Add execution details button
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🔍 View Execution Details"
                    },
                    "action_id": "view_execution_details",
                    "value": f"execution_details_{int(time.time())}"
                }
            ]
        })
        
        await say(
            blocks=blocks,
            text=message_text,  # Fallback for notifications
            thread_ts=thread_ts
        )
    
    # =============================================================================
    # BLOCK KIT MODAL EXECUTION SYSTEM
    # =============================================================================
    
    def _filter_execution_message(self, text: str, msg_type: str) -> str:
        """Filter and clean execution messages for display"""
        # All ClientAgent messages come as msg_type "thinking", so we filter by content
        
        if text.startswith("🔧 Executing"):
            # Tool execution: "🔧 Executing reg_search..." → "Executing reg_search"
            clean_text = text.replace("🔧 Executing ", "").replace("...", "")
            return f"Executing {clean_text}"
            
        elif text.startswith("📋 Got result"):
            # Skip verbose result messages - too much detail
            return None
            
        elif any(keyword in text.lower() for keyword in ["search", "tool", "need", "should", "execute", "call", "discover", "find"]):
            # Show meaningful thinking that indicates what the agent is doing
            return text
            
        # Skip generic thinking or overly detailed thoughts
        return None
    
    async def _open_execution_modal(self, user_id: str, message_text: str, trigger_ts: str) -> dict:
        """Open Block Kit modal for live execution tracking"""
        try:
            modal_view = {
                "type": "modal",
                "title": {
                    "type": "plain_text",
                    "text": "🤖 Agent Execution"
                },
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn", 
                            "text": f"*Query:* {message_text[:100]}{'...' if len(message_text) > 100 else ''}"
                        }
                    },
                    {
                        "type": "section",
                        "block_id": "status_section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Status:* 🔄 Working..."
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "block_id": "execution_steps",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🔄 Execution Steps:*\n• Just a sec..."
                        }
                    }
                ]
            }
            
            # Use a webhook or direct API call since we don't have trigger_id
            # For now, we'll simulate the modal opening
            logger.info(f"Would open execution modal for user {user_id}")
            return {"view": {"id": f"modal_{user_id}_{trigger_ts}"}}
            
        except Exception as e:
            logger.error(f"Failed to open execution modal: {e}")
            return None
    
    async def _update_execution_modal(self, view_id: str, text: str, msg_type: str):
        """Update execution modal with filtered progress"""
        try:
            if not view_id:
                return
                
            # Filter the message
            filtered_text = self._filter_execution_message(text, msg_type)
            if not filtered_text:
                return  # Skip this message
            
            # For now, just log the filtered content (in real implementation, would update modal)
            logger.info(f"MODAL UPDATE [{view_id}]: {filtered_text}")
            
        except Exception as e:
            logger.error(f"Failed to update execution modal: {e}")
    
    async def _complete_execution_modal(self, view_id: str, duration_ms: int = None):
        """Mark execution modal as complete"""
        try:
            if not view_id:
                return
                
            duration_text = f" ({duration_ms/1000:.1f}s)" if duration_ms else ""
            logger.info(f"MODAL COMPLETE [{view_id}]: Done!{duration_text}")
            
        except Exception as e:
            logger.error(f"Failed to complete execution modal: {e}")
    
    async def _handle_execution_details_view(self, ack, body, respond, logger):
        """Handle when user clicks 'View Execution Details' button"""
        await ack()
        
        try:
            user_id = body.get("user", {}).get("id")
            
            # Create a sample execution modal (in real implementation, would fetch stored details)
            modal_view = {
                "type": "modal",
                "title": {
                    "type": "plain_text",
                    "text": "🤖 Execution Details"
                },
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Query:* What's the weather in London?"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Status:* ✅ Complete (4.2s)"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🔄 Execution Steps:*\n• Just a sec...\n• I need to search for weather tools\n• Executing reg_search\n• Now I'll use the weather.search tool\n• Executing weather.search\n• Done!"
                        }
                    }
                ]
            }
            
            # Open the modal (would need proper trigger_id in real implementation)
            logger.info(f"Would show execution details modal for user {user_id}")
            
            # For now, send an ephemeral message
            await respond({
                "text": "🔍 *Execution Details*\n\n*Steps taken:*\n• I need to search for weather tools\n• Executing reg_search\n• Now I'll use the weather.search tool\n• Executing weather.search\n• Done!\n\n*Duration:* 4.2s",
                "response_type": "ephemeral"
            })
            
        except Exception as e:
            logger.error(f"Error showing execution details: {e}")
            await respond({
                "text": "Sorry, I couldn't load the execution details.",
                "response_type": "ephemeral"
            })
    
    async def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack interface...")
        
        # Verify environment variables (no SLACK_APP_TOKEN needed for webhooks)
        required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing Slack configuration: {missing_vars}")
        
        logger.info("Slack interface initialized for webhook mode")
        logger.info("Ready to receive events via webhook endpoints")
    
    async def stop(self):
        """Stop the Slack bot"""
        logger.info("Stopping Slack interface...")
        logger.info("Slack interface stopped")

    def get_fastapi_handler(self):
        """Get the FastAPI handler for webhook integration"""
        return self.handler

# For webhook deployment with FastAPI
def create_slack_app():
    """Create Slack interface for FastAPI integration"""
    slack_interface = SlackInterface()
    return slack_interface.get_fastapi_handler()

# For standalone testing (webhook simulation)
if __name__ == "__main__":
    import asyncio
    
    async def main():
        slack_interface = SlackInterface()
        await slack_interface.start()
        
        print("=" * 60)
        print("🚀 SLACK WEBHOOK MODE")
        print("=" * 60)
        print("Slack interface is ready for webhook events!")
        print("")
        print("📝 Next steps:")
        print("1. Deploy this with FastAPI/uvicorn")
        print("2. Set up webhook URL in Slack app settings")
        print("3. Configure event subscriptions")
        print("")
        print("Example deployment:")
        print("  uvicorn slack_webhook_server:app --host 0.0.0.0 --port 8000")
        print("=" * 60)
        
        # Keep alive for testing
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await slack_interface.stop()
    
    asyncio.run(main()) 