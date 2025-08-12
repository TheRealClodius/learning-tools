import os
import logging
import asyncio
import re
import time
import json
from typing import Dict, Any, Optional, List
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

# Import agent components
from agents.client_agent import ClientAgent

# Import rate limit handler
from runtime.rate_limit_handler import (
    RateLimitHandler, RateLimitConfig, RateLimitError,
    with_rate_limit, get_global_handler
)

# Import database for user caching
from database import user_db

# Import modular Slack components
from .formatters.markdown_parser import MarkdownToSlackParser
from .formatters.mention_resolver import SlackMentionResolver
from agents.execution_summarizer import ExecutionSummarizer
from .services.cache_service import SlackCacheService
from .services.user_service import SlackUserService
from .handlers.streaming_handler import SlackStreamingHandler
from .handlers.modal_handler import SlackModalHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SlackInterface:
    """
    Simple Slack bot interface - just messages in, responses out
    """
    
    def __init__(self):
        # Initialize Slack app
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Initialize agent
        self.agent = ClientAgent()
        
        # Initialize cleanup task tracking
        self._cleanup_task_started = False
        
        # Initialize modular services
        self.cache_service = SlackCacheService()
        self.user_service = SlackUserService(self.app.client)
        self.modal_handler = SlackModalHandler(self.cache_service)
        self.mention_resolver = SlackMentionResolver(self.cache_service, self.user_service)
        
        # Setup handlers
        self._setup_handlers()
        
        # Create FastAPI handler
        self.handler = AsyncSlackRequestHandler(self.app)
    
    async def _ensure_cleanup_task_started(self):
        """Start the cleanup task if not already started"""
        if not self._cleanup_task_started:
            self._cleanup_task_started = True
            asyncio.create_task(self._cleanup_cache_periodically())
    
    def _setup_handlers(self):
        """Setup basic Slack event handlers"""
        
        @self.app.event("message")
        async def handle_message(event, say, logger):
            await self._handle_message(event, say, logger)
        
        @self.app.event("app_mention")
        async def handle_mention(event, say, logger):
            await self._handle_message(event, say, logger)
        
        @self.app.action("view_execution_details")
        async def handle_execution_details_button(ack, body, client):
            # DEBUG: Log button click
            logger.info("DEBUG: view_execution_details button clicked!")
            logger.info(f"DEBUG: Button body keys: {list(body.keys())}")
            
            # Ensure cleanup task is started
            await self._ensure_cleanup_task_started()
            
            # CRITICAL: Ack immediately to avoid Slack 3s timeout - this must be first!
            try:
                await ack()
                logger.info(f"Acknowledged view_execution_details button click from user {body.get('user', {}).get('id')}")
            except Exception as e:
                logger.warning(f"Slack ack failed for view_execution_details: {e}")
                # If ack fails, we should not proceed as the trigger_id might be invalid
                return
            
            # DEBUG: Log after ack
            logger.info("DEBUG: Successfully acknowledged button click, proceeding to show modal...")
            
            # OPTIMIZATION: Remove artificial delay - every millisecond counts with trigger_id timeout
            # Original: await asyncio.sleep(0.1)  # REMOVED - wasting precious time
            
            # IMPROVED: Fast-fail approach with minimal retries to stay within 3s window
            max_retries = 2  # Reduced from 3 to save time
            retry_delay = 0.05  # Optimized: Start at 50ms
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"DEBUG: Attempt {attempt + 1} to show modal...")
                    await self.modal_handler.show_execution_details_modal(body, client)
                    logger.info(f"Successfully opened execution details modal on attempt {attempt + 1}")
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed to open modal: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 0.15)  # Optimized: Cap at 150ms
                    else:
                        # Final attempt failed, notify user
                        logger.error("DEBUG: All attempts to open modal failed!")
                        try:
                            channel_id = body.get("channel", {}).get("id") or body.get("container", {}).get("channel_id")
                            user_id = body.get("user", {}).get("id")
                            if channel_id and user_id:
                                await client.chat_postEphemeral(
                                    channel=channel_id,
                                    user=user_id,
                                    text="Unable to open execution details. Please try again."
                                )
                        except:
                            pass

        @self.app.action("flow_next_page")
        async def handle_flow_next_page(ack, body, client):
            await ack()
            try:
                meta = json.loads(body.get("view", {}).get("private_metadata", "{}"))
                message_ts = meta.get("message_ts")
                page = int(meta.get("page", 0)) + 1
                total_pages = int(meta.get("total_pages", 1))
                if page >= total_pages:
                    page = total_pages - 1
                await self.modal_handler.update_execution_modal_page(client, body, message_ts, page)
            except Exception as e:
                logger.error(f"Error handling next page: {e}")

        @self.app.action("flow_prev_page")
        async def handle_flow_prev_page(ack, body, client):
            await ack()
            try:
                meta = json.loads(body.get("view", {}).get("private_metadata", "{}"))
                message_ts = meta.get("message_ts")
                page = int(meta.get("page", 0)) - 1
                if page < 0:
                    page = 0
                await self.modal_handler.update_execution_modal_page(client, body, message_ts, page)
            except Exception as e:
                logger.error(f"Error handling prev page: {e}")
        
        @self.app.event("assistant_thread_context_changed")
        async def handle_assistant_thread_context_changed(body, logger):
            """Handle assistant thread context changed events"""
            logger.info(f"Assistant thread context changed: {body}")
            # This event is informational - no action needed
            # Just acknowledging to avoid unhandled request warnings
    
    async def _handle_message(self, event: Dict[str, Any], say, logger):
        """Handle incoming messages - simplified version"""
        try:
            # Ensure cleanup task is started
            await self._ensure_cleanup_task_started()
            
            t_start = time.time()
            # Skip bot messages
            if event.get("bot_id"):
                return
            
            # Extract basics
            user_id = event.get("user")
            channel_id = event.get("channel")
            message_text = event.get("text", "").strip()
            thread_ts = event.get("thread_ts") or event.get("ts")
            
            # PERFORMANCE OPTIMIZATION: Parallel API calls (saves 200-400ms)
            # These operations are independent and can run concurrently
            mention_task = asyncio.create_task(
                self.mention_resolver.parse_slack_mentions(message_text, self.app.client)
            )
            user_info_task = asyncio.create_task(self._get_user_info(user_id))
            
            # Wait for both operations to complete in parallel
            t_parallel_start = time.time()
            (message_text, mention_context), user_info = await asyncio.gather(
                mention_task, user_info_task
            )
            logger.info(f"SLACK-TIMING: parallel API calls took {int((time.time()-t_parallel_start)*1000)} ms user={user_id}")
            
            if not message_text:
                greeting_blocks = MarkdownToSlackParser.parse_to_blocks("Hi! How can I help you? 👋")
                if not greeting_blocks:
                    greeting_blocks = [{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Hi! How can I help you? 👋"
                        }
                    }]
                
                await self.app.client.chat_postMessage(
                    channel=channel_id,
                    blocks=greeting_blocks,
                    text="Hi! How can I help you?",  # Fallback for notifications
                    thread_ts=thread_ts
                )
                return
            
            logger.info(f"User {user_id}: {message_text[:50]}...")
            
            # Process asynchronously - pass user_info to avoid duplicate API call
            asyncio.create_task(self._process_and_respond(
                user_id, channel_id, message_text, thread_ts, mention_context, user_info
            ))

            logger.info(f"SLACK-TIMING: _handle_message dispatched in {int((time.time()-t_start)*1000)} ms for user={user_id}")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_and_respond(self, user_id: str, channel_id: str, message_text: str, thread_ts: str, mention_context: dict, user_info: dict):
        """Process message and send response with execution streaming"""
        streaming_handler = None
        try:
            t_overall = time.time()
            # Create streaming handler for this message
            streaming_handler = SlackStreamingHandler(self.app.client, channel_id, thread_ts, user_id)
            
            # Start streaming display
            t_stream_start = time.time()
            await streaming_handler.start_streaming()
            logger.info(f"SLACK-TIMING: start_streaming took {int((time.time()-t_stream_start)*1000)} ms user={user_id}")
            
            # Create simple streaming callback
            async def slack_streaming_callback(content: str, content_type: str):
                if content_type == "thinking" and content.strip():
                    await streaming_handler.update_thinking(content.strip())
                elif content_type == "tool_start":
                    # Check if content includes tool args (for execute_tool)
                    if isinstance(content, dict):
                        tool_name = content.get("name", "unknown")
                        tool_args = content.get("args", {})
                        await streaming_handler.start_tool(tool_name, tool_args)
                    else:
                        await streaming_handler.start_tool(content)
                elif content_type in ["tool_details", "operation", "result", "result_detail", "error"]:
                    # DEBUG: These intermediate operations are now only logged for debugging
                    # Gemini creates complete summaries in ClientAgent, so no need to show these in UI
                    logger.debug(f"INTERMEDIATE-OP: {content_type}: {content[:100]}...")
                elif content_type == "tool_result":
                    # Complete tool with final summary (content is already the complete summary)
                    await streaming_handler.complete_tool(content.strip())
                elif content_type == "tool_summary_chunk":
                    # Stream Gemini summary chunks as they arrive for progressive updates
                    await streaming_handler.append_to_current_tool(content)

            
            # Process through agent with streaming
            import datetime
            from zoneinfo import ZoneInfo
            
            # User info already fetched in parallel during message handling
            user_timezone = user_info.get('timezone', 'UTC')
            
            try:
                user_tz = ZoneInfo(user_timezone)
                current_time = datetime.datetime.now(user_tz)
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception as e:
                logger.warning(f"Failed to get user timezone {user_timezone}: {e}")
                # Fallback to UTC
                current_time = datetime.datetime.now(datetime.timezone.utc)
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            context = {
                "platform": "slack",
                "user_id": user_id,
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "timestamp": timestamp,
                "user_timezone": user_timezone,
                "user_name": user_info.get('real_name', ''),
                "user_display_name": user_info.get('display_name', ''),
                "user_title": user_info.get('title', '')
            }
            
            t_agent = time.time()
            response = await self.agent.process_request(
                message_text, 
                context=context,
                streaming_callback=slack_streaming_callback
            )
            logger.info(f"SLACK-TIMING: agent.process_request took {int((time.time()-t_agent)*1000)} ms user={user_id}")
            
            # Get response text and agent info
            response_text = response.get("response", "No response generated")
            agent_used = response.get("agent", "unknown")
            
            # Convert mentions back to Slack format for proper linking
            response_text = await self.mention_resolver.resolve_and_format_mentions(response_text, mention_context, self.app.client)
            
            # Handle response based on which agent was used
            t_finish = time.time()
            if agent_used.startswith("gemini"):
                # Gemini responses: simple update without execution details
                await self._handle_simple_response(response_text, streaming_handler)
            else:
                # Claude responses: full execution details tracking
                await streaming_handler.finish_with_response(response_text, self)
            logger.info(f"SLACK-TIMING: finish_with_response took {int((time.time()-t_finish)*1000)} ms user={user_id}")
            
            logger.info(f"SLACK-TIMING: end-to-end for user={user_id} ms={int((time.time()-t_overall)*1000)}")
            logger.info(f"Sent response to {user_id}: {response_text[:50]}...")
            
        except RateLimitError as e:
            logger.warning(f"Rate limit error for user {user_id}: {e}")
            
            # Show user-friendly rate limit message
            if streaming_handler and streaming_handler.message_ts:
                try:
                    # The RateLimitError already contains a user-friendly message
                    await streaming_handler.finish_with_response(str(e), self)
                except:
                    pass  # Fallback failed, just log
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Check if this is a rate limit error that wasn't caught above
            error_str = str(e).lower()
            if 'rate_limit_error' in error_str or '429' in error_str:
                friendly_msg = (
                    "⏳ The service is experiencing high demand. "
                    "I'll automatically retry your request in a moment. "
                    "If this persists, please try again in a few minutes."
                )
                if streaming_handler and streaming_handler.message_ts:
                    try:
                        await streaming_handler.finish_with_response(friendly_msg, self)
                    except:
                        pass
            else:
                # Other errors - show generic message
                if streaming_handler and streaming_handler.message_ts:
                    try:
                        error_text = f"❌ **Error Processing Request**\n\nSorry, I encountered an error while processing your message:\n\n`{str(e)}`"
                        await streaming_handler.finish_with_response(error_text, self)
                    except:
                        pass  # Fallback failed, just log
    
    async def _get_user_timezone(self, user_id: str) -> str:
        """Get user's timezone from their Slack profile"""
        try:
            # Call Slack API to get user info
            response = await self.app.client.users_info(user=user_id)
            
            if response["ok"]:
                user_info = response["user"]
                
                # Try to get timezone from user profile
                timezone = user_info.get("tz")
                if timezone:
                    logger.info(f"Found timezone for user {user_id}: {timezone}")
                    return timezone
                
                # Fallback: try to get from tz_label 
                tz_label = user_info.get("tz_label")
                if tz_label:
                    logger.info(f"Using tz_label for user {user_id}: {tz_label}")
                    # Map common timezone labels to IANA timezone identifiers
                    timezone_mapping = {
                        "Eastern Standard Time": "America/New_York",
                        "Eastern Daylight Time": "America/New_York", 
                        "Central Standard Time": "America/Chicago",
                        "Central Daylight Time": "America/Chicago",
                        "Mountain Standard Time": "America/Denver",
                        "Mountain Daylight Time": "America/Denver",
                        "Pacific Standard Time": "America/Los_Angeles",
                        "Pacific Daylight Time": "America/Los_Angeles",
                        "Greenwich Mean Time": "UTC",
                        "Central European Time": "Europe/Paris",
                        "Central European Summer Time": "Europe/Paris",
                        "Eastern European Time": "Europe/Bucharest",
                        "Eastern European Summer Time": "Europe/Bucharest"
                    }
                    
                    mapped_tz = timezone_mapping.get(tz_label, "UTC")
                    logger.info(f"Mapped {tz_label} to {mapped_tz} for user {user_id}")
                    return mapped_tz
                    
            logger.warning(f"Could not get timezone for user {user_id}, using UTC")
            return "UTC"
            
        except Exception as e:
            logger.error(f"Error getting user timezone for {user_id}: {e}")
            return "UTC"
    
    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information with database caching
        
        First checks database, then fetches from Slack API if not found or stale
        """
        try:
            # Check database first
            cached_user = await user_db.get_user(user_id)
            
            if cached_user:
                logger.info(f"Found user {user_id} in database cache")
                return {
                    'real_name': cached_user.get('real_name', ''),
                    'display_name': cached_user.get('display_name', ''),
                    'title': cached_user.get('title', ''),
                    'timezone': cached_user.get('timezone', 'UTC')
                }
            
            # Not in database, fetch from Slack API
            logger.info(f"User {user_id} not in database, fetching from Slack API")
            response = await self.app.client.users_info(user=user_id)
            
            if response["ok"]:
                user_info = response["user"]
                
                # Save to database for future use
                await user_db.save_user(user_id, user_info)
                
                # Extract key fields
                profile = user_info.get('profile', {})
                return {
                    'real_name': user_info.get('real_name', profile.get('real_name', '')),
                    'display_name': profile.get('display_name', user_info.get('name', '')),
                    'title': profile.get('title', ''),
                    'timezone': user_info.get('tz', 'UTC')
                }
            else:
                logger.error(f"Slack API error for user {user_id}: {response.get('error')}")
                return {
                    'real_name': '',
                    'display_name': '',
                    'title': '',
                    'timezone': 'UTC'
                }
                
        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {e}")
            return {
                'real_name': '',
                'display_name': '',
                'title': '',
                'timezone': 'UTC'
            }
    
    async def _handle_simple_response(self, response_text: str, streaming_handler):
        """Handle simple Gemini responses without execution details tracking"""
        try:
            # Parse the response as markdown and convert to Slack blocks
            response_blocks = MarkdownToSlackParser.parse_to_blocks(response_text)
            if not response_blocks:
                response_blocks = [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": response_text
                    }
                }]
            
            # Simple update - no execution details button for Gemini responses
            await streaming_handler.app_client.chat_update(
                channel=streaming_handler.channel_id,
                ts=streaming_handler.message_ts,
                blocks=response_blocks,
                text=response_text  # Fallback for notifications
            )
            
        except Exception as e:
            logger.error(f"Error updating simple response: {e}")
            # Fallback to simple text block
            await streaming_handler.app_client.chat_update(
                channel=streaming_handler.channel_id,
                ts=streaming_handler.message_ts,
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": response_text or "Error displaying response"
                    }
                }],
                text=response_text
            )
    
    async def _cleanup_cache_periodically(self):
        """Background task to clean up stale cache entries - delegated to cache service"""
        await self.cache_service._cleanup_cache_periodically()
    
    def get_fastapi_handler(self):
        """Get FastAPI handler for webhook integration"""
        return self.handler



# For FastAPI integration
def create_slack_app():
    """Create simplified Slack interface for FastAPI"""
    slack_interface = SlackInterface()
    return slack_interface.get_fastapi_handler()