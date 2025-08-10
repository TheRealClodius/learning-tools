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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarkdownToSlackParser:
    """Parse markdown content and convert to Slack Block Kit format"""
    
    @staticmethod
    def parse_to_blocks(text: str) -> List[Dict[str, Any]]:
        """Parse markdown text and return Slack blocks"""
        if not text or not text.strip():
            return []
        
        blocks = []
        lines = text.strip().split('\n')
        current_section = []
        in_code_block = False
        code_block_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if code_block_content:
                        code_text = '\n'.join(code_block_content)
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"```\n{code_text}\n```"
                            }
                        })
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    if current_section:
                        # Flush current section
                        section_text = '\n'.join(current_section)
                        if section_text.strip():
                            blocks.extend(MarkdownToSlackParser._create_section_blocks(section_text))
                        current_section = []
                    in_code_block = True
                i += 1
                continue
            
            if in_code_block:
                code_block_content.append(line)
                i += 1
                continue
            
            # Handle headers
            if line.startswith('#'):
                # Flush current section
                if current_section:
                    section_text = '\n'.join(current_section)
                    if section_text.strip():
                        blocks.extend(MarkdownToSlackParser._create_section_blocks(section_text))
                    current_section = []
                
                # Process header
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('# ').strip()
                
                if header_level == 1:
                    # Main header - use larger text
                    blocks.append({
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": header_text
                        }
                    })
                else:
                    # Sub-headers - use bold text
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{header_text}*"
                        }
                    })
                
                i += 1
                continue
            
            # Handle dividers
            if line.strip() in ['---', '***', '___']:
                if current_section:
                    section_text = '\n'.join(current_section)
                    if section_text.strip():
                        blocks.extend(MarkdownToSlackParser._create_section_blocks(section_text))
                    current_section = []
                
                blocks.append({"type": "divider"})
                i += 1
                continue
            
            # Handle lists and regular content
            current_section.append(line)
            i += 1
        
        # Flush remaining content
        if current_section:
            section_text = '\n'.join(current_section)
            if section_text.strip():
                blocks.extend(MarkdownToSlackParser._create_section_blocks(section_text))
        
        return blocks
    
    @staticmethod
    def _create_section_blocks(text: str) -> List[Dict[str, Any]]:
        """Create section blocks from text, handling long content"""
        if not text.strip():
            return []
        
        # Convert markdown to Slack mrkdwn
        slack_text = MarkdownToSlackParser._convert_markdown_to_slack(text)
        
        # Split long content into multiple blocks (Slack has a 3000 char limit)
        max_length = 2800  # Leave some margin
        
        if len(slack_text) <= max_length:
            return [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": slack_text
                }
            }]
        
        # Split into multiple blocks
        blocks = []
        parts = MarkdownToSlackParser._split_text_intelligently(slack_text, max_length)
        
        for part in parts:
            if part.strip():
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": part.strip()
                    }
                })
        
        return blocks
    
    @staticmethod
    def _convert_markdown_to_slack(text: str) -> str:
        """Convert markdown formatting to Slack mrkdwn"""
        # Convert bold (**text** or __text__ -> *text*)
        text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        text = re.sub(r'__(.*?)__', r'*\1*', text)
        
        # Convert italic (*text* or _text_ -> _text_) - but be careful not to conflict with bold
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'_\1_', text)
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'_\1_', text)
        
        # Convert inline code (`text` stays as `text`)
        # Already compatible with Slack
        
        # Convert links [text](url) -> <url|text>
        # Handle URLs with parentheses by using a more sophisticated pattern
        def replace_link(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<{url}|{link_text}>'
        
        # Match [text](url) where url can contain balanced parentheses
        text = re.sub(r'\[([^\]]+)\]\(([^)]*(?:\([^)]*\)[^)]*)*)\)', replace_link, text)
        
        # Convert blockquotes (> text -> > text)
        # Already compatible with Slack
        
        # Handle lists (- or * -> •)
        lines = text.split('\n')
        converted_lines = []
        
        for line in lines:
            # Convert unordered lists
            if re.match(r'^\s*[-*+]\s+', line):
                indent = len(line) - len(line.lstrip())
                content = re.sub(r'^\s*[-*+]\s+', '', line)
                bullet = '•' if indent == 0 else '◦'
                converted_lines.append(' ' * indent + bullet + ' ' + content)
            # Convert ordered lists (1. text -> 1. text)
            elif re.match(r'^\s*\d+\.\s+', line):
                converted_lines.append(line)
            else:
                converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    @staticmethod
    def _split_text_intelligently(text: str, max_length: int) -> List[str]:
        """Split text at logical boundaries"""
        if len(text) <= max_length:
            return [text]
        
        parts = []
        current_part = ""
        lines = text.split('\n')
        
        for line in lines:
            # If adding this line would exceed the limit
            if len(current_part + '\n' + line) > max_length:
                if current_part:
                    parts.append(current_part)
                    current_part = line
                else:
                    # Single line is too long, split it by sentences or spaces
                    if len(line) > max_length:
                        words = line.split(' ')
                        temp_line = ""
                        for word in words:
                            if len(temp_line + ' ' + word) > max_length:
                                if temp_line:
                                    parts.append(temp_line)
                                temp_line = word
                            else:
                                temp_line = temp_line + ' ' + word if temp_line else word
                        current_part = temp_line
                    else:
                        current_part = line
            else:
                current_part = current_part + '\n' + line if current_part else line
        
        if current_part:
            parts.append(current_part)
        
        return parts

class SlackStreamingHandler:
    """Handles progressive message updates for agent execution streaming"""
    
    def __init__(self, app_client, channel_id: str, thread_ts: str):
        self.app_client = app_client
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.message_ts: Optional[str] = None
        self.content_blocks = []  # Chronological list of thinking and tool blocks
        self.current_tool_block = None  # Currently active tool block
        self.all_thinking = []  # All thinking content accumulated
        self.execution_summary = []  # Store execution details for modal
        
    async def start_streaming(self):
        """Post initial thinking message"""
        response = await self.app_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "_Reasoning..._"
                        }
                    ]
                }
            ],
            text="Reasoning...",
            thread_ts=self.thread_ts
        )
        self.message_ts = response["ts"]
        
    async def update_thinking(self, thinking_line: str):
        """Add thinking content - accumulate all thinking in current block"""
        if not self.message_ts or not thinking_line.strip():
            return
            
        self.all_thinking.append(thinking_line.strip())
        await self._update_display()
        
    async def start_tool(self, tool_name: str):
        """Start a new tool execution block"""
        # End any current thinking block by moving it to completed blocks
        if self.all_thinking:
            thinking_text = "\n".join(self.all_thinking)
            self.content_blocks.append(("thinking", thinking_text))
            # Store thinking in execution summary too
            self.execution_summary.append(("thinking", thinking_text))
            self.all_thinking = []  # Clear current thinking
        
        # Start new tool block
        self.current_tool_block = {
            "name": tool_name,
            "operations": [],
            "status": "running"
        }
        await self._update_display()
        
    async def add_tool_operation(self, operation: str):
        """Add operation to current tool block"""
        if self.current_tool_block:
            self.current_tool_block["operations"].append(operation)
            await self._update_display()
    
    async def complete_tool(self, result_summary: str):
        """Complete current tool and add result summary"""
        if self.current_tool_block:
            self.current_tool_block["status"] = "completed"
            
            # Special handling for execute_tool - create complete narrative
            if self.current_tool_block["name"].startswith("⚡️"):
                # Extract the existing operation that has the purpose
                existing_ops = self.current_tool_block.get("operations", [])
                if existing_ops and existing_ops[0].startswith("⚡️Using"):
                    # Parse the existing operation to get tool name and purpose
                    first_op = existing_ops[0]
                    # Change "Using" to "Used" and append the result
                    narrative = first_op.replace("⚡️Using", "⚡️Used")
                    
                    # Add the result in a narrative way
                    if result_summary:
                        # Clean up the result summary for narrative flow
                        if result_summary.startswith("Retrieved"):
                            narrative += f". {result_summary}"
                        elif result_summary.startswith("Found"):
                            narrative += f". {result_summary}"
                        elif result_summary.startswith("Completed"):
                            narrative += f". {result_summary}"
                        elif result_summary.startswith("⚠️"):
                            narrative += f". {result_summary}"
                        else:
                            narrative += f". Got {result_summary}"
                    else:
                        narrative += "."
                    
                    # Replace all operations with the single narrative
                    self.current_tool_block["operations"] = [narrative]
                else:
                    # Fallback if we don't have the expected format
                    if result_summary:
                        summary = self._create_result_summary(self.current_tool_block["name"], result_summary)
                        if summary:
                            self.current_tool_block["operations"].append(summary)
            else:
                # Regular tool handling
                if result_summary:
                    summary = self._create_result_summary(self.current_tool_block["name"], result_summary)
                    if summary:
                        self.current_tool_block["operations"].append(summary)
            
            # Add completed tool block to content blocks and execution summary
            completed_tool = self.current_tool_block.copy()
            self.content_blocks.append(("tool", completed_tool))
            self.execution_summary.append(("tool", completed_tool))
            self.current_tool_block = None  # Clear current tool
            # Note: Don't clear all_thinking here - let new thinking accumulate naturally
            await self._update_display()
    
    def _create_result_summary(self, tool_name: str, result_data: str) -> str:
        """Create a human-readable summary of tool results"""
        if not result_data.strip():
            return ""
            
        # Clean up the result data
        result_data = result_data.strip()
        
        # Handle execute_tool results specially - extract the actual tool name from the ⚡️ prefix
        if tool_name.startswith("⚡️"):
            actual_tool = tool_name.replace("⚡️", "").strip()
            
            # Try to parse the result for better summary
            try:
                import json
                if result_data.startswith("{") or result_data.startswith("["):
                    result_obj = json.loads(result_data)
                    
                    # Handle different result structures
                    if isinstance(result_obj, dict):
                        if "status" in result_obj and result_obj["status"] == "success":
                            if "data" in result_obj:
                                # Summarize based on the tool type and data
                                if "weather" in actual_tool.lower():
                                    return f"Retrieved weather data successfully"
                                elif "slack" in actual_tool.lower():
                                    if "messages" in str(result_obj.get("data", "")):
                                        count = len(result_obj.get("data", {}).get("messages", []))
                                        return f"Found {count} Slack messages"
                                    elif "channels" in str(result_obj.get("data", "")):
                                        count = len(result_obj.get("data", {}).get("channels", []))
                                        return f"Retrieved {count} Slack channels"
                                    else:
                                        return f"Completed Slack operation successfully"
                                elif "search" in actual_tool.lower():
                                    return f"Search completed with results"
                                else:
                                    return f"Operation completed successfully"
                            else:
                                return f"Completed successfully"
                        elif "error" in result_obj:
                            return f"⚠️ Error: {result_obj.get('error', 'Unknown error')[:100]}"
                        else:
                            # Try to extract meaningful info from the result
                            if len(result_data) > 100:
                                return f"Received detailed response"
                            else:
                                return f"Result: {result_data[:80]}..."
                    elif isinstance(result_obj, list):
                        return f"Retrieved {len(result_obj)} items"
                    else:
                        return f"Completed with result"
                else:
                    # Plain text result
                    if len(result_data) > 100:
                        return f"Received response: {result_data[:80]}..."
                    else:
                        return f"Result: {result_data}"
            except:
                # If parsing fails, provide generic summary
                if len(result_data) > 100:
                    return f"Completed with detailed output"
                else:
                    return f"Result: {result_data[:50]}..."
        
        # Handle different tool types with appropriate summaries
        elif "reg_search" in tool_name or "registry" in tool_name.lower():
            if "found" in result_data.lower() or "tools" in result_data.lower():
                return f"found relevant tools and capabilities"
            elif "no" in result_data.lower() and "results" in result_data.lower():
                return "no matching tools found"
            else:
                return "searched tool registry"
                

                
        elif "weather" in tool_name.lower():
            return f"retrieved weather information"
            
        elif "perplexity" in tool_name.lower() or "search" in tool_name.lower():
            # Extract more meaningful info from perplexity results
            if "answer" in result_data.lower():
                return f"found comprehensive answer via web search"
            elif "think" in result_data.lower():
                return f"retrieved analytical response from web search"  
            else:
                return f"completed web search with results"
        
        elif "memory" in tool_name.lower():
            # Handle memory tool results
            if "memory_retrieve" in tool_name:
                if "short_term_memory" in result_data and "retrieved_pages" in result_data:
                    return f"retrieved conversation history and context"
                elif "status" in result_data and "success" in result_data:
                    return f"found relevant conversation history"
                else:
                    return f"searched conversation history"
            elif "memory_add" in tool_name:
                if "success" in result_data.lower():
                    return f"saved conversation to memory"
                else:
                    return f"attempted to save conversation"
            else:
                return f"completed memory operation"
            
        else:
            # Generic summary - try to extract key info
            if len(result_data) > 100:
                return f"completed with results"
            else:
                return f"returned: {result_data[:50]}..."
        
    async def _update_display(self):
        """Update the streaming message display with chronological thinking and tool blocks"""
        if not self.message_ts:
            return
            
        blocks = []
        
        # Add all completed content blocks chronologically
        for block_type, content in self.content_blocks:
            if block_type == "thinking":
                # Thinking block: small font, italics - handle multi-line properly
                thinking_lines = content.split('\n')
                thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": thinking_formatted
                        }
                    ]
                })
            elif block_type == "tool":
                # Tool block: small font, italics, bold tool name  
                tool_info = content
                tool_lines = [f"*{tool_info['name']}*"]
                
                for operation in tool_info['operations']:
                    tool_lines.append(operation.strip())
                
                # Add status indicator (only if we have operations)
                if tool_info.get('operations'):
                    if tool_info['status'] == 'completed':
                        pass  # No extra indicator for completed - operations already show completion
                    else:
                        tool_lines.append("...")
                
                # Format each line with italics
                tool_formatted = '\n'.join([f"_{line}_" for line in tool_lines if line])
                
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": tool_formatted
                        }
                    ]
                })
        
        # Add current thinking block if we have ongoing thinking (only when no tool is running)
        if self.all_thinking and not self.current_tool_block:
            thinking_lines = self.all_thinking
            thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": thinking_formatted
                    }
                ]
            })
        
        # Add current tool block if we have one running
        if self.current_tool_block:
            tool_info = self.current_tool_block
            tool_lines = [f"*{tool_info['name']}*"]
            
            for operation in tool_info['operations']:
                tool_lines.append(operation.strip())
            
            # Add running indicator (only if we have operations)
            if tool_info.get('operations'):
                tool_lines.append("...")
            
            # Format each line with italics
            tool_formatted = '\n'.join([f"_{line}_" for line in tool_lines if line])
            
            blocks.append({
                "type": "context", 
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": tool_formatted
                    }
                ]
            })
        
        # If no blocks yet, show initial message
        if not blocks:
            blocks = [{
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn", 
                        "text": "_Reasoning..._"
                    }
                ]
            }]
        
        await self.app_client.chat_update(
            channel=self.channel_id,
            ts=self.message_ts,
            blocks=blocks,
            text="Agent working..."
        )

        
    async def finish_with_response(self, final_response: str, slack_interface=None):
        """Replace thinking message with final response and add execution details button"""
        if not self.message_ts:
            return
        
        # Store any remaining thinking in execution summary
        if self.all_thinking:
            thinking_text = "\n".join(self.all_thinking)
            self.execution_summary.append(("thinking", thinking_text))
        
        # Store any current tool in execution summary
        if self.current_tool_block:
            self.execution_summary.append(("tool", self.current_tool_block.copy()))
        
        # Store execution details in the interface cache for button access
        if slack_interface and self.execution_summary:
            # Add timestamp to cache entry
            slack_interface.execution_details_cache[self.message_ts] = (time.time(), self.execution_summary.copy())
        
        # Parse the response as markdown and convert to Slack blocks
        try:
            response_blocks = MarkdownToSlackParser.parse_to_blocks(final_response)
            
            # If no blocks were generated, fall back to simple text
            if not response_blocks:
                response_blocks = [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": final_response or "No response generated"
                    }
                }]
            
            # Add discrete execution details link if we have execution data
            if self.execution_summary:
                # Use small gray button (no "primary" style = default gray)
                execution_button_block = {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "view flow"
                            },
                            "action_id": "view_execution_details",
                            "value": self.message_ts
                        }
                    ]
                }
                response_blocks.append(execution_button_block)
            
            await self.app_client.chat_update(
                channel=self.channel_id,
                ts=self.message_ts,
                blocks=response_blocks,
                text=final_response  # Fallback for notifications
            )
            
        except Exception as e:
            logger.error(f"Error parsing markdown response: {e}")
            # Fallback to simple text block
            await self.app_client.chat_update(
                channel=self.channel_id,
                ts=self.message_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": final_response or "Error displaying response"
                        }
                    }
                ],
                text=final_response
            )

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
        
        # Store execution details for button access with timestamps
        self.execution_details_cache = {}  # message_ts -> (timestamp, execution_summary)
        self.cache_ttl = 3600  # 1 hour TTL for cache entries
        
        # Cache for user and channel lookups
        self.user_cache = {}  # { 'user_name': 'USER_ID' }
        self.channel_cache = {} # { 'channel_name': 'CHANNEL_ID' }
        
        # Setup handlers
        self._setup_handlers()
        
        # Create FastAPI handler
        self.handler = AsyncSlackRequestHandler(self.app)
        
        # Flag to track if background task has been started
        self._cleanup_task_started = False
    
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
            # Ensure cleanup task is started
            await self._ensure_cleanup_task_started()
            
            # Ack immediately to avoid Slack 3s timeout
            try:
                await ack()
                logger.info(f"Acknowledged view_execution_details button click from user {body.get('user', {}).get('id')}")
            except Exception as e:
                logger.warning(f"Slack ack failed for view_execution_details: {e}")
                # If ack fails, we should not proceed as the trigger_id might be invalid
                return
            
            # Use a small delay to ensure Slack has processed the ack
            await asyncio.sleep(0.1)
            
            # Open the modal with retry logic for reliability
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    await self._show_execution_details_modal(body, client)
                    logger.info(f"Successfully opened execution details modal on attempt {attempt + 1}")
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed to open modal: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Final attempt failed, notify user
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
                await self._update_execution_modal_page(client, body, message_ts, page)
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
                await self._update_execution_modal_page(client, body, message_ts, page)
            except Exception as e:
                logger.error(f"Error handling prev page: {e}")
    
    async def _show_execution_details_modal(self, body, client):
        """Show execution details in a modal"""
        try:
            # Log the incoming request details
            trigger_id = body.get("trigger_id")
            user_id = body.get("user", {}).get("id")
            logger.info(f"Opening execution details modal for user {user_id} with trigger_id {trigger_id}")
            
            # Validate trigger_id exists
            if not trigger_id:
                logger.error("No trigger_id found in body - cannot open modal")
                raise ValueError("Missing trigger_id")
            
            # Get the message timestamp from the button value
            message_ts = body["actions"][0]["value"]
            logger.info(f"Looking up execution details for message_ts: {message_ts}")
            
            # Get execution details from cache
            cached_entry = self.execution_details_cache.get(message_ts)
            
            if not cached_entry:
                logger.warning(f"No execution details found in cache for message_ts: {message_ts}")
                logger.debug(f"Current cache keys: {list(self.execution_details_cache.keys())}")
                await client.chat_postEphemeral(
                    channel=body["channel"]["id"],
                    user=body["user"]["id"],
                    text="Execution details not found or expired."
                )
                return
            
            # Check if cache entry is stale
            timestamp, execution_details = cached_entry
            if time.time() - timestamp > self.cache_ttl:
                logger.warning(f"Cache entry for message_ts {message_ts} is stale. TTL: {self.cache_ttl}s, Current age: {int(time.time() - timestamp)}s")
                await client.chat_postEphemeral(
                    channel=body["channel"]["id"],
                    user=body["user"]["id"],
                    text="Execution details are outdated. Please try again."
                )
                return

            logger.info(f"Found {len(execution_details)} execution detail blocks")
            
            # Build paginated modal pages with full content (no truncation) while respecting Slack limits
            pages = self._build_execution_pages(execution_details)
            total_pages = len(pages) if pages else 1
            page_index = 0
            logger.info(f"Built {total_pages} pages for modal")
            
            modal_view = self._build_modal_view(pages[page_index] if pages else [], page_index, total_pages, message_ts)
            
            # Log modal size for debugging
            modal_json = json.dumps(modal_view)
            logger.info(f"Modal JSON size: {len(modal_json)} bytes")
            
            # Open the modal
            logger.info(f"Attempting to open modal with trigger_id: {trigger_id}")
            response = await client.views_open(
                trigger_id=trigger_id,
                view=modal_view
            )
            
            if response.get("ok"):
                logger.info(f"Modal opened successfully with view_id: {response.get('view', {}).get('id')}")
            else:
                logger.error(f"Slack API returned not ok: {response}")
            
        except Exception as e:
            logger.error(f"Error showing execution details modal: {e}", exc_info=True)
            # Log additional context
            logger.error(f"Body keys: {list(body.keys())}")
            logger.error(f"User: {body.get('user', {}).get('id')}")
            logger.error(f"Channel: {body.get('channel', {}).get('id')}")
            
            try:
                channel_id = body.get("channel", {}).get("id") or body.get("container", {}).get("channel_id")
                user_id = body.get("user", {}).get("id")
                if channel_id and user_id:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text="Error displaying execution details."
                    )
            except:
                pass  # Ignore ephemeral message errors

    def _split_text_for_slack(self, text: str, max_len: int = 2900) -> list:
        """Split large text into multiple chunks that fit Slack text object limits."""
        if not text:
            return []
        lines = text.split("\n")
        chunks = []
        current = ""
        for line in lines:
            candidate = (current + "\n" + line) if current else line
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single line is too long, hard split it
                while len(line) > max_len:
                    chunks.append(line[:max_len])
                    line = line[max_len:]
                current = line
        if current:
            chunks.append(current)
        return chunks

    def _build_execution_pages(self, execution_details: list, max_blocks_per_page: int = 95) -> list:
        """Build pages of Slack blocks for the entire execution, no truncation (paginate instead)."""
        pages = [[]]
        block_count = 0

        def add_block(block):
            nonlocal block_count
            if block_count >= max_blocks_per_page:
                pages.append([])
                block_count = 0
            pages[-1].append(block)
            block_count += 1

        for block_type, content in execution_details:
            if block_type == "thinking":
                thinking_lines = content.split('\n') if isinstance(content, str) else [content]
                thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
                for chunk in self._split_text_for_slack(f"*Reasoning:*\n{thinking_formatted}"):
                    add_block({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": chunk}
                    })
            elif block_type == "tool":
                tool_info = content
                header = f"*{tool_info['name']}*"
                add_block({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"_{header}_"}
                })
                operations_text = '\n'.join([op.strip() for op in tool_info.get('operations', []) if op])
                if tool_info.get('operations') and tool_info.get('status') != 'completed':
                    operations_text += "\n..."
                for chunk in self._split_text_for_slack(operations_text):
                    add_block({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"_{chunk}_"}
                    })
            add_block({"type": "divider"})

        # Remove trailing divider from last page
        if pages and pages[-1] and pages[-1][-1].get("type") == "divider":
            pages[-1].pop()

        return pages

    def _build_modal_view(self, page_blocks: list, page_index: int, total_pages: int, message_ts: str) -> dict:
        """Construct a modal view with navigation for the given page of blocks."""
        nav_elements = []
        if total_pages > 1:
            nav_elements = [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Prev"},
                    "action_id": "flow_prev_page"
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": f"Page {page_index+1}/{total_pages}"},
                    "action_id": "noop_page_display",
                    "value": "noop",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Next"},
                    "action_id": "flow_next_page"
                }
            ]

        blocks = list(page_blocks)
        if nav_elements:
            blocks.append({
                "type": "actions",
                "elements": nav_elements
            })

        return {
            "type": "modal",
            "title": {"type": "plain_text", "text": "Execution Details"},
            "blocks": blocks,
            "close": {"type": "plain_text", "text": "Close"},
            "private_metadata": json.dumps({
                "message_ts": message_ts,
                "page": page_index,
                "total_pages": total_pages
            })
        }

    async def _update_execution_modal_page(self, client, body, message_ts: str, page_index: int) -> None:
        """Update the modal view to a different page using views.update."""
        cached_entry = self.execution_details_cache.get(message_ts)
        if not cached_entry:
            logger.error(f"No execution details found for message_ts: {message_ts}")
            return
        
        timestamp, execution_details = cached_entry
        pages = self._build_execution_pages(execution_details)
        total_pages = len(pages) if pages else 1
        if page_index >= total_pages:
            page_index = total_pages - 1
        modal_view = self._build_modal_view(pages[page_index] if pages else [], page_index, total_pages, message_ts)
        await client.views_update(
            view_id=body["view"]["id"],
            view=modal_view
        )
    
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
            
            # Clean message (resolve mentions to readable names)
            message_text, mention_context = await self._parse_slack_mentions(message_text)
            
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
            
            # Process asynchronously - no status messages
            asyncio.create_task(self._process_and_respond(
                user_id, channel_id, message_text, thread_ts, mention_context
            ))

            logger.info(f"SLACK-TIMING: _handle_message dispatched in {int((time.time()-t_start)*1000)} ms for user={user_id}")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_and_respond(self, user_id: str, channel_id: str, message_text: str, thread_ts: str, mention_context: dict):
        """Process message and send response with execution streaming"""
        streaming_handler = None
        try:
            t_overall = time.time()
            # Create streaming handler for this message
            streaming_handler = SlackStreamingHandler(self.app.client, channel_id, thread_ts)
            
            # Start streaming display
            t_stream_start = time.time()
            await streaming_handler.start_streaming()
            logger.info(f"SLACK-TIMING: start_streaming took {int((time.time()-t_stream_start)*1000)} ms user={user_id}")
            
            # Create simple streaming callback
            async def slack_streaming_callback(content: str, content_type: str):
                if content_type == "thinking" and content.strip():
                    await streaming_handler.update_thinking(content.strip())
                elif content_type == "tool_start":
                    await streaming_handler.start_tool(content)
                elif content_type == "tool_details":
                    await streaming_handler.add_tool_operation(content)
                elif content_type == "operation":
                    await streaming_handler.add_tool_operation(content)

                elif content_type == "result":
                    await streaming_handler.add_tool_operation(content)
                elif content_type == "result_detail":
                    await streaming_handler.add_tool_operation(content) 
                elif content_type == "error":
                    await streaming_handler.add_tool_operation(content)
                elif content_type == "tool_result":
                    # Extract and summarize the result
                    tool_name, result_preview = content.split(":", 1) if ":" in content else (content, "")
                    await streaming_handler.complete_tool(result_preview.strip())

            
            # Process through agent with streaming
            import datetime
            from zoneinfo import ZoneInfo
            
            # Get user's full information including name, role, and timezone
            user_info = await self._get_user_info(user_id)
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
            
            # Get response text from Claude (not status message)
            response_text = response.get("response", "No response generated")
            
            # Convert mentions back to Slack format for proper linking
            response_text = await self._resolve_and_format_mentions(response_text, mention_context)
            
            # Replace thinking message with final response
            t_finish = time.time()
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
    
    async def _parse_slack_mentions(self, text: str) -> tuple[str, dict]:
        """Parse incoming Slack message to clean text and extract mentions.
        
        Returns:
            tuple: (cleaned_text, mention_context)
            mention_context: dict mapping readable names back to Slack IDs
        """
        import re
        
        logger.info(f"Raw message text: {repr(text)}")
        
        # Store mappings for reverse conversion
        mention_context = {
            'users': {},  # name -> user_id
            'channels': {}  # name -> channel_id
        }
        
        # Debug: Find ALL potential mentions first 
        all_mentions = re.findall(r'<[^>]+>', text)
        logger.info(f"Found all <> patterns: {all_mentions}")
        
        # Find all user mentions - more flexible pattern to handle different formats
        # Slack can send <@U123456789>, <@WLZK08431>, or other variants - not all start with U!
        user_mentions = re.findall(r'<@([A-Z0-9]+)(?:\|([^>]+))?>', text)
        logger.info(f"Found user mentions: {user_mentions}")
        
        # Additional debug: try broader pattern
        broad_user_mentions = re.findall(r'<@([^>|]+)(?:\|([^>]+))?>', text)
        logger.info(f"Broader user mention pattern found: {broad_user_mentions}")
        
        for user_id, display_name in user_mentions:
            try:
                # If display name is provided in the mention, use it
                if display_name:
                    name = display_name
                    logger.info(f"Using provided display name for {user_id}: {name}")
                else:
                    # Get user info from Slack API
                    user_info = await self.app.client.users_info(user=user_id)
                    if user_info.get("ok"):
                        user = user_info.get("user", {})
                        # Use display name if available, otherwise real name, otherwise username
                        name = (user.get("profile", {}).get("display_name") or 
                               user.get("real_name") or 
                               user.get("name", user_id))
                        logger.info(f"Resolved user {user_id} to: {name}")
                    else:
                        name = user_id
                        logger.warning(f"Failed to get user info for {user_id}")
                
                # Store mapping for reverse conversion
                mention_context['users'][name] = user_id
                
                # Replace the mention with readable name
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{name}')
                
            except Exception as e:
                logger.error(f"Error resolving user {user_id}: {e}")
                # Fallback: just show @user_id
                name = user_id
                mention_context['users'][name] = user_id
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{user_id}')
        
        # Find all channel mentions <#C123456789|channel-name> or <#C123456789>
        channel_mentions = re.findall(r'<#(C[A-Z0-9]+)(?:\|([^>]+))?>', text)
        logger.info(f"Found channel mentions: {channel_mentions}")
        
        for channel_id, channel_name in channel_mentions:
            try:
                logger.info(f"Processing channel mention: channel_id={channel_id}, channel_name={channel_name}")
                
                if channel_name:
                    # Channel name is already provided in the mention
                    name = channel_name
                    logger.info(f"Using provided channel name for {channel_id}: {name}")
                else:
                    # Get channel info from Slack API
                    channel_info = await self.app.client.conversations_info(channel=channel_id)
                    if channel_info.get("ok"):
                        channel = channel_info.get("channel", {})
                        name = channel.get("name", channel_id)
                        logger.info(f"Resolved channel {channel_id} to: {name}")
                    else:
                        name = channel_id
                        logger.warning(f"Failed to get channel info for {channel_id}")
                
                # Store mapping for reverse conversion with the original channel name if available
                display_name = channel_name if channel_name else name
                mention_context['channels'][display_name] = channel_id
                
                # Replace the mention with readable name
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                logger.info(f"Trying to replace pattern: '{mention_pattern}' with '#{display_name}'")
                logger.info(f"Text before replacement: {repr(text)}")
                
                original_text = text
                text = text.replace(mention_pattern, f'#{display_name}')
                
                if text == original_text:
                    logger.warning(f"Channel mention replacement failed! Pattern '{mention_pattern}' not found in text")
                else:
                    logger.info(f"Successfully replaced channel mention")
                
                logger.info(f"Text after replacement: {repr(text)}")
                
            except Exception as e:
                logger.error(f"Error resolving channel {channel_id}: {e}")
                # Fallback: just show #channel_id
                fallback_name = channel_name if channel_name else channel_id
                mention_context['channels'][fallback_name] = channel_id
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                text = text.replace(mention_pattern, f'#{fallback_name}')
        
        # Handle special mentions like @everyone, @here, @channel
        text = re.sub(r'<!everyone>', '@everyone', text)
        text = re.sub(r'<!here>', '@here', text)
        text = re.sub(r'<!channel>', '@channel', text)
        
        # Remove any remaining bot mentions (our own bot)
        text = re.sub(r'<@[UW][A-Z0-9]{8,}>', '', text)
        
        final_text = ' '.join(text.split()).strip()
        logger.info(f"Final cleaned text: {repr(final_text)}")
        logger.info(f"Mention context: {mention_context}")
        
        return final_text, mention_context
    
    async def _resolve_and_format_mentions(self, text: str, mention_context: dict) -> str:
        """
        Resolve and format @user and #channel mentions in agent responses.
        
        This function first uses the initial `mention_context` as a cache.
        If a mention is not found in the cache, it will perform a live lookup.
        """
        import re
        
        # --- Step 1: Handle User Mentions ---
        
        # Find all potential user mentions, e.g., @andrei, @andrei.clodius, @andrei-clodius
        user_mention_pattern = r'@([a-zA-Z0-9._-]+)'
        
        # Create a list of all unique user names mentioned
        mentioned_user_names = set(re.findall(user_mention_pattern, text))
        
        # First, check the cache from the original message
        cached_users = mention_context.get('users', {})
        
        for name in mentioned_user_names:
            if name in cached_users:
                # This user was in the original message, use the cached ID
                user_id = cached_users[name]
                text = text.replace(f'@{name}', f'<@{user_id}>')
            else:
                # Live lookup needed for this user
                user_id = await self._find_user_id(name)
                if user_id:
                    text = text.replace(f'@{name}', f'<@{user_id}>')
                else:
                    logger.warning(f"Could not resolve user: @{name}")
        
        # --- Step 2: Handle Channel Mentions ---
        
        # Find all potential channel mentions, e.g., #general, #team-updates
        channel_mention_pattern = r'#([a-zA-Z0-9_-]+)'
        
        # Create a list of all unique channel names mentioned
        mentioned_channel_names = set(re.findall(channel_mention_pattern, text))
        
        # First, check the cache from the original message
        cached_channels = mention_context.get('channels', {})
        
        for name in mentioned_channel_names:
            if name in cached_channels:
                # This channel was in the original message, use the cached ID
                channel_id = cached_channels[name]
                text = text.replace(f'#{name}', f'<#{channel_id}|{name}>')
            else:
                # Live lookup needed for this channel
                channel_id = await self._find_channel_id(name)
                if channel_id:
                    text = text.replace(f'#{name}', f'<#{channel_id}|{name}>')
                else:
                    logger.warning(f"Could not resolve channel: #{name}")
        
        logger.info(f"Resolved mentions back to Slack format: {repr(text)}")
        return text
        
    async def _find_user_id(self, user_name: str) -> Optional[str]:
        """Find a user ID by name, using a cache to reduce API calls."""
        # Check cache first
        if user_name in self.user_cache:
            return self.user_cache[user_name]
        
        # If not in cache, fetch all users and populate cache
        try:
            logger.info("User cache empty. Fetching all users from Slack.")
            async for page in await self.app.client.users_list(limit=1000):
                for user in page.get("members", []):
                    if user.get("deleted") or user.get("is_bot"):
                        continue
                    
                    # Cache by display name, real name, and username
                    display_name = user.get("profile", {}).get("display_name", "").lower()
                    real_name = user.get("real_name", "").lower()
                    name = user.get("name", "").lower()
                    user_id = user.get("id")
                    
                    if display_name: self.user_cache[display_name] = user_id
                    if real_name: self.user_cache[real_name] = user_id
                    if name: self.user_cache[name] = user_id
            
            logger.info(f"User cache populated with {len(self.user_cache)} users.")
            
            # Try to find the user again from the populated cache
            return self.user_cache.get(user_name.lower())
            
        except Exception as e:
            logger.error(f"Error fetching users from Slack: {e}")
            return None

    async def _find_channel_id(self, channel_name: str) -> Optional[str]:
        """Find a channel ID by name, using a cache."""
        # Check cache first
        if channel_name in self.channel_cache:
            return self.channel_cache[channel_name]
            
        # If not in cache, fetch all public channels and populate cache
        try:
            logger.info("Channel cache empty. Fetching all public channels from Slack.")
            async for page in await self.app.client.conversations_list(
                limit=1000, 
                types="public_channel"
            ):
                for channel in page.get("channels", []):
                    name = channel.get("name", "").lower()
                    channel_id = channel.get("id")
                    if name and channel_id:
                        self.channel_cache[name] = channel_id
            
            logger.info(f"Channel cache populated with {len(self.channel_cache)} channels.")

            # Try to find the channel again from the populated cache
            return self.channel_cache.get(channel_name.lower())

        except Exception as e:
            logger.error(f"Error fetching channels from Slack: {e}")
            return None

    
    def _get_recent_insights_status(self, user_id: str) -> Optional[str]:
        """Get recent insights status for user from agent's user_buffers"""
        try:
            # Access the agent's user_buffers
            user_buffers = self.agent.user_buffers
            
            if user_id not in user_buffers:
                return None
            
            # Check for recent insights status
            recent_data = user_buffers[user_id].get('recent', {})
            insights_status = recent_data.get('insights_status')
            
            if not insights_status:
                return None
            
            # Check if the status is recent (within last 5 minutes)
            import time
            current_time = time.time()
            status_time = insights_status.get('timestamp', 0)
            
            # Only show status if it's from the last 5 minutes
            if current_time - status_time > 300:  # 5 minutes
                return None
            
            return insights_status.get('message', 'No new insights')
            
        except Exception as e:
            logger.error(f"Error getting insights status for user {user_id}: {e}")
            return None
    
    def get_fastapi_handler(self):
        """Get FastAPI handler for webhook integration"""
        return self.handler

    async def _cleanup_cache_periodically(self):
        """Background task to clean up stale cache entries."""
        while True:
            current_time = time.time()
            keys_to_delete = []
            for message_ts, (timestamp, _) in self.execution_details_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    keys_to_delete.append(message_ts)
                    logger.info(f"Cache entry for message_ts {message_ts} expired. Deleting.")
            
            for message_ts in keys_to_delete:
                del self.execution_details_cache[message_ts]
            
            logger.info(f"Cache cleanup complete. {len(keys_to_delete)} entries deleted.")
            await asyncio.sleep(self.cache_ttl) # Check every TTL seconds

# For FastAPI integration
def create_slack_app():
    """Create simplified Slack interface for FastAPI"""
    slack_interface = SlackInterface()
    return slack_interface.get_fastapi_handler()