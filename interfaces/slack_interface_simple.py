import os
import logging
import asyncio
import re
from typing import Dict, Any, Optional, List
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

# Import agent components
from agents.client_agent import ClientAgent

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
        self.thinking_steps = []
        self.current_tool = None
        self.tools_executed = []
        self.operation_log = []  # For detailed operations
        self.current_status = None
        
    async def start_streaming(self):
        """Post initial thinking message"""
        response = await self.app_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "🤔 *Thinking...*"
                    }
                }
            ],
            text="🤔 Thinking...",
            thread_ts=self.thread_ts
        )
        self.message_ts = response["ts"]
        
    async def update_thinking(self, thinking_line: str):
        """Add a new thinking step and update message"""
        if not self.message_ts:
            return
            
        # Only add non-empty thinking lines
        if thinking_line.strip():
            self.thinking_steps.append(thinking_line.strip())
        
        await self._update_display()
        
    async def add_status(self, status: str):
        """Add a status update"""
        self.current_status = status
        await self._update_display()
        
    async def add_operation_detail(self, detail: str, emoji: str):
        """Add an operation detail with emoji"""
        self.operation_log.append(f"{emoji} {detail}")
        # Keep only last 5 operations to avoid message bloat
        if len(self.operation_log) > 5:
            self.operation_log = self.operation_log[-5:]
        await self._update_display()
        
    async def _update_display(self):
        """Update the streaming message display with complete transparency"""
        if not self.message_ts:
            return
            
        blocks = []
        
        # Main status with current operation
        main_text = "🤔 *Agent Working...*"
        if self.current_status:
            main_text = f"🤖 *{self.current_status}*"
        elif self.current_tool:
            main_text = f"🔧 *Executing: {self.current_tool}*"
            
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": main_text
            }
        })
        
        # Recent thinking steps
        if self.thinking_steps:
            recent_thinking = self.thinking_steps[-2:]  # Last 2 thinking steps
            thinking_text = "\n".join([f"💭 {step}" for step in recent_thinking])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Reasoning:*\n" + thinking_text
                }
            })
        
        # Operation details (registry searches, memory ops, etc.) - using smaller context blocks
        if self.operation_log:
            # Add a label for operations
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Operations:*"
                }
            })
            
            # Add operations in smaller context blocks (2 operations per block for readability)
            recent_operations = self.operation_log[-4:]  # Last 4 operations
            for i in range(0, len(recent_operations), 2):
                operation_batch = recent_operations[i:i+2]
                context_elements = []
                for op in operation_batch:
                    context_elements.append({
                        "type": "mrkdwn",
                        "text": op
                    })
                
                blocks.append({
                    "type": "context",
                    "elements": context_elements
                })
        
        # Tools completed summary - using smaller context blocks
        if self.tools_executed:
            completed_tools = ", ".join(self.tools_executed[-3:])  # Last 3 tools
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"✅ *Completed:* {completed_tools}"
                    }
                ]
            })
        
        await self.app_client.chat_update(
            channel=self.channel_id,
            ts=self.message_ts,
            blocks=blocks,
            text="🤖 Agent working..."
        )
        
    async def set_executing_tool(self, tool_name: str):
        """Update to show tool being executed"""
        self.current_tool = tool_name
        await self._update_display()  # Trigger update without new thinking
        
    async def tool_completed(self, tool_name: str, result_preview: str):
        """Mark tool as completed"""
        self.current_tool = None
        self.tools_executed.append(tool_name)
        
        # Brief update to show completion
        await self.update_thinking(f"Got result from {tool_name}: {result_preview[:50]}...")
        
    async def finish_with_response(self, final_response: str):
        """Replace thinking message with final response using proper markdown parsing"""
        if not self.message_ts:
            return
        
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
        
        # Setup handlers
        self._setup_handlers()
        
        # Create FastAPI handler
        self.handler = AsyncSlackRequestHandler(self.app)
    
    def _setup_handlers(self):
        """Setup basic Slack event handlers"""
        
        @self.app.event("message")
        async def handle_message(event, say, logger):
            await self._handle_message(event, say, logger)
        
        @self.app.event("app_mention")
        async def handle_mention(event, say, logger):
            await self._handle_message(event, say, logger)
    
    async def _handle_message(self, event: Dict[str, Any], say, logger):
        """Handle incoming messages - simplified version"""
        try:
            # Skip bot messages
            if event.get("bot_id"):
                return
            
            # Extract basics
            user_id = event.get("user")
            channel_id = event.get("channel")
            message_text = event.get("text", "").strip()
            thread_ts = event.get("thread_ts") or event.get("ts")
            
            # Clean message (resolve mentions to readable names)
            message_text, mention_mappings = await self._clean_message(message_text)
            
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
                user_id, channel_id, message_text, thread_ts, mention_mappings
            ))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_and_respond(self, user_id: str, channel_id: str, message_text: str, thread_ts: str, mention_mappings: dict):
        """Process message and send response with execution streaming"""
        streaming_handler = None
        try:
            # Create streaming handler for this message
            streaming_handler = SlackStreamingHandler(self.app.client, channel_id, thread_ts)
            
            # Start streaming display
            await streaming_handler.start_streaming()
            
            # Create comprehensive streaming callback
            async def slack_streaming_callback(content: str, content_type: str):
                if content_type == "thinking" and content.strip():
                    await streaming_handler.update_thinking(content.strip())
                elif content_type == "status":
                    await streaming_handler.add_status(content)
                elif content_type == "tool_start":
                    await streaming_handler.set_executing_tool(content)
                elif content_type == "tool_details":
                    await streaming_handler.add_operation_detail(content, "🔧")
                elif content_type == "operation":
                    await streaming_handler.add_operation_detail(content, "⚙️")
                elif content_type == "memory":
                    await streaming_handler.add_operation_detail(content, "💾")
                elif content_type == "result":
                    await streaming_handler.add_operation_detail(content, "✅")
                elif content_type == "result_detail":
                    await streaming_handler.add_operation_detail(content, "📊")
                elif content_type == "error":
                    await streaming_handler.add_operation_detail(content, "❌")
                elif content_type == "tool_result":
                    tool_name, result_preview = content.split(":", 1) if ":" in content else (content, "")
                    await streaming_handler.tool_completed(tool_name.strip(), result_preview.strip())
            
            # Process through agent with streaming
            context = {
                "platform": "slack",
                "user_id": user_id,
                "channel_id": channel_id,
                "thread_ts": thread_ts
            }
            
            response = await self.agent.process_request(
                message_text, 
                context=context,
                streaming_callback=slack_streaming_callback
            )
            
            # Get response text from Claude (not status message)
            response_text = response.get("response", "No response generated")
            
            # Convert mentions back to Slack format for proper linking
            response_text = self._convert_mentions_to_slack(response_text, mention_mappings)
            
            # Replace thinking message with final response
            await streaming_handler.finish_with_response(response_text)
            
            logger.info(f"Sent response to {user_id}: {response_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # If streaming was started, try to show error in that message
            if streaming_handler and streaming_handler.message_ts:
                try:
                    error_text = f"❌ **Error Processing Request**\n\nSorry, I encountered an error while processing your message:\n\n`{str(e)}`"
                    await streaming_handler.finish_with_response(error_text)
                except:
                    pass  # Fallback failed, just log
    
    async def _clean_message(self, text: str) -> tuple[str, dict]:
        """Clean message text and resolve mentions to readable names
        
        Returns:
            tuple: (cleaned_text, mention_mappings)
            mention_mappings: dict mapping readable names back to Slack IDs
        """
        import re
        
        logger.info(f"Raw message text: {repr(text)}")
        
        # Store mappings for reverse conversion
        mention_mappings = {
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
                mention_mappings['users'][name] = user_id
                
                # Replace the mention with readable name
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{name}')
                
            except Exception as e:
                logger.error(f"Error resolving user {user_id}: {e}")
                # Fallback: just show @user_id
                name = user_id
                mention_mappings['users'][name] = user_id
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{user_id}')
        
        # Find all channel mentions <#C123456789|channel-name> or <#C123456789>
        channel_mentions = re.findall(r'<#(C[A-Z0-9]+)(?:\|([^>]+))?>', text)
        logger.info(f"Found channel mentions: {channel_mentions}")
        
        for channel_id, channel_name in channel_mentions:
            try:
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
                
                # Store mapping for reverse conversion
                mention_mappings['channels'][name] = channel_id
                
                # Replace the mention with readable name
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                text = text.replace(mention_pattern, f'#{name}')
                
            except Exception as e:
                logger.error(f"Error resolving channel {channel_id}: {e}")
                # Fallback: just show #channel_id
                name = channel_id
                mention_mappings['channels'][name] = channel_id
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                text = text.replace(mention_pattern, f'#{channel_id}')
        
        # Handle special mentions like @everyone, @here, @channel
        text = re.sub(r'<!everyone>', '@everyone', text)
        text = re.sub(r'<!here>', '@here', text)
        text = re.sub(r'<!channel>', '@channel', text)
        
        # Remove any remaining bot mentions (our own bot)
        text = re.sub(r'<@[UW][A-Z0-9]{8,}>', '', text)
        
        final_text = ' '.join(text.split()).strip()
        logger.info(f"Final cleaned text: {repr(final_text)}")
        logger.info(f"Mention mappings: {mention_mappings}")
        
        return final_text, mention_mappings
    
    def _convert_mentions_to_slack(self, text: str, mention_mappings: dict) -> str:
        """Convert readable mentions back to Slack mention format for responses"""
        import re
        
        # Convert user mentions: @Name -> <@U123456789>
        for name, user_id in mention_mappings.get('users', {}).items():
            # Use word boundaries to avoid partial matches
            pattern = f'@{re.escape(name)}'
            text = re.sub(pattern, f'<@{user_id}>', text)
        
        # Convert channel mentions: #channel-name -> <#C123456789>
        for name, channel_id in mention_mappings.get('channels', {}).items():
            pattern = f'#{re.escape(name)}'
            text = re.sub(pattern, f'<#{channel_id}>', text)
        
        logger.info(f"Converted mentions back to Slack format: {repr(text)}")
        return text
    
    def get_fastapi_handler(self):
        """Get FastAPI handler for webhook integration"""
        return self.handler

# For FastAPI integration
def create_slack_app():
    """Create simplified Slack interface for FastAPI"""
    slack_interface = SlackInterface()
    return slack_interface.get_fastapi_handler()