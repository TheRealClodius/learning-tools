"""
Slack Streaming Handler

Manages real-time message updates during agent execution with a clean architecture:
- Displays progressive thinking content from Claude
- Shows tool execution with pre-generated Gemini summaries from ClientAgent  
- Provides interactive execution details modal support
- Focuses purely on UI formatting (no LLM logic)
"""

import time
import logging
from typing import Dict, Any, Optional, List

from ..formatters.markdown_parser import MarkdownToSlackParser

logger = logging.getLogger(__name__)


class SlackStreamingHandler:
    """
    Handles progressive message updates for agent execution streaming.
    
    Architecture:
    - Reasoning blocks: Thinking content from Claude
    - Tool blocks: Pre-generated summaries from ClientAgent (Gemini)
    - Pure UI formatting: No LLM logic in this class
    """
    
    def __init__(self, app_client, channel_id: str, thread_ts: str, user_id: str = None):
        self.app_client = app_client
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.user_id = user_id  # Store user_id for database storage
        self.message_ts: Optional[str] = None
        self.content_blocks = []  # Chronological list of thinking and tool blocks
        self.current_tool_block = None  # Currently active tool block
        self.all_thinking = []  # All thinking content accumulated
        self.execution_summary = []  # Store execution details for modal
        # Summarization is now handled by ClientAgent
        
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
        
    async def update_thinking(self, thinking_block: str):
        """Add thinking content block and update display efficiently"""
        if not self.message_ts or not thinking_block.strip():
            return
            
        # Split thinking block into lines for display formatting
        lines = thinking_block.strip().split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        # Add all lines from this thinking block
        self.all_thinking.extend(clean_lines)
        
        # DEBUG: Log thinking collection
        logger.info(f"DEBUG-THINKING: Added {len(clean_lines)} thinking lines, total entries: {len(self.all_thinking)}")
        
        await self._update_display()
    
    async def start_tool(self, tool_name: str, tool_args: Dict = None):
        """Start a new tool execution block"""
        # End any current thinking block by moving it to completed blocks
        if self.all_thinking:
            thinking_text = "\n".join(self.all_thinking)
            self.execution_summary.append(("thinking", thinking_text))
            logger.info(f"DEBUG-THINKING: Tool start - stored {len(self.all_thinking)} thinking lines in execution_summary")
            self.all_thinking = []
        
        # Store any previous tool in execution summary
        if self.current_tool_block:
            self.execution_summary.append(("tool", self.current_tool_block.copy()))
        
        # DEBUG: Log tool start
        logger.debug(f"DEBUG: Starting tool {tool_name}, execution_summary now has {len(self.execution_summary)} items")
        
        # Tool args are now handled by ClientAgent for summarization
        
        # Start new tool block
        self.current_tool_block = {
            "name": tool_name,
            "operations": [],
            "status": "running"
        }
        await self._update_display()
        
    async def append_to_current_tool(self, chunk: str):
        """Append a chunk to the current tool's operations (for streaming summaries)"""
        if self.current_tool_block:
            # If this is the first chunk, replace any existing operations
            if len(self.current_tool_block["operations"]) == 0:
                self.current_tool_block["operations"] = [chunk]
            else:
                # Append to the last operation (streaming continuation)
                self.current_tool_block["operations"][-1] += chunk
            await self._update_display()
        
    async def complete_tool(self, result_summary: str):
        """Complete current tool with pre-generated summary from ClientAgent"""
        if not self.current_tool_block:
            return
            
        self.current_tool_block["status"] = "completed"
        
        if result_summary:
            # All tools now get complete summaries from ClientAgent (Gemini)
            # Just use the summary directly - no need for complex logic
            self.current_tool_block["operations"] = [result_summary]
        
        # Store completed tool for display and modal
        completed_tool = self.current_tool_block.copy()
        self.content_blocks.append(("tool", completed_tool))
        self.execution_summary.append(("tool", completed_tool))
        
        # Reset for next tool
        self.current_tool_block = None
        await self._update_display()
    
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
                # Ensure text is never empty (Slack requires at least 1 character)
                if not thinking_formatted:
                    thinking_formatted = "_Received an empty response_"
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
                # Tool block: Just italicize the Gemini-generated summary content
                tool_info = content
                tool_lines = []
                
                # Add operations directly (they're already Gemini-formatted summaries)
                for operation in tool_info['operations']:
                    tool_lines.append(operation.strip())
                
                # Format each line with italics (no bold names, no status indicators)
                tool_formatted = '\n'.join([f"_{line}_" for line in tool_lines if line])
                # Ensure text is never empty (Slack requires at least 1 character)
                if not tool_formatted:
                    tool_formatted = "_Received an empty response_"
                
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": tool_formatted
                        }
                    ]
                })
        
        # Add current thinking block if we have ongoing thinking
        if self.all_thinking:
            thinking_lines = self.all_thinking
            thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
            # Ensure text is never empty (Slack requires at least 1 character)
            if not thinking_formatted:
                thinking_formatted = "_Received an empty response_"
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
            tool_lines = []
            
            # Add operations directly (they're Gemini summaries or start messages)
            for operation in tool_info['operations']:
                tool_lines.append(operation.strip())
            
            # Format each line with italics (no bold names, no status indicators)
            tool_formatted = '\n'.join([f"_{line}_" for line in tool_lines if line])
            # Ensure text is never empty (Slack requires at least 1 character)
            if not tool_formatted:
                tool_formatted = "_Received an empty response_"
            
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
            logger.info(f"DEBUG-THINKING: Stored {len(self.all_thinking)} thinking lines in execution_summary")
        
        # Store any current tool in execution summary
        if self.current_tool_block:
            self.execution_summary.append(("tool", self.current_tool_block.copy()))
        
        # DEBUG: Log execution summary state
        logger.info(f"DEBUG: finish_with_response - execution_summary has {len(self.execution_summary)} items")
        if self.execution_summary:
            logger.info(f"DEBUG: execution_summary types: {[item[0] for item in self.execution_summary[:5]]}")
        
        # IMPROVED: Store execution details in database for persistent access
        if slack_interface and self.execution_summary:
            try:
                # Store in database via cache service
                await slack_interface.cache_service.store_execution_details(
                    self.message_ts, self.channel_id, self.user_id or 'unknown', self.execution_summary.copy()
                )
            except Exception as e:
                logger.error(f"Failed to store execution details in database: {e}")
        else:
            logger.warning(f"DEBUG: Not storing execution details - slack_interface={slack_interface is not None}, execution_summary={len(self.execution_summary) if self.execution_summary else 0}")
        
        # Parse the response as markdown and convert to Slack blocks
        try:
            response_blocks = MarkdownToSlackParser.parse_to_blocks(final_response)
            if not response_blocks:
                response_blocks = [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": final_response
                    }
                }]
            
            # Add discrete execution details link if we have execution data
            if self.execution_summary:
                logger.info(f"DEBUG: Adding view flow button for message_ts={self.message_ts}")
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
            else:
                logger.info("DEBUG: No execution_summary, not adding view flow button")
            
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
