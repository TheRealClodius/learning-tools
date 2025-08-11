"""
Slack Modal Handler

Manages modal interactions for execution details display with pagination.
Handles modal creation, page navigation, and size optimization for Slack's limits.
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SlackModalHandler:
    """Handles modal interactions for execution details display"""
    
    def __init__(self, cache_service):
        self.cache_service = cache_service
        self.MAX_MODAL_SIZE = 500000  # 500KB conservative limit for Slack modals
        self.MAX_BLOCKS_PER_PAGE = 95  # Conservative block limit per page
    
    async def show_execution_details_modal(self, body: Dict[str, Any], client) -> None:
        """Show execution details in a modal - optimized for trigger_id timeout"""
        try:
            # DEBUG: Log the entire body to understand what we're receiving
            logger.info(f"DEBUG: Received body for modal: {json.dumps(body, default=str)[:500]}")
            
            # FAST VALIDATION: Check critical fields immediately
            trigger_id = body.get("trigger_id")
            if not trigger_id:
                logger.error("No trigger_id found in body - cannot open modal")
                logger.error(f"DEBUG: Body keys available: {list(body.keys())}")
                raise ValueError("Missing trigger_id")
            
            user_id = body.get("user", {}).get("id")
            message_ts = body["actions"][0]["value"]
            
            logger.info(f"Fast modal open for user {user_id}, message_ts: {message_ts}, trigger_id: {trigger_id[:8]}...")
            
            # FAST DATABASE LOOKUP: Get execution details from database immediately
            cached_entry = await self.cache_service.get_execution_details(message_ts)
            
            if not cached_entry:
                # Error already logged in cache_service
                raise ValueError(f"Execution details not found for message_ts: {message_ts}")
            
            # Extract execution details from cache
            timestamp, execution_details = cached_entry
            logger.info(f"Found {len(execution_details)} execution detail blocks")
            
            # Build paginated modal pages with full content (no truncation) while respecting Slack limits
            pages = self._build_execution_pages(execution_details)
            total_pages = len(pages) if pages else 1
            page_index = 0
            logger.info(f"Built {total_pages} pages for modal")
            
            # DEBUG: Log first page content
            if pages:
                logger.info(f"DEBUG: First page has {len(pages[0])} blocks")
                logger.info(f"DEBUG: First block type: {pages[0][0].get('type') if pages[0] else 'empty'}")
            
            modal_view = self._build_modal_view(pages[page_index] if pages else [], page_index, total_pages, message_ts)
            
            # DEBUG: Log modal structure
            logger.info(f"DEBUG: Modal view keys: {list(modal_view.keys())}")
            logger.info(f"DEBUG: Modal has {len(modal_view.get('blocks', []))} blocks")
            
            # CRITICAL SIZE CHECK: Validate modal size before sending to avoid silent failures
            modal_json = json.dumps(modal_view)
            modal_size = len(modal_json)
            logger.info(f"Modal JSON size: {modal_size} bytes")
            
            if modal_size > self.MAX_MODAL_SIZE:
                logger.error(f"Modal size ({modal_size} bytes) exceeds safe limit ({self.MAX_MODAL_SIZE} bytes)")
                raise ValueError(f"Modal too large: {modal_size} bytes > {self.MAX_MODAL_SIZE} bytes")
            
            # OPTIMIZED MODAL OPENING: Open the modal with minimal overhead
            logger.info(f"Opening modal with trigger_id: {trigger_id[:8]}... (size: {modal_size} bytes)")
            
            # DEBUG: Log before API call
            logger.info("DEBUG: About to call client.views_open...")
            
            response = await client.views_open(
                trigger_id=trigger_id,
                view=modal_view
            )
            
            # DEBUG: Log full response
            logger.info(f"DEBUG: views_open response: {json.dumps(response, default=str)[:500]}")
            
            # IMPROVED ERROR HANDLING: Log both success and failure details
            if response.get("ok"):
                view_id = response.get('view', {}).get('id', 'unknown')
                logger.info(f"✅ Modal opened successfully! view_id: {view_id}")
            else:
                error_msg = response.get('error', 'unknown_error')
                logger.error(f"❌ Slack API returned not ok: {error_msg}, full response: {response}")
                raise Exception(f"Slack API error: {error_msg}")
            
        except Exception as e:
            logger.error(f"Error showing execution details modal: {e}", exc_info=True)
            
            # FAST ERROR CLASSIFICATION: Determine user-friendly error message
            error_str = str(e).lower()
            if "not found" in error_str:
                user_message = "Execution details not found or expired."
            elif "outdated" in error_str or "stale" in error_str:
                user_message = "Execution details are outdated. Please try again."
            elif "too large" in error_str or "size" in error_str:
                user_message = "Execution details are too large to display. Please check logs."
            elif "trigger_id" in error_str:
                user_message = "Request expired. Please click the button again."
            else:
                user_message = "Unable to display execution details. Please try again."
            
            # FAST USER NOTIFICATION: Send ephemeral message efficiently
            try:
                channel_id = body.get("channel", {}).get("id") or body.get("container", {}).get("channel_id")
                user_id = body.get("user", {}).get("id")
                if channel_id and user_id:
                    # Use specific error message for better user experience
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=user_message
                    )
                    logger.info(f"Sent error message to user {user_id}: {user_message}")
            except Exception as ephemeral_error:
                logger.warning(f"Failed to send ephemeral error message: {ephemeral_error}")
            
            # RE-RAISE: Let the retry logic in the caller handle this
            raise
    
    async def update_execution_modal_page(self, client, body: Dict[str, Any], message_ts: str, page_index: int) -> None:
        """Update the modal view to a different page using views.update"""
        cached_entry = await self.cache_service.get_execution_details(message_ts)
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
    
    def _split_text_for_slack(self, text: str, max_len: int = 2900) -> List[str]:
        """Split large text into multiple chunks that fit Slack text object limits"""
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

    def _build_execution_pages(self, execution_details: List, max_blocks_per_page: int = None) -> List[List[Dict[str, Any]]]:
        """Build pages of Slack blocks for the entire execution, no truncation (paginate instead)"""
        if max_blocks_per_page is None:
            max_blocks_per_page = self.MAX_BLOCKS_PER_PAGE
            
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
                
                # If no thinking content, show a helpful message or skip
                if not thinking_formatted:
                    thinking_formatted = "_Agent processed request without detailed reasoning_"
                
                for chunk in self._split_text_for_slack(f"*Reasoning:*\n{thinking_formatted}"):
                    add_block({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": chunk}
                    })
            elif block_type == "tool":
                tool_info = content
                # Operations are Gemini-generated summaries - just display them
                operations_text = '\n'.join([op.strip() for op in tool_info.get('operations', []) if op])
                
                # If no operations text, try to show something more helpful
                if not operations_text:
                    tool_name = tool_info.get('name', 'Unknown tool')
                    tool_status = tool_info.get('status', 'unknown')
                    if tool_status == 'completed':
                        operations_text = f"Tool '{tool_name}' completed successfully"
                    else:
                        operations_text = f"Tool '{tool_name}' executed ({tool_status})"
                
                # Create blocks for the tool information
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

    def _build_modal_view(self, page_blocks: List[Dict[str, Any]], page_index: int, total_pages: int, message_ts: str) -> Dict[str, Any]:
        """Construct a modal view with navigation for the given page of blocks"""
        # DEBUG: Log input parameters
        logger.info(f"DEBUG: _build_modal_view called with {len(page_blocks)} blocks, page {page_index+1}/{total_pages}")
        
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

        blocks = list(page_blocks) if page_blocks else []
        
        # SAFEGUARD: Ensure we always have at least one block
        if not blocks:
            logger.warning("DEBUG: No blocks provided to modal, adding default message")
            blocks = [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_No execution details available for this interaction._"
                }
            }]
        
        if nav_elements:
            blocks.append({
                "type": "actions",
                "elements": nav_elements
            })
        
        # DEBUG: Log final block count
        logger.info(f"DEBUG: Modal will have {len(blocks)} blocks total")

        modal = {
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
        
        # DEBUG: Validate modal structure
        logger.info(f"DEBUG: Modal structure validated - has type: {modal.get('type')}, title: {modal.get('title') is not None}, blocks: {len(modal.get('blocks', []))}")
        
        return modal
