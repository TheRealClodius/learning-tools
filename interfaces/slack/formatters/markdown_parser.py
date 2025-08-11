"""
Markdown to Slack Block Kit Parser

Converts markdown content to Slack's Block Kit format for rich message display.
Handles headers, code blocks, lists, links, and text formatting with proper
splitting for Slack's character limits.
"""

import re
from typing import Dict, Any, List


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
