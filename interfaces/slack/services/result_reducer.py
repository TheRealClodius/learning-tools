"""
Slack Result Reducer

Implements the progressive reduction pipeline for Slack search results:
Fetch → Filter → Group → Rank/Dedup → Summarize → Cite → (Optionally) Drill-down
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SlackResultReducer:
    """
    Handles progressive reduction of Slack search results
    """
    
    def __init__(self):
        self.result_cache = {}  # Cache for result sets
        
    async def reduce_results(self, 
                           results: Dict[str, Any],
                           reduction_level: str = "preview") -> Dict[str, Any]:
        """
        Progressive result reduction
        
        Args:
            results: Raw search results
            reduction_level: 
                - "preview": Just metadata and previews
                - "summary": Include computed summaries
                - "full": Full content with citations
        
        Returns:
            Reduced result set with appropriate detail level
        """
        # Generate result set ID
        result_set_id = f"rs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Basic structure
        reduced = {
            "result_set_id": result_set_id,
            "count": self._get_total_count(results),
            "facets": self._compute_facets(results),
            "page": {
                "items": self._reduce_items(results, reduction_level),
                "next_page_token": results.get("next_page_token")
            }
        }
        
        # Cache the result set
        self.result_cache[result_set_id] = {
            "raw_results": results,
            "reduced": reduced
        }
        
        return reduced
        
    def _get_total_count(self, results: Dict[str, Any]) -> int:
        """Extract total result count"""
        if "matches" in results:
            return len(results["matches"])
        return results.get("count", 0)
        
    def _compute_facets(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute facets from results
        - Channels
        - Dates
        - Authors
        - Types (messages, threads, etc)
        """
        facets = {
            "slack": {
                "channels": [],
                "dates": [],
                "authors": []
            }
        }
        
        # Channel facets
        channel_counts = {}
        date_counts = {}
        author_counts = {}
        
        for item in results.get("matches", []):
            # Channel counts
            channel = item.get("channel")
            if channel:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                
            # Date counts (group by day)
            ts = item.get("ts")
            if ts:
                day = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
                date_counts[day] = date_counts.get(day, 0) + 1
                
            # Author counts
            author = item.get("user")
            if author:
                author_counts[author] = author_counts.get(author, 0) + 1
        
        # Convert to sorted lists
        facets["slack"]["channels"] = [
            {"name": k, "n": v} 
            for k, v in sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        facets["slack"]["dates"] = [
            {"day": k, "n": v}
            for k, v in sorted(date_counts.items(), key=lambda x: x[0], reverse=True)
        ]
        
        facets["slack"]["authors"] = [
            {"id": k, "n": v}
            for k, v in sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return facets
        
    def _reduce_items(self, 
                     results: Dict[str, Any], 
                     reduction_level: str) -> List[Dict[str, Any]]:
        """
        Reduce individual items based on reduction level
        """
        reduced_items = []
        
        for item in results.get("matches", []):
            reduced_item = {
                "id": f"slack:ch/{item.get('channel')}/{item.get('ts')}",
                "thread_id": f"slack:th/{item.get('thread_ts')}" if item.get("thread_ts") else None,
                "ts": datetime.fromtimestamp(float(item.get("ts", 0))).isoformat(),
                "author": item.get("user"),
                "text_preview": self._create_preview(item.get("text", "")),
                "reactions": len(item.get("reactions", [])),
                "attachments": len(item.get("files", []))
            }
            
            # Add full content for "full" reduction level
            if reduction_level == "full":
                reduced_item["text"] = item.get("text")
                reduced_item["reactions_detail"] = item.get("reactions", [])
                reduced_item["files"] = item.get("files", [])
            
            # Add summary for "summary" level
            elif reduction_level == "summary":
                reduced_item["summary"] = self._create_summary(item)
                
            reduced_items.append(reduced_item)
            
        return reduced_items
        
    def _create_preview(self, text: str, max_length: int = 100) -> str:
        """Create a short preview of the text"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
        
    def _create_summary(self, item: Dict[str, Any]) -> str:
        """Create a summary of the item"""
        # This would be where we'd add LLM-based summarization
        # For now, just return a structured summary
        summary_parts = []
        
        if item.get("text"):
            summary_parts.append(f"Message: {self._create_preview(item['text'], 50)}")
            
        if item.get("reactions"):
            summary_parts.append(f"Reactions: {len(item['reactions'])}")
            
        if item.get("files"):
            summary_parts.append(f"Attachments: {len(item['files'])}")
            
        if item.get("thread_ts"):
            summary_parts.append("Part of thread")
            
        return " | ".join(summary_parts)
        
    async def drill_down(self, 
                        result_set_id: str,
                        item_id: str) -> Optional[Dict[str, Any]]:
        """
        Drill down into a specific item
        - Fetches full content
        - Gets thread context if available
        - Includes all metadata
        """
        if result_set_id not in self.result_cache:
            return None
            
        cached = self.result_cache[result_set_id]
        
        # Find the item in the raw results
        for item in cached["raw_results"].get("matches", []):
            if f"slack:ch/{item.get('channel')}/{item.get('ts')}" == item_id:
                return {
                    "id": item_id,
                    "type": "message",
                    "content": item.get("text"),
                    "thread_ts": item.get("thread_ts"),
                    "user": item.get("user"),
                    "ts": item.get("ts"),
                    "channel": item.get("channel"),
                    "reactions": item.get("reactions", []),
                    "files": item.get("files", [])
                }
                
        return None
