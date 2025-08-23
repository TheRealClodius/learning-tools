"""
Progressive Reducer for Slack Search Results

Implements a sophisticated multi-stage reduction pipeline:
Stage A: Upstream Filtering
Stage B: Grouping
Stage C: Rank + Diversify
Stage D: Map-Reduce Summarization
Stage E: Drill-down
"""

import time
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict

from .token_manager import TokenManager
from .guard_rails import GuardRails
from .summarizer import ProgressiveSummarizer
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

@dataclass
class ReductionConfig:
    """Configuration for the reduction pipeline"""
    page_size: int = 50
    max_preview_length: int = 200
    cluster_summary_tokens: int = 200
    global_summary_tokens: int = 400
    max_evidence_items: int = 10
    mmr_lambda: float = 0.7
    mmr_items_per_cluster: int = 5

@dataclass
class ClusterFeatures:
    """Features computed for a message cluster"""
    cluster_id: str
    size: int
    time_span: Dict[str, str]
    engagement: Dict[str, int]
    participants: List[str]
    channel: str

class ProgressiveReducer:
    """
    Implements the progressive reduction pipeline for Slack search results
    """
    
    def __init__(self,
                 config: Optional[ReductionConfig] = None,
                 run_id: Optional[str] = None,
                 token_manager: Optional[TokenManager] = None,
                 progress_tracker: Optional[ProgressTracker] = None):
        self.config = config or ReductionConfig()
        self.result_cache: Dict[str, Any] = {}
        self.token_manager = token_manager or TokenManager()
        self.guard_rails = GuardRails()
        self.summarizer = ProgressiveSummarizer(model=self.token_manager.model)
        self.run_id = run_id or f"run_{int(time.time())}"
        self.current_query: Optional[str] = None
        self.progress_tracker = progress_tracker
        
    async def reduce_results(self,
                           results: Dict[str, Any],
                           query: str,
                           filters: Optional[Dict[str, Any]] = None,
                           page: int = 1) -> Dict[str, Any]:
        """
        Main entry point for the reduction pipeline
        
        Args:
            results: Raw search results
            query: Original search query
            filters: Applied filters (time, channel, etc)
            page: Current page number for pagination
            
        Returns:
            Reduced and structured results
        """
        # Store query for use in summarization
        self.current_query = query
        # Check for result explosion
        suggested_filters = self.guard_rails.check_result_explosion(results, page)
        if suggested_filters:
            # Add filter suggestions to results
            results["suggested_filters"] = suggested_filters
        # Process with locking to prevent multi-agent collisions
        result_set_id = f"rs_{int(time.time())}"
        
        async def process_pipeline():
            # Stage A: Upstream Filtering
            filtered_results = self._apply_filters(results, filters)
            
            # Deduplicate results
            filtered_results = self.guard_rails.deduplicate_results(filtered_results)
            
            # Stage B: Grouping
            clusters = self._group_results(filtered_results)
            
            # Check token budget and compress if needed
            ctx_estimate = sum(
                self.token_manager.count_tokens(str(cluster))
                for cluster in clusters.values()
            )
            clusters = self.guard_rails.check_token_budget(
                ctx_estimate,
                self.token_manager.limits.TOTAL_CLUSTERS,
                clusters
            )
            
            # Stage C: Rank + Diversify (always rank to avoid ordering bias)
            ranked = self._rank_and_diversify(clusters, query)
            
            # Stage D: Map-Reduce Summarization
            summarized = await self._map_reduce_summarize(ranked)
            return {
                "summarized": summarized,
                "filtered": filtered_results,
                "clusters": clusters
            }
            
        # Process with locking
        pipeline_result = await self.guard_rails.process_with_lock(
            result_set_id=result_set_id,
            run_id=self.run_id,
            process_fn=process_pipeline
        )
        
        if pipeline_result is None:
            logger.error(f"Failed to acquire lock for result_set_id={result_set_id}")
            return {
                "success": False,
                "message": "Processing lock acquisition failed"
            }
            
        summarized = pipeline_result["summarized"]
        filtered_results = pipeline_result["filtered"]
        clusters = pipeline_result["clusters"]
        
        # Update progress tracker if available
        if self.progress_tracker:
            # Record findings from summaries
            for cluster_id, cluster in summarized["clusters"].items():
                self.progress_tracker.add_finding(
                    summary=cluster["summary_text"],
                    evidence_ids=cluster["evidence_ids"]
                )
            
            # Record iteration state
            self.progress_tracker.record_iteration_state(
                findings=self.progress_tracker.findings_history,
                facets=self._compute_facets(filtered_results),
                suggested_areas=self._get_suggested_areas(summarized),
                token_usage=self.token_manager.get_budget_status()
            )
        
        # Prepare final output
        output = {
            "result_set_id": f"rs_{int(time.time())}",
            "metadata": {
                "total_results": len(filtered_results),
                "clusters": len(clusters),
                "query": query
            },
            "facets": self._compute_facets(filtered_results),
            "summaries": summarized,
            "clusters": clusters
        }
        
        # Cache for drill-down
        self.result_cache[output["result_set_id"]] = {
            "raw_results": results,
            "filtered": filtered_results,
            "clusters": clusters
        }
        
        return output
        
    def _apply_filters(self,
                      results: Dict[str, Any],
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Stage A: Apply upstream filters
        """
        filtered = []
        
        for item in results.get("matches", []):
            if self._passes_filters(item, filters):
                # Extract only needed fields
                filtered.append({
                    "id": f"slack:ch/{item.get('channel')}/{item.get('ts')}",
                    "thread_id": item.get("thread_ts"),
                    "channel": item.get("channel"),
                    "ts": float(item.get("ts", 0)),
                    "author": item.get("user"),
                    "text_preview": self._create_preview(item.get("text", "")),
                    "reactions": item.get("reactions", []),
                    "replies": item.get("reply_count", 0),
                    "text": item.get("text"),  # Keep full text for summarization
                    "is_parent": item.get("ts") == item.get("thread_ts")
                })
        
        return filtered[:self.config.page_size]
        
    def _passes_filters(self, item: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if item passes all filters including content quality"""
        # First check content quality (always applied)
        if not self._passes_content_quality_filter(item):
            return False
            
        if not filters:
            return True
            
        # Time window filter
        if "time_range" in filters:
            ts = float(item.get("ts", 0))
            if ts < filters["time_range"].get("start", 0) or \
               ts > filters["time_range"].get("end", float("inf")):
                return False
                
        # Channel filter
        if "channels" in filters and item.get("channel") not in filters["channels"]:
            return False
            
        # Author filter
        if "authors" in filters and item.get("user") not in filters["authors"]:
            return False
            
        return True
        
    def _passes_content_quality_filter(self, item: Dict[str, Any]) -> bool:
        """Filter out low-quality administrative content"""
        text = item.get("text", "").strip().lower()
        
        # Skip empty or very short messages
        if len(text) < 10:
            return False
            
        # Skip channel join/leave messages
        join_leave_patterns = [
            "has joined the channel",
            "has left the channel", 
            "joined the channel",
            "left the channel",
            "was added to the channel",
            "was removed from the channel",
            "set the channel topic",
            "set the channel purpose",
            "renamed the channel",
            "archived the channel",
            "unarchived the channel"
        ]
        
        for pattern in join_leave_patterns:
            if pattern in text:
                return False
                
        # Skip bot notifications and system messages
        bot_patterns = [
            "this message was deleted",
            "message deleted",
            "uploaded a file",
            "shared a file",
            "started a call",
            "ended a call",
            "reminder:",
            "scheduled for",
            "workflow completed",
            "app installed",
            "integration added"
        ]
        
        for pattern in bot_patterns:
            if pattern in text:
                return False
                
        # Skip messages that are just reactions or acknowledgments
        if text in ["👍", "👎", "ok", "okay", "thanks", "thank you", "got it", "sure", "yes", "no", "lol", "haha"]:
            return False
            
        # Require some technical or substantive content
        # Messages should have either:
        # 1. Technical keywords, OR
        # 2. Questions/discussions (contain "?", "how", "what", "why", "when"), OR  
        # 3. Reasonable length (>30 chars) with engagement (reactions/replies)
        
        technical_keywords = [
            "architecture", "design", "system", "api", "database", "service",
            "implementation", "framework", "library", "code", "function",
            "method", "class", "interface", "protocol", "algorithm", "pattern",
            "infrastructure", "deployment", "configuration", "performance",
            "security", "authentication", "authorization", "testing", "bug",
            "feature", "requirement", "specification", "documentation"
        ]
        
        discussion_indicators = ["?", "how", "what", "why", "when", "where", "which", "should", "could", "would"]
        
        has_technical_content = any(keyword in text for keyword in technical_keywords)
        has_discussion_content = any(indicator in text for indicator in discussion_indicators)
        has_engagement = item.get("reply_count", 0) > 0 or len(item.get("reactions", [])) > 0
        
        # Accept if it has technical content or is a discussion
        if has_technical_content or has_discussion_content:
            return True
            
        # Accept longer messages with engagement (likely substantive)
        if len(text) > 30 and has_engagement:
            return True
            
        # Reject everything else as likely administrative noise
        return False
        
    def _group_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage B: Group results into clusters
        """
        # Group by thread_id
        clusters = defaultdict(list)
        for item in results:
            thread_id = item.get("thread_id") or item.get("ts")
            clusters[thread_id].append(item)
            
        # Compute cluster features
        featured_clusters = {}
        for thread_id, items in clusters.items():
            featured_clusters[thread_id] = {
                "items": items,
                "features": self._compute_cluster_features(thread_id, items)
            }
            
        return featured_clusters
        
    def _compute_cluster_features(self,
                                cluster_id: str,
                                items: List[Dict[str, Any]]) -> ClusterFeatures:
        """Compute features for a cluster"""
        timestamps = [item["ts"] for item in items]
        participants = {item["author"] for item in items}
        total_reactions = sum(len(item["reactions"]) for item in items)
        
        return ClusterFeatures(
            cluster_id=cluster_id,
            size=len(items),
            time_span={
                "start": datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()
            },
            engagement={
                "replies": len(items) - 1,  # Subtract parent
                "reactions": total_reactions,
                "participants": len(participants)
            },
            participants=list(participants),
            channel=items[0]["channel"]
        )
        
    def _rank_and_diversify(self,
                           clusters: Dict[str, Any],
                           query: str) -> Dict[str, Any]:
        """
        Stage C: Rank clusters and diversify items within each
        """
        # Score and rank clusters
        scored_clusters = []
        for cluster_id, cluster in clusters.items():
            cluster_score = self._score_cluster(cluster, query)
            scored_clusters.append((cluster_score, cluster_id))
            
        # Sort clusters by score
        scored_clusters.sort(reverse=True)
        
        # For each cluster, diversify its items
        ranked_clusters = {}
        for _, cluster_id in scored_clusters:
            cluster = clusters[cluster_id]
            items = cluster["items"]
            
            # Score individual items
            scored_items = [(self._score_item(item), item) for item in items]
            
            # Apply MMR to select diverse items
            diverse_items = self._apply_mmr(
                scored_items,
                k=self.config.mmr_items_per_cluster
            )
            
            ranked_clusters[cluster_id] = {
                "items": diverse_items,
                "features": cluster["features"]
            }
            
        return ranked_clusters
        
    def _score_cluster(self, cluster: Dict[str, Any], query: str) -> float:
        """Score a cluster based on its features"""
        features = cluster["features"]
        
        # Normalize time to 0-1 (newer is higher)
        now = time.time()
        time_start = datetime.fromisoformat(features.time_span["start"]).timestamp()
        time_norm = max(0, min(1, 1 - (now - time_start) / (30 * 24 * 3600)))  # 30 days max
        
        # Normalize engagement (log scale)
        engagement = features.engagement
        engagement_norm = min(1, (
            math.log1p(engagement["reactions"]) +
            math.log1p(engagement["replies"]) +
            math.log1p(engagement["participants"])
        ) / 10)  # Arbitrary normalization factor
        
        return (
            0.35 * time_norm +
            0.45 * engagement_norm +
            0.20 * self._compute_query_match(cluster, query)
        )
        
    def _score_item(self, item: Dict[str, Any]) -> float:
        """Score an individual item"""
        # Normalize time
        now = time.time()
        time_norm = max(0, min(1, 1 - (now - item["ts"]) / (30 * 24 * 3600)))
        
        # Normalize engagement
        engagement_norm = min(1, (
            math.log1p(len(item["reactions"])) +
            math.log1p(item["replies"])
        ) / 5)  # Arbitrary normalization factor
        
        return (
            0.35 * time_norm +
            0.25 * engagement_norm +
            0.40 * (1.0 if item["is_parent"] else 0.5)  # Boost thread parents
        )
        
    def _apply_mmr(self,
                   scored_items: List[Tuple[float, Dict[str, Any]]],
                   k: int) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance for diversity"""
        if not scored_items:
            return []
            
        # Sort by score
        scored_items.sort(reverse=True)
        
        selected = [scored_items[0][1]]  # Start with highest scored item
        remaining = scored_items[1:]
        
        while len(selected) < k and remaining:
            # Find item that maximizes MMR
            best_mmr = float("-inf")
            best_idx = -1
            
            for i, (score, item) in enumerate(remaining):
                # Compute similarity to already selected items
                max_sim = max(
                    self._compute_similarity(item, sel)
                    for sel in selected
                )
                
                # MMR formula
                mmr = self.config.mmr_lambda * score - \
                      (1 - self.config.mmr_lambda) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(remaining[best_idx][1])
            remaining.pop(best_idx)
            
        return selected
        
    def _compute_similarity(self,
                          item1: Dict[str, Any],
                          item2: Dict[str, Any]) -> float:
        """
        Compute similarity between two items
        Currently uses simple text overlap, could be enhanced with embeddings
        """
        text1 = item1.get("text", "").lower().split()
        text2 = item2.get("text", "").lower().split()
        
        if not text1 or not text2:
            return 0.0
            
        # Convert to sets for overlap computation
        set1 = set(text1)
        set2 = set(text2)
        
        # Jaccard similarity
        return len(set1 & set2) / len(set1 | set2)
        
    def _compute_query_match(self, cluster: Dict[str, Any], query: str) -> float:
        """
        Compute how well the cluster matches the query
        Currently uses simple term overlap, could be enhanced
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
            
        # Check all items in cluster
        matches = 0
        for item in cluster["items"]:
            text = item.get("text", "").lower()
            matches += sum(1 for term in query_terms if term in text)
            
        return min(1.0, matches / (len(query_terms) * 2))  # Normalize to 0-1
        
    async def _map_reduce_summarize(self, ranked_clusters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage D: Map-Reduce summarization with token limits
        - Per cluster: ≤200 tokens
        - Global brief: ≤400 tokens
        - Total clusters in context: ≤10
        """
        # Reset token budget for new summarization
        self.token_manager.budget.reset()
        
        # First pass: Create cluster summaries within token limits
        cluster_summaries = {}
        for cluster_id, cluster in ranked_clusters.items():
            if len(cluster_summaries) >= self.token_manager.limits.MAX_CLUSTERS:
                logger.warning(f"Reached max clusters limit ({self.token_manager.limits.MAX_CLUSTERS})")
                break
                
            # Create initial summary
            summary = {
                "size": cluster["features"].size,
                "time_span": cluster["features"].time_span,
                "engagement": cluster["features"].engagement,
                "preview": self._create_preview(
                    cluster["items"][0].get("text", "")
                ) if cluster["items"] else ""
            }
            
            # Get evidence IDs
            main_message = cluster["items"][0] if cluster["items"] else None
            thread_ref = main_message.get("thread_id") if main_message else None
            
            evidence_ids = [
                f"slack:ch/{main_message['channel']}/{main_message['ts']}"
                if main_message else None,
                f"slack:th/{thread_ref}" if thread_ref else None
            ]
            evidence_ids = [id for id in evidence_ids if id]  # Remove None
            
            # Generate summary using Gemini
            summary_text = await self.summarizer.summarize_cluster(cluster)
            
            # Enforce cluster token limit
            summary_text = self.token_manager.enforce_cluster_limit(summary_text, cluster_id)
            
            # Store enforced summary with evidence IDs
            cluster_summaries[cluster_id] = {
                **summary,
                "summary_text": summary_text,
                "evidence_ids": evidence_ids  # Store IDs for drill-down
            }
        
        # Create global brief
        global_brief = {
            "cluster_count": len(cluster_summaries),
            "total_messages": sum(
                len(c["items"]) for c in ranked_clusters.values()
            ),
            "time_range": self._get_time_range(ranked_clusters),
            "top_channels": self._get_top_channels(ranked_clusters)
        }
        
        # Create global brief with key findings and evidence IDs
        key_findings = []
        all_evidence_ids = set()  # Track all evidence IDs
        
        # Get top 3 clusters by engagement
        top_clusters = sorted(
            cluster_summaries.items(),
            key=lambda x: (
                x[1]["engagement"]["reactions"] + 
                x[1]["engagement"]["replies"]
            ),
            reverse=True
        )[:3]
        
        # Create bullet points with evidence IDs
        for cluster_id, cluster in top_clusters:
            finding = cluster["summary_text"].split("\n")[0]  # Get first line
            evidence_ids = cluster["evidence_ids"]
            all_evidence_ids.update(evidence_ids)
            key_findings.append(finding)
        
        # Generate global brief using Gemini
        brief_text = await self.summarizer.create_global_brief(
            clusters=cluster_summaries,
            query=self.current_query,
            goal=f"Summarize and analyze: {self.current_query}"
        )
        
        # Enforce global brief token limit
        brief_text = self.token_manager.enforce_global_brief_limit(brief_text)
        
        # Check token budget status
        budget_status = self.token_manager.get_budget_status()
        warnings = self.token_manager.check_limits()
        if warnings:
            for warning in warnings:
                logger.warning(f"Token limit warning: {warning}")
        
        return {
            "global": {
                **global_brief,
                "brief": brief_text,
                "token_status": budget_status,
                "evidence_ids": list(all_evidence_ids)  # All evidence IDs used in brief
            },
            "clusters": cluster_summaries,
            "warnings": warnings
        }
        
    def _get_time_range(self, clusters: Dict[str, Any]) -> Dict[str, str]:
        """Get overall time range for clusters"""
        all_times = []
        for cluster in clusters.values():
            time_span = cluster["features"].time_span
            all_times.extend([
                datetime.fromisoformat(time_span["start"]).timestamp(),
                datetime.fromisoformat(time_span["end"]).timestamp()
            ])
            
        if not all_times:
            return {"start": "", "end": ""}
            
        return {
            "start": datetime.fromtimestamp(min(all_times), tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(max(all_times), tz=timezone.utc).isoformat()
        }
        
    def _get_top_channels(self, clusters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get most active channels"""
        channel_counts = defaultdict(int)
        for cluster in clusters.values():
            channel_counts[cluster["features"].channel] += 1
            
        return [
            {"name": channel, "count": count}
            for channel, count in sorted(
                channel_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
    def _compute_facets(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute result facets"""
        facets = {
            "channels": defaultdict(int),
            "dates": defaultdict(int),
            "authors": defaultdict(int)
        }
        
        for item in results:
            # Channel facets
            facets["channels"][item["channel"]] += 1
            
            # Date facets (group by day)
            date = datetime.fromtimestamp(item["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            facets["dates"][date] += 1
            
            # Author facets
            facets["authors"][item["author"]] += 1
            
        # Convert to sorted lists
        return {
            "channels": [
                {"name": k, "n": v}
                for k, v in sorted(
                    facets["channels"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ],
            "dates": [
                {"day": k, "n": v}
                for k, v in sorted(
                    facets["dates"].items(),
                    key=lambda x: x[0],
                    reverse=True
                )
            ],
            "authors": [
                {"id": k, "n": v}
                for k, v in sorted(
                    facets["authors"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        }
        
    def _create_preview(self, text: str) -> str:
        """Create a preview of text with bounded length"""
        if not text:
            return ""
        if len(text) <= self.config.max_preview_length:
            return text
        return text[:self.config.max_preview_length] + "..."
        
    def _get_suggested_areas(self, summarized: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get suggested areas for exploration based on summaries"""
        suggestions = []
        
        # Check for high-engagement clusters
        for cluster_id, cluster in summarized["clusters"].items():
            engagement = cluster["engagement"]
            if engagement["reactions"] > 5 or engagement["replies"] > 3:
                suggestions.append({
                    "area": f"thread_{cluster_id}",
                    "reason": "High engagement thread",
                    "priority": "high"
                })
                
        # Check for active channels
        top_channels = summarized["global"]["top_channels"]
        if top_channels:
            suggestions.append({
                "area": f"channel_{top_channels[0]['name']}",
                "reason": "Most active channel",
                "priority": "medium"
            })
            
        # Check time range for recency
        time_range = summarized["global"]["time_range"]
        if time_range.get("end"):
            end_time = datetime.fromisoformat(time_range["end"])
            if (datetime.now(timezone.utc) - end_time).days <= 7:
                suggestions.append({
                    "area": "recent_activity",
                    "reason": "Recent activity detected",
                    "priority": "high"
                })
                
        return suggestions
        
    async def drill_down(self,
                        result_set_id: str,
                        item_id: str) -> Optional[Dict[str, Any]]:
        """
        Stage E: Drill-down to get full content
        Enforces evidence token budget (≤3,000 tokens per step)
        """
        if result_set_id not in self.result_cache:
            return None
            
        cached = self.result_cache[result_set_id]
        
        # Find the item and its cluster
        target_item = None
        for item in cached["filtered"]:
            if item["id"] == item_id:
                target_item = item
                break
                
        if not target_item:
            return None
            
        # Get cluster context
        cluster = self._find_cluster(target_item, cached["clusters"])
        if not cluster:
            return None
            
        # Prepare evidence items (target + context)
        evidence_items = [
            {
                "id": target_item["id"],
                "content": target_item.get("text", ""),
                "is_target": True
            }
        ]
        
        # Add thread context if available
        thread_items = [
            {
                "id": item["id"],
                "content": item.get("text", ""),
                "is_target": False
            }
            for item in cluster["items"]
            if item["id"] != target_item["id"]  # Exclude target item
        ]
        evidence_items.extend(thread_items)
        
        # Select evidence within token budget
        # Prioritize target item and immediate context
        selected_evidence = self.token_manager.select_evidence(
            evidence_items,
            prioritize_ids=[target_item["id"]]
        )
        
        # Check if target item made it within budget
        target_included = any(e["id"] == target_item["id"] for e in selected_evidence)
        if not target_included:
            logger.warning(f"Target item {item_id} exceeded evidence token budget")
            return None
            
        # Format response with selected evidence
        return {
            "id": item_id,
            "type": "message",
            "content": target_item.get("text"),
            "thread_id": target_item.get("thread_id"),
            "author": target_item.get("author"),
            "ts": target_item.get("ts"),
            "channel": target_item.get("channel"),
            "reactions": target_item.get("reactions", []),
            "cluster": {
                "id": cluster["features"].cluster_id,
                "size": cluster["features"].size,
                "context": [
                    {
                        "id": e["id"],
                        "content": e["content"]
                    }
                    for e in selected_evidence
                    if not e.get("is_target")  # Exclude target item from context
                ]
            },
            "token_status": {
                "evidence_tokens": self.token_manager.budget.evidence_tokens,
                "limit": self.token_manager.limits.TOTAL_EVIDENCE
            }
        }
        
    def _find_cluster(self,
                     item: Dict[str, Any],
                     clusters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the cluster containing an item"""
        thread_id = item.get("thread_id") or item.get("ts")
        return clusters.get(thread_id)