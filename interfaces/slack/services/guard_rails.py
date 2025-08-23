"""
GuardRails for Progressive Reduction

Implements safety measures and failure mode handling:
- Result explosion control
- Deduplication
- Token budget management
- Ranking enforcement
- Multi-agent collision prevention
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class GuardRailLimits:
    """Hard limits for various guardrails"""
    MAX_PAGES: int = 5
    MAX_CLUSTERS: int = 10
    MAX_RESULTS_PER_CLUSTER: int = 50
    DEDUP_JACCARD_THRESHOLD: float = 0.8
    COMPRESSION_CLUSTER_LIMIT: int = 3
    LOCK_TIMEOUT: int = 30  # seconds

class ResultSetLock:
    """Lock manager for result set processing"""
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        self.owners: Dict[str, str] = {}  # result_set_id -> run_id
        self.timestamps: Dict[str, float] = {}
        
    async def acquire(self, result_set_id: str, run_id: str) -> bool:
        """Try to acquire lock for result set"""
        if result_set_id not in self.locks:
            self.locks[result_set_id] = asyncio.Lock()
            
        try:
            # Try to acquire with timeout
            await asyncio.wait_for(
                self.locks[result_set_id].acquire(),
                timeout=GuardRailLimits.LOCK_TIMEOUT
            )
            self.owners[result_set_id] = run_id
            self.timestamps[result_set_id] = time.time()
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Lock acquisition timeout for result_set_id={result_set_id}, "
                f"run_id={run_id}"
            )
            return False
            
    def release(self, result_set_id: str, run_id: str) -> bool:
        """Release lock if owned by run_id"""
        if (
            result_set_id in self.owners and
            self.owners[result_set_id] == run_id
        ):
            self.locks[result_set_id].release()
            del self.owners[result_set_id]
            del self.timestamps[result_set_id]
            return True
        return False
        
    def cleanup_stale(self):
        """Clean up stale locks"""
        now = time.time()
        stale_ids = [
            result_set_id
            for result_set_id, timestamp in self.timestamps.items()
            if now - timestamp > GuardRailLimits.LOCK_TIMEOUT
        ]
        for result_set_id in stale_ids:
            logger.warning(f"Cleaning up stale lock for result_set_id={result_set_id}")
            if not self.locks[result_set_id].locked():
                del self.locks[result_set_id]
                del self.owners[result_set_id]
                del self.timestamps[result_set_id]

class GuardRails:
    """
    Implements safety measures and failure mode handling
    """
    
    def __init__(self):
        self.limits = GuardRailLimits()
        self.lock_manager = ResultSetLock()
        self.seen_titles = set()  # For deduplication
        
    def _compute_title_hash(self, title: str) -> str:
        """Compute shingled hash of title for deduplication"""
        # Normalize
        title = title.lower().strip()
        words = title.split()
        
        # Create shingles (word pairs)
        shingles = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words)-1)
        ]
        
        # Hash each shingle
        shingle_hashes = [
            hashlib.md5(s.encode()).hexdigest()
            for s in sorted(shingles)
        ]
        
        # Combine hashes
        return hashlib.md5("".join(shingle_hashes).encode()).hexdigest()
        
    def _compute_jaccard_similarity(self, title1: str, title2: str) -> float:
        """Compute Jaccard similarity between normalized titles"""
        # Normalize
        words1 = set(title1.lower().strip().split())
        words2 = set(title2.lower().strip().split())
        
        # Compute Jaccard
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
        
    def check_result_explosion(self,
                             results: Dict[str, Any],
                             current_page: int) -> Dict[str, Any]:
        """
        Handle result explosion
        Returns: Suggested filters based on facets
        """
        total_results = len(results.get("matches", []))
        
        if current_page >= self.limits.MAX_PAGES:
            # Suggest filters based on facets
            suggested_filters = {}
            
            # Time-based filter
            dates = results.get("facets", {}).get("dates", [])
            if dates:
                # Suggest last 7 days if results span longer
                date_range = dates[-1]["day"]  # Oldest
                if (time.time() - time.mktime(time.strptime(date_range, "%Y-%m-%d"))) > 7*24*3600:
                    suggested_filters["time_range"] = {
                        "start": (time.time() - 7*24*3600)
                    }
                    
            # Channel/label filter
            channels = results.get("facets", {}).get("channels", [])
            if channels and len(channels) > 1:
                # Suggest top channel if multiple exist
                suggested_filters["channels"] = [channels[0]["name"]]
                
            logger.warning(
                f"Result explosion: {total_results} results, page {current_page}. "
                f"Suggesting filters: {suggested_filters}"
            )
            
            return suggested_filters
            
        return {}
        
    def deduplicate_results(self,
                          items: List[Dict[str, Any]],
                          field: str = "text") -> List[Dict[str, Any]]:
        """
        Deduplicate items based on field content
        Uses shingled title hash and Jaccard similarity
        """
        unique_items = []
        title_hashes = set()
        
        for item in items:
            title = item.get(field, "")
            title_hash = self._compute_title_hash(title)
            
            # Check hash
            if title_hash in title_hashes:
                continue
                
            # Check Jaccard similarity with existing items
            is_duplicate = False
            for existing in unique_items:
                similarity = self._compute_jaccard_similarity(
                    title,
                    existing.get(field, "")
                )
                if similarity >= self.limits.DEDUP_JACCARD_THRESHOLD:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_items.append(item)
                title_hashes.add(title_hash)
                
        if len(items) > len(unique_items):
            logger.info(
                f"Deduplication removed {len(items) - len(unique_items)} items"
            )
            
        return unique_items
        
    def check_token_budget(self,
                          ctx_estimate: int,
                          budget: int,
                          clusters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle token budget overflow
        Returns: Compressed clusters if needed
        """
        if ctx_estimate <= budget:
            return clusters
            
        logger.warning(
            f"Token budget exceeded: {ctx_estimate} > {budget}. "
            f"Compressing to top {self.limits.COMPRESSION_CLUSTER_LIMIT} clusters"
        )
        
        # Sort clusters by engagement score
        scored_clusters = [
            (self._compute_cluster_score(cluster), cluster_id, cluster)
            for cluster_id, cluster in clusters.items()
        ]
        scored_clusters.sort(reverse=True)
        
        # Keep only top clusters
        compressed = {
            cluster_id: cluster
            for _, cluster_id, cluster in scored_clusters[:self.limits.COMPRESSION_CLUSTER_LIMIT]
        }
        
        return compressed
        
    def _compute_cluster_score(self, cluster: Dict[str, Any]) -> float:
        """Compute engagement score for cluster"""
        features = cluster.get("features")
        if hasattr(features, 'engagement'):
            # ClusterFeatures dataclass - use dot notation
            engagement = features.engagement
            return (
                engagement.get("reactions", 0) +
                engagement.get("replies", 0) * 2 +  # Weight replies more
                engagement.get("participants", 0)
            )
        else:
            # Dict format - use get notation
            engagement = cluster.get("features", {}).get("engagement", {})
            return (
                engagement.get("reactions", 0) +
                engagement.get("replies", 0) * 2 +  # Weight replies more
                engagement.get("participants", 0)
            )
        
    async def process_with_lock(self,
                              result_set_id: str,
                              run_id: str,
                              process_fn: callable,
                              *args,
                              **kwargs) -> Optional[Dict[str, Any]]:
        """
        Process results with locking to prevent multi-agent collisions
        """
        # Try to acquire lock
        if not await self.lock_manager.acquire(result_set_id, run_id):
            return None
            
        try:
            # Process results
            results = await process_fn(*args, **kwargs)
            
            # Add processing metadata
            results["_meta"] = {
                "run_id": run_id,
                "timestamp": time.time()
            }
            
            return results
            
        finally:
            # Always release lock
            self.lock_manager.release(result_set_id, run_id)
            
        # Clean up any stale locks
        self.lock_manager.cleanup_stale()
